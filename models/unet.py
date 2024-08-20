from collections import defaultdict
import random
from data.graph import apply_dillation, get_add, get_erase
from models.loss import PseudoFMeasure
from typing import Any, Callable
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.core.optimizer import LightningOptimizer
import torch
from torch.optim.optimizer import Optimizer

import wandb

from torch import optim, nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torchmetrics.classification as metrics
from torchmetrics.aggregation import MeanMetric
from config import CHANNELS, DIST, LINE_MEAN, LINE_STD, MapType, get_input_slice
from utils.logging import log_image
import segmentation_models_pytorch as smp
from segmentation_models_pytorch.losses import DiceLoss
from utils.utils import UNNORMALIZE, get_product

archs = [
    'unet',
    'unetplusplus',
    'manet',
    'linknet',
    'fpn',
    'pspnet',
    'deeplabv3',
    'deeplabv3plus',
    'pan'
]

IMAGE_TO_LOG = 5


class UNet(pl.LightningModule):
    def __init__(self, arch: str, in_channels: int, out_channels: int, lr: float, is_interactive: False, enc_str: str = None, patch_size=512, base_model='weights/basemodel/epoch=35-pf_measure.ckpt'):
        """
        :param in_channels: number of channels in the input image
        :param out_channels: number of channels in the result feature map
        """
        super().__init__()

        self.lr = lr

        aux_params = {'dropout': 0.0}

        self.model = smp.create_model(
            arch, enc_str, None, in_channels + 2*is_interactive, out_channels, aux_params=aux_params)

        self.freeze = is_interactive
        if self.freeze:
            self.model_init = UNet.load_from_checkpoint(
                base_model, freeze=False, strict=False, map_location=self.device)
            self.model_init.freeze = False
            for param in self.model_init.parameters():
                param.requires_grad = False

        self.loss = DiceLoss(mode='binary')

        self.batch_to_log = (None, -1)

        self.metrics = nn.ModuleDict({
            'pf_measure': PseudoFMeasure(),
            'jaccard_index': metrics.BinaryJaccardIndex(),
            'dice': metrics.Dice(),
            'f1': metrics.BinaryF1Score(),
            'auroc': metrics.BinaryAUROC()
        })

        self.loss_metric = MeanMetric()

        if self.freeze:
            self.human_input_metric = MeanMetric()
            self.pf_measure_init = PseudoFMeasure()
            self.pf_measure_mask = PseudoFMeasure()
        self.patch_size = patch_size
        self.save_hyperparameters()

    def forward(self, x: torch.Tensor, gt: torch.Tensor = None, y_init=None, add=True, delete=True, human_input=None, y_cur=None):
        output = {}
        if self.freeze:
            if y_init is None:
                with torch.no_grad():
                    if self.model_init.training:
                        self.model_init.eval()
                    y_init = self.model_init(
                        x, gt)['y_after'].clone().detach()
            output['y_init'] = y_init
            y_init = y_init.cpu().numpy() >= 0
            np_gt = gt.cpu().numpy()
            init_mask = torch.zeros(
                (x.shape[0], 1, x.shape[2], x.shape[3])).to(x.device)
            x = torch.cat((x, output['y_init'], init_mask), dim=1)

            if human_input is not None:
                y_cur = y_cur.numpy() >= 0
                y_cur[human_input == 1] = 1
                y_cur[human_input == -1] = 0
                human_input_new = self.get_human_input(
                    x, np_gt, y_cur, add, delete).to(x.device)
                human_input[human_input ==
                            0] = human_input_new[human_input == 0]
            else:
                human_input = self.get_human_input(
                    x, np_gt, y_init, add, delete).to(x.device)

            x[:, -1:, :, :] = human_input.clone()
            output['human_input'] = human_input.clone().cpu().squeeze()
            y_middle = output['y_init'].clone()
            y_middle[human_input == -1] = -torch.inf
            y_middle[human_input == 1] = torch.inf
            output['y_middle'] = y_middle
        y = self.model(x)
        y = y[0] if type(y) is tuple else y
        output['y_after'] = y
        return output

    def get_human_input(self, x, np_gt, np_y, add=True, delete=True):
        if add and delete:
                try:
                    new_input = torch.stack([torch.Tensor(get_add(np_gt[i], np_y[i], not self.training) if random.random() < 0.5 else get_erase(np_gt[i], np_y[i], not self.training))
                                             for i in range(x.shape[0])]).unsqueeze(dim=1)
                except RuntimeError:
                    new_input = torch.stack([torch.Tensor(get_add(np_gt[i], np_y[i], not self.training)) for i in range(x.shape[0])]).unsqueeze(dim=1)
        elif add:
                new_input = torch.stack([torch.Tensor(get_add(
                    np_gt[i], np_y[i], not self.training)) for i in range(x.shape[0])]).unsqueeze(dim=1)
        elif delete:
                new_input = torch.stack([torch.Tensor(get_erase(np_gt[i], np_y[i], not self.training))
                                         for i in range(x.shape[0])]).unsqueeze(dim=1)
        else:
                raise RuntimeError('Need to specify either add or delete')
        return new_input

    def training_step(self, batch, batch_idx):
        x, y_gt = batch
        output = self(x, y_gt)

        loss = self.loss(output['y_after'], y_gt)

        if batch_idx == 0:
            image = None
            if self.freeze:
                image = self.create_image(
                    x[:IMAGE_TO_LOG], y_gt[:IMAGE_TO_LOG], output['y_after'][:IMAGE_TO_LOG], output['human_input'][:IMAGE_TO_LOG], output['y_init'][:IMAGE_TO_LOG].clone().squeeze())
            else:
                image = self.create_image(
                    x[:IMAGE_TO_LOG], y_gt[:IMAGE_TO_LOG], output['y_after'][:IMAGE_TO_LOG])
            self.logger.experiment.log({f'visualization/train': wandb.Image(
                image)})

        self.log('train/loss', loss.item(), on_step=True,
                 on_epoch=True, prog_bar=True, sync_dist=True)
        if self.freeze:
            self.log('train/human_input', (output['human_input'] != 0).sum()/get_product(y_gt.shape), on_step=True,
                     on_epoch=True, sync_dist=True)

        return loss

    def log_images(self):
        x, target = self.batch_to_log[0]
        sorting = reversed(torch.argsort(
            target.reshape((target.shape[0], -1)).sum(-1)))[:IMAGE_TO_LOG]
        x, target = x[sorting], target[sorting]

        model_device = next(self.parameters()).device
        if x.device != model_device:
            x = x.to(model_device)
            target = target.to(model_device)

        with torch.no_grad():
            output = self(x, target)
            preds = output['y_after']
            image = None
            if self.freeze:
                image = self.create_image(x, target, preds,
                                          output['human_input'], output['y_init'].clone().squeeze())
            else:
                image = self.create_image(x, target, preds)
            self.logger.experiment.log({f'visualization/val': wandb.Image(
                image)})

    def create_image(self, x, target, preds, human_input=None, y_init=None):
        x = UNNORMALIZE(x)
        channels = get_input_slice([MapType.NX, MapType.NY, MapType.NZ]
                                   ) if MapType.NX in CHANNELS else get_input_slice([MapType.DEPTH])
        x = x[:, channels].clone().detach()
        preds = preds.clone().detach().squeeze(1)
        target = target.clone().detach().squeeze(1)
        preds = torch.round(torch.sigmoid(preds))
        if y_init is not None:
            y_init = torch.round(torch.sigmoid(y_init))

        return log_image(x, target, preds, None if human_input is None else (human_input.unsqueeze(0) if human_input.shape.__len__() == 2 else human_input), None if y_init is None else (y_init.squeeze(0) if y_init.shape.__len__() == 4 else y_init))

    def on_validation_epoch_start(self) -> None:
        self.init_eval()
        return super().on_validation_epoch_start()

    def on_test_epoch_start(self) -> None:
        self.init_eval()
        return super().on_test_epoch_start()

    def init_eval(self):
        self.metric_storage = defaultdict(lambda: [])
        self.dataset_id = 0

    def validation_step(self, batch, _, dataloader_idx=0):
        if dataloader_idx != self.dataset_id:
            self.store_metrics()
            self.dataset_id = dataloader_idx

        x, y_gt = batch

        # select batch with most annotations to log
        if self.trainer.current_epoch == 0 and y_gt.sum() > self.batch_to_log[1]:
            self.batch_to_log = (batch, y_gt.sum())

        output = self(x, y_gt)
        loss = self.loss(output['y_after'], y_gt)
        self.loss_metric.update(loss.item())
        if self.freeze:
            self.human_input_metric.update(
                (output['human_input'] != 0).sum()/get_product(y_gt.shape))
        self.calc_metrics(y_gt, output)

    def store_metrics(self):
        self.metric_storage['loss'].append(self.loss_metric.compute().item())
        self.loss_metric.reset()
        for metric_name, metric in self.metrics.items():
            self.metric_storage[metric_name].append(metric.compute().item())
            metric.reset()

        if self.freeze:
            self.metric_storage['human_input'].append(
                self.human_input_metric.compute().item())
            self.human_input_metric.reset()
            self.metric_storage['pf_measure_init'].append(self.pf_measure_init.compute(
            ).item())
            self.pf_measure_init.reset()
            self.metric_storage['pf_measure_mask'].append(self.pf_measure_mask.compute(
            ).item())
            self.pf_measure_mask.reset()

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        return self.validation_step(batch, batch_idx, dataloader_idx=dataloader_idx)

    def calc_metrics(self, target, output):
        preds = output['y_after']
        preds = preds.squeeze()
        target = target.squeeze().round().int()

        for metric in self.metrics.values():
            metric.update(preds, target)

        if self.freeze:
            self.pf_measure_init.update(output['y_init'].squeeze(), target)
            self.pf_measure_mask.update(output['y_middle'].squeeze(), target)

    def log_metrics(self, name):
        for metric_name, values in self.metric_storage.items():
            self.log(f"metric/{name}/mean/{metric_name}",
                     np.mean(values), on_epoch=True)
            self.log(f"metric/{name}/std/{metric_name}",
                     np.std(values), on_epoch=True)
        if self.freeze:
            initial = np.asarray(self.metric_storage['pf_measure_init'])
            with_mask = np.asarray(self.metric_storage['pf_measure_mask'])
            final = np.asarray(self.metric_storage['pf_measure'])
            self.log(f"metric/{name}/mean/pf_gain_over_init",
                     np.mean(final/initial), on_epoch=True)
            self.log(f"metric/{name}/std/pf_gain_over_init",
                     np.std(final/initial), on_epoch=True)
            self.log(f"metric/{name}/mean/pf_gain_over_mask",
                     np.mean(final/with_mask), on_epoch=True, prog_bar=True)
            self.log(f"metric/{name}/std/pf_gain_over_mask",
                     np.std(final/with_mask), on_epoch=True)

    def on_validation_epoch_end(self) -> None:
        self.store_metrics()
        self.log_metrics(name='val')
        self.log_images()
        return super().on_validation_epoch_end()

    def on_test_epoch_end(self) -> None:
        self.store_metrics()
        self.log_metrics(name='test')
        return super().on_test_epoch_end()

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr)

        scheduler = {
            'scheduler': ReduceLROnPlateau(optimizer, mode='max', patience=5, factor=0.5, verbose=True),
            'monitor': 'metric/val/mean/pf_gain_over_mask' if self.freeze else 'metric/val/mean/pf_measure',
        }

        return [optimizer], [scheduler]
