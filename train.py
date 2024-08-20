import glob
import click
from utils.gpu import set_cuda_precision
from data.dataloader import EtMirADataLoader
from pytorch_lightning.loggers import WandbLogger
from torchvision.transforms import ToTensor, Compose
from utils.utils import NORMALIZE
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
import pytorch_lightning as pl
from models.unet import UNet
from config import CHANNELS

import os


@click.command()
@click.option('--device', default='gpu', help='accelarator to train on')
@click.option('--gpu', default=0, help='id of gpu')
@click.option('--arch', default='unet', help='decoder architecture')
@click.option('--batch_train', default=32, help='batch size used for training')
@click.option('--batch_val', default=32, help='batch size used for validating')
@click.option('--lr', default=3e-4, help='learning rate')
@click.option('--ckpt', default=None, help='path to checkpoint')
@click.option('--base_model', default='weights/basemodel/epoch=39-pf_measure.ckpt', help='path to basemodel')
@click.option('--enc_str', default='efficientnet-b6', help='type of encoder')
@click.option('--is_interactive', default=False, help='use frozen model for initial prediction')
@click.option('--augment', default=True, help='use augmetation')
@click.option('--patch_size', default=512, help='size of patch')
@click.option('--seed', default=69, help='random seed')
@click.option('--name', default='', help='name of the experiment')
@click.option('--debug', default=False, help='set debug mode')
def train_model(device, gpu, arch, batch_train, batch_val, lr, ckpt, base_model, enc_str, is_interactive, augment, patch_size, seed, name, debug):
    if ckpt is not None:
        if os.path.exists(ckpt) and ".ckpt" in ckpt:
            print("Resume training from checkpoint")
        else:
            raise FileNotFoundError(f"Could not locate checkpoint at {ckpt}")

    pl.seed_everything(seed, workers=True)

    if device == 'gpu':
        set_cuda_precision()

    dataloader = EtMirADataLoader(transform_input=Compose((ToTensor(), NORMALIZE)),
                                  transform_gt=ToTensor(),
                                  use_augment=augment,
                                  patch_size=patch_size,
                                  batch_size_train=batch_train,
                                  batch_size_val=batch_val)
    trainloader = dataloader.get_train_dataloader()
    valloader = dataloader.get_val_dataloader()
    testloader = dataloader.get_test_dataloader()

    if ckpt:
        print(f"Restoring states from {ckpt}")
        unet = UNet.load_from_checkpoint(ckpt, strict=False)
    else:
        unet = UNet(arch, len(CHANNELS), 1, lr, is_interactive, enc_str, patch_size, base_model)

    # start a new wandb run to track this script
    wandb = WandbLogger(entity='rafael-sterzinger',
                        project="iteretmira", name=None if not name else name)
    wandb.experiment.config['channels'] = CHANNELS

    early_stopping = EarlyStopping(monitor='metric/val/mean/pf_gain_over_mask' if is_interactive else 'metric/val/mean/pf_measure',
                                   patience=10,
                                   mode='max',
                                   verbose=True, min_delta=0.001)
    checkpoint_callback = ModelCheckpoint(mode='max', filename='{epoch:02d}-pf_measure',
                                          save_last=True, monitor='metric/val/mean/pf_gain_over_mask' if is_interactive else 'metric/val/mean/pf_measure')

    devices = [gpu]
    gradient_clip = 2  # torch.inf
    trainer = pl.Trainer(max_epochs=-1, logger=wandb, log_every_n_steps=1, accumulate_grad_batches=1, # deterministic=True, # for reproducibility, however, expect slowdowns
                         devices='auto' if device == 'cpu' else devices, accelerator=device, gradient_clip_val=gradient_clip, callbacks=[checkpoint_callback, early_stopping], fast_dev_run=debug)

    trainer.fit(model=unet, train_dataloaders=trainloader,
                val_dataloaders=[valloader]*(5 if is_interactive else 1))

    if not debug:
        ckpt = glob.glob(os.path.join(
            wandb.name, wandb.version, 'checkpoints', '*pf*'))
        ckpt = ckpt[0]
        unet = UNet.load_from_checkpoint(ckpt)
    else:
        assert ckpt
        unet = UNet.load_from_checkpoint(ckpt)
    trainer.test(model=unet, dataloaders=[testloader]*(5 if is_interactive else 1))


if __name__ == '__main__':
    train_model()
