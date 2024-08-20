import re
from PIL import Image
import pandas as pd
from data.dataset import EtMirADatasetValidation
import click
import heapq
import os
import cv2
import torch

import numpy as np
from config import DIR_GT, DIR_MASKS, DIR_ROOT, INIT_DIM, SAMPLE_SIZE, VAL_PATH
from models.unet import UNet
from torchvision.transforms import ToTensor, Compose
from torch.utils.data import DataLoader
from models.loss import PseudoFMeasure

from config import REGEX_MASK
from utils.utils import NORMALIZE, create_weight_map, get_iou, get_pf, get_predicted_patches, get_soft_image, inverse_sigmoid, stich_image, visualize_overlap
import pytorch_lightning as pl

EVALUATE_MIRRORS = [
    "ANSA-VI-1700_R"
    "ANSA-VI-1701_R"
    "wels-11944_R"
]


@click.command()
@click.option('--device', default=0, help='accelarator to train on')
@click.option('--ckpt', default="weights/itermodel/epoch=55-pf_measure.ckpt", help='path to checkpoint', required=True)
@click.option('--base_model', default='weights/basemodel/epoch=39-pf_measure.ckpt', help='path to base model checkpoint', required=True)
@click.option('--is_interactive', default=True, help='whether prediction is interactive or not', required=True)
@click.option('--mirror', default="ANSA-VI-1701_R", help='specifies the mirror on which the simulation should be run', required=True)
@click.option('--seed', default=0, help='seed', required=True, type=int)
def eval_model(device, ckpt, base_model, is_interactive, mirror, seed):
    pl.seed_everything(seed, workers=True)
    torch.set_num_threads(8)
    EVALUATE_MIRRORS = [mirror]

    if os.path.exists(ckpt) and ".ckpt" in ckpt:
        print(f"Start evaluating checkpoint: {ckpt}")
    else:
        raise FileNotFoundError(f"Could not locate checkpoint at {ckpt}")

    device = f'cuda:{device}'
    patch_size = None
    try:
        patch_size = torch.load(ckpt, map_location=device)['patch_size']
    except:
        patch_size = 512

    unet = UNet.load_from_checkpoint(
        base_model, freeze=False, strict=False, map_location=device)
    unet.to(device)
    unet.eval()

    weight_map = create_weight_map(0.5)
    pfmeasure = PseudoFMeasure()

    for valid_mirror in sorted(os.listdir(DIR_GT)):
        res = re.search(REGEX_MASK, valid_mirror)
        if res:
            view_id = res.group(1)
            if view_id not in EVALUATE_MIRRORS:
                continue
            dir_eval = os.path.join('out', view_id)
            if not os.path.exists(dir_eval):
                os.makedirs(dir_eval)
            f_gt = os.path.join(DIR_GT, valid_mirror)
            f_mask = os.path.join(DIR_MASKS, f"{view_id}_mask.png")
            assert os.path.exists(f_mask)
            obj_id = res.group(2)
            dir_obj = os.path.join(DIR_ROOT, obj_id, 'PS')
            if os.path.exists(dir_obj):
                f_normals = os.path.join(dir_obj, f'{view_id}_N.tif')
                f_albedo = os.path.join(dir_obj, f'{view_id}_RHO.tif')
                f_depth = os.path.join(dir_obj, f'{view_id}_U.tif')
                if os.path.isfile(f_normals) and os.path.isfile(f_albedo) and os.path.isfile(f_depth):
                    print(f'Create instance for {view_id}')
                else:
                    print(f"Could not locate normal, albedo, or depth map")
                    continue
            else:
                raise FileNotFoundError(
                    f"Could not locate mirror at {dir_obj}")

            data = []

            dataset = DataLoader(EtMirADatasetValidation(VAL_PATH, transform_input=Compose([ToTensor(), NORMALIZE]), transform_gt=ToTensor(
            ), patch_size=patch_size, return_coords=True, id=view_id), batch_size=256, shuffle=False, prefetch_factor=2, num_workers=8)

            dim = np.array(INIT_DIM)//(patch_size//SAMPLE_SIZE)
            dim = np.array([dim[1], dim[0]])
            gt = cv2.imread(f_gt, cv2.IMREAD_UNCHANGED)[
                :, :, 3]
            gt = cv2.resize(
                gt, dim, interpolation=cv2.INTER_NEAREST).astype(bool)

            mask = cv2.imread(f_mask, cv2.IMREAD_UNCHANGED)[
                :, :, 2]
            mask = cv2.resize(
                mask, dim, interpolation=cv2.INTER_NEAREST).astype(bool)

            coords, inputs, outputs, gts = get_predicted_patches(
                device, unet, dataset)
            init_output = outputs.copy()
            init_prediction = stich_image(
                weight_map, patch_size, coords, inputs, outputs)

            image_hard, entry = visualize_and_evaluate(
                dir_eval, view_id, gt, mask, init_prediction, iter=0, input=False, visualize=True)
            entry['iou_human'] = entry['iou']
            entry['pfm_human'] = entry['pfm']
            entry['human_input'] = 0
            print(f'{view_id}: iou={entry["iou_human"]}, pfm={entry["pfm_human"]}')

            data.append(to_numpy(entry))

            if is_interactive:
                del unet
                unet = UNet.load_from_checkpoint(
                    ckpt, map_location='cpu', strict=False)
                unet.to('cpu')
                unet.eval()
                priority_queue = []
                human_patch_pfm = {}
                for patch_id in range(inputs.shape[0]):
                    pfm = pfmeasure(torch.Tensor(
                        outputs[patch_id]), torch.Tensor(gts[patch_id])).item()
                    heapq.heappush(priority_queue, (pfm, patch_id))
                    human_patch_pfm[patch_id] = pfm
                del patch_id

                human_input = np.zeros_like(image_hard)

                for iter in range(3000):
                    while True:
                        if len(priority_queue) == 0:
                            break
                        init_pfm, wrong_patch = heapq.heappop(
                            priority_queue)
                        x_temp = torch.Tensor(inputs[wrong_patch]).unsqueeze(0)
                        gt_temp = torch.Tensor(gts[wrong_patch]).unsqueeze(0)
                        y_cur = torch.Tensor(
                            outputs[wrong_patch]).unsqueeze(0)
                        y_init = torch.Tensor(
                            init_output[wrong_patch]).unsqueeze(0)

                        coord = coords[:,
                                       wrong_patch]//(patch_size//SAMPLE_SIZE)
                        x_min, y_min = coord[1], coord[0]
                        x_max, y_max = (
                            gt_temp.shape[-2:] + np.array([x_min, y_min]))
                        human_input_temp = torch.Tensor(
                            human_input[x_min:x_max, y_min:y_max]).clone().unsqueeze(0).unsqueeze(0)
                        save_orig = False
                        if human_input_temp.sum() == 0:
                            save_orig = False

                        with torch.no_grad():
                            if gt_temp.sum() != 0:
                                # returns full human input and current prediction
                                patch_w_addition = unet(
                                    x_temp, gt_temp, y_init, add=True, delete=False, human_input=human_input_temp.clone(), y_cur=y_cur)

                                add_output = patch_w_addition['y_after'].clone(
                                ).squeeze()
                                add_output[patch_w_addition['human_input']
                                           == 1] = np.inf
                                add_output[patch_w_addition['human_input']
                                           == -1] = -np.inf
                                add_pf = pfmeasure(
                                    add_output.unsqueeze(0), torch.Tensor(gts[wrong_patch])).item()

                                try:
                                    patch_w_subtraction = unet(
                                        x_temp, gt_temp,  y_init, add=False, delete=True, human_input=human_input_temp.clone(), y_cur=y_cur)
                                    sub_output = patch_w_subtraction['y_after'].clone(
                                    ).squeeze()
                                    sub_output[patch_w_subtraction['human_input']
                                               == 1] = np.inf
                                    sub_output[patch_w_subtraction['human_input']
                                               == -1] = -np.inf

                                    sub_pf = pfmeasure(
                                        sub_output.unsqueeze(0), torch.Tensor(gts[wrong_patch])).item()
                                except:
                                    sub_pf = -1

                                if add_pf > sub_pf:
                                    human_and_init = y_init.clone().squeeze()
                                    human_and_init[patch_w_addition['human_input']
                                                   == 1] = np.inf
                                    human_and_init[patch_w_addition['human_input']
                                                   == -1] = -np.inf
                                    gain_patch_human = pfmeasure(
                                        human_and_init.unsqueeze(0), torch.Tensor(gts[wrong_patch])).item()

                                    if np.isclose(add_pf, init_pfm):
                                        continue
                                    if gain_patch_human > human_patch_pfm[wrong_patch]:
                                        human_patch_pfm[wrong_patch] = gain_patch_human
                                    else:
                                        continue

                                    if seed == 0:
                                        if save_orig:
                                            image = unet.create_image(
                                                x_temp, gt_temp, y_init, torch.zeros_like(patch_w_addition['human_input']), y_init)
                                            image.save(
                                                f'{dir_eval}/000_{wrong_patch}.png')
                                        image = unet.create_image(
                                            x_temp, gt_temp, patch_w_addition['y_after'], patch_w_addition['human_input'], patch_w_addition['y_init'])
                                        image.save(
                                            f'{dir_eval}/{iter}_{wrong_patch}.png')
                                    outputs[wrong_patch] = patch_w_addition['y_after']
                                    human_input[x_min:x_max,
                                                y_min:y_max] = patch_w_addition['human_input']

                                    heapq.heappush(priority_queue,
                                                   (add_pf, wrong_patch))
                                else:
                                    human_and_init = y_init.clone().squeeze()
                                    human_and_init[patch_w_subtraction['human_input']
                                                   == 1] = np.inf
                                    human_and_init[patch_w_subtraction['human_input']
                                                   == -1] = -np.inf
                                    gain_patch_human = pfmeasure(
                                        human_and_init.unsqueeze(0), torch.Tensor(gts[wrong_patch])).item()
                                    if np.isclose(sub_pf, init_pfm):
                                        continue
                                    if gain_patch_human > human_patch_pfm[wrong_patch]:
                                        human_patch_pfm[wrong_patch] = gain_patch_human
                                    else:
                                        continue

                                    if seed == 0:
                                        if save_orig:
                                            image = unet.create_image(
                                                x_temp, gt_temp, y_init, torch.zeros_like(patch_w_addition['human_input']), y_init)
                                            image.save(
                                                f'{dir_eval}/000_{wrong_patch}.png')
                                        image = unet.create_image(
                                            x_temp, gt_temp, patch_w_subtraction['y_after'], patch_w_subtraction['human_input'], patch_w_subtraction['y_init'])
                                        image.save(
                                            f'{dir_eval}/{iter}_{wrong_patch}.png')
                                    outputs[wrong_patch] = patch_w_subtraction['y_after']
                                    human_input[x_min:x_max,
                                                y_min:y_max] = patch_w_subtraction['human_input']
                                    heapq.heappush(priority_queue,
                                                   (sub_pf, wrong_patch))
                            else:
                                continue

                            if iter % 10 == 0:
                                init_prediction_temp = init_prediction.copy()
                                init_prediction_temp[human_input == 1] = 1
                                init_prediction_temp[human_input == -1] = 0

                                # human input evaluation
                                image_hard, entry = visualize_and_evaluate(
                                    dir_eval, view_id, gt, mask, init_prediction_temp, iter=iter, input=False, visualize=iter % 200 == 0)

                                print(iter, wrong_patch,
                                      f"pfm_mask: {entry['pfm']}")

                                prediction = stich_image(
                                    weight_map, patch_size, coords, inputs, outputs)

                                prediction[human_input == 1] = 1
                                prediction[human_input == -1] = 0

                                image_hard, entry_after = visualize_and_evaluate(
                                    dir_eval, view_id, gt, mask, prediction, iter=iter, input=False, visualize=iter % 200 == 0)

                                print(iter, wrong_patch,
                                      f"pfm_after: {entry_after['pfm']}")

                                entry_after['iou_human'] = entry['iou']
                                entry_after['pfm_human'] = entry['pfm']
                                entry_after['human_input'] = (
                                    human_input != 0).sum()
                                data.append(to_numpy(entry_after))
                                pd.DataFrame(data).to_csv(
                                    os.path.join(dir_eval, f'{view_id}_{seed}.csv'))
                        break


def to_numpy(entry):
    return {key: value.item() if type(value) == torch.Tensor else value for key, value in entry.items()}


def visualize_and_evaluate(dir_eval, view_id, gt, mask, prediction, iter, input, visualize=False):
    orig_dim, image_soft = get_soft_image(prediction)
    image_soft[~mask] = 0
    if visualize:
        cv2.imwrite(os.path.join(
                    dir_eval, f"{view_id}_soft_{'mask_' if input else ''}{iter}.png"), (image_soft).astype(np.uint8))
    image_soft = image_soft/255.0

    entry = {}

    entry['iou'] = get_iou(image_soft, gt)
    entry['pfm'] = get_pf(image_soft, gt)

    image_hard = np.where(image_soft >= 0.5,  np.ones_like(
        image_soft, dtype=bool), np.zeros_like(image_soft, dtype=bool))
    if visualize:
        cv2.imwrite(os.path.join(
            dir_eval, f"{view_id}_drawings_{'mask_' if input else ''}{iter}.png"), (image_hard*255).astype(np.uint8))

    if visualize:
        image_error = visualize_overlap(
            image_hard, gt)
        cv2.imwrite(os.path.join(
            dir_eval, f"{view_id}_error_{'mask_' if input else ''}{iter}.png"), image_error)

    return image_hard/255, entry


if __name__ == '__main__':
    eval_model()
