import numpy as np
import seaborn as sns
from random import shuffle
from scipy import stats
from scipy.optimize import curve_fit
import cv2
import os
from skimage.morphology import skeletonize
import scipy
from tqdm import tqdm
import random
import matplotlib.pyplot as plt
import click

from config import DIR_GT, DIST, MAX_AREA_TO_REMOVE, SAMPLE_SIZE


def read_mask(path):
    gt = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    try:
        gt = gt[:, :, 3]
    except:
        gt = gt[:, :, 0]
    return gt.astype(bool)


def draw_test(gt, name='test.png'):
    gt = (gt*255).astype(np.uint8)
    cv2.imwrite(name, gt)


def signed_distance_function(gt_stride):
    distance = scipy.ndimage.distance_transform_edt(gt_stride)
    return distance


def get_line_strokes(paths):
    distances_list = []
    for path in tqdm(paths):
        gt = read_mask(path)
        distance_map = signed_distance_function(gt)
        gt_skelet = skeletonize(gt)
        distances = distance_map[gt_skelet]
        distances_list.append(distances)
    return np.concatenate(distances_list, axis=0)


def patch_iterator(gt, patch_size=512, stride=256, skip_empty=True):
    for i in range(int(gt.shape[0]/stride)):
        for j in range(int(gt.shape[1]/stride)):
            gt_stride = gt[
                i*stride:i*stride+patch_size,
                j*stride:j*stride+patch_size
            ]
            if skip_empty and np.any(gt_stride):
                yield gt_stride
            else:
                yield gt_stride


def get_erase(mask, pred, eval):
    mask_orig = mask.copy()
    pred_orig = pred.copy()
    mask = apply_dillation(skeletonize(mask.astype(
        bool).squeeze()), DIST.mean() + 2*DIST.std())
    pred = pred.astype(bool).squeeze()
    pred_skelet = skeletonize(pred)
    false_pos = ~mask.astype(bool).squeeze() & pred_skelet
    edges = get_edges(false_pos, eval).astype(int)
    if edges.sum() == 0:
        raise RuntimeError()
        return get_add(mask_orig, pred_orig, eval)
    return -edges


def get_add(mask, pred, eval):
    mask = mask.astype(bool).squeeze()
    mask_skelet = skeletonize(mask)
    false_neg = mask_skelet & ~pred.astype(bool).squeeze()
    return get_edges(false_neg, eval).astype(int)


def select_neighborhood_indices(data, neighborhood):
    if len(data) < 2*neighborhood:
        return list(range(len(data))), []
    # Choose a random index
    random_index = random.randint(0, len(data) - 1)

    # Calculate the start and end indices for the neighborhood
    start_index = random_index - neighborhood
    end_index = random_index + neighborhood

    # Handle cyclic behavior
    if start_index < 0:
        start_index += len(data)
    if end_index >= len(data):
        end_index -= len(data)

    # Select the neighborhood indices
    if start_index <= end_index:
        neighborhood_indices = list(range(start_index, end_index + 1))
    else:
        neighborhood_indices = list(
            range(start_index, len(data))) + list(range(0, end_index + 1))

    # Select all other indices
    other_indices = [i for i in range(
        len(data)) if i not in neighborhood_indices]
    assert len(other_indices) + len(neighborhood_indices) == len(data)

    return neighborhood_indices, other_indices


def get_edges(input, eval=None):
    # 8 neighbourhood
    kernel = np.array([[1, 1, 1], [1, 10, 1], [1, 1, 1]])

    # apply kernel to mask, do cross correlation
    convolution = scipy.signal.convolve2d(input, kernel, mode='same')
    edges = convolution == 12

    s = [[1, 1, 1], [1, 1, 1], [1, 1, 1]]
    labels, num_labels = scipy.ndimage.label(edges, structure=s)
    edge_list = [labels == i for i in range(1, num_labels + 1)]
    # shuffle(edge_list)

    edge_list.sort(key=lambda x: -np.sum(x))
    concat = np.zeros_like(input, dtype=bool)

    if edge_list.__len__() > 0:
        concat = np.bitwise_or(concat, edge_list[0])

        # select a 2*neighborhood_size + 1 wide line segment
        neighborhood_size = 5
        coords = np.argwhere(concat)
        to_keep, to_remove = select_neighborhood_indices(
            coords, neighborhood_size)
        if len(to_remove) > 0:
            coords = coords[to_remove]
            concat[coords.T[0], coords.T[1]] = False

    return apply_dillation(concat, DIST.mean() - 2*DIST.std())


def apply_dillation(mask, line_thinkness=DIST.mean()):
    return scipy.ndimage.binary_dilation(mask, iterations=round(line_thinkness))


def main():
    root = DIR_GT
    out = 'masks_christian'
    if not os.path.exists(root):
        print(f'Root directory {root} does not exist')
        return
    os.makedirs(out, exist_ok=True)
    paths = [os.path.join(root, file_name) for file_name in os.listdir(root)]
    distances = get_line_strokes(paths)
    histogram_path = os.path.join(out, 'histogram.pdf')

    # Calculate mean
    mean_value = np.mean(distances)
    std_value = np.std(distances)
    distances = list(filter(lambda x: mean_value-2*std_value <=
                     x <= mean_value+2*std_value, distances))

    mean_value = np.mean(distances)
    std_value = np.std(distances)  # Plot histogram with KDE
    sns.histplot(distances, kde=False, bins=8, stat='density',
                 color='tab:blue', edgecolor='None')

    # Plot mean as a vertical line
    plt.axvline(mean_value, color='tab:red', linestyle='--',
                label=f'Mean: {mean_value:.2f}')

    # Fit a Gaussian distribution to the data
    params = stats.gamma.fit(distances)
    x = np.linspace(mean_value-4*std_value, mean_value+4*std_value, 1000)
    pdf = stats.gamma.pdf(x, *params)
    print(f'a:{params[0]}', f'floc:{params[1]}', f'scale:{params[2]}')

    # Plot Gaussian fit
    plt.plot(x, pdf, color='tab:red', linestyle='-', label='Gamma Fit')
    plt.fill_between(x, pdf, color='tab:red', alpha=0.3)

    # Add labels and title
    plt.xlabel('Line Width')
    plt.ylabel('Density')
    plt.legend()

    plt.savefig(histogram_path)

    print('Average line thikness: ', mean_value)
    print('Standard Deviation of line thikness', std_value)

    mask_path = os.path.join(root, 'ANSA-VI-1700_R_drawings.png')
    if not os.path.exists(mask_path):
        print(f'Mask file {mask_path} does not exist')
        return
    mask = read_mask(mask_path)
    original_path = os.path.join(out, 'original.png')
    draw_test(mask, original_path)
    samples = random.sample(list(patch_iterator(mask)), 3)
    for i, sample in enumerate(samples):
        sample = sample.copy()
        sample_path = os.path.join(out, f'sample_{i}.png')
        draw_test(sample, sample_path)
        skelet = skeletonize(sample)
        skelet_path = os.path.join(out, f'sample_{i}_skelet.png')
        draw_test(skelet, skelet_path)
        edges = get_edges(sample)
        edges.sort(key=lambda x: -np.sum(x))
        for j, edge in enumerate(edges[:3]):
            edge_path = os.path.join(out, f'sample_{i}_{j}_edge.png')
            draw_test(edge, edge_path)
            edge = apply_dillation(edge, line_thinkness=3)
            draw_test(edge)


if __name__ == '__main__':
    main()
