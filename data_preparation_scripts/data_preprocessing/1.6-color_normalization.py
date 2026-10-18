import argparse
import os
from pathlib import Path

import stainNorm_Reinhard
from skimage import color, io
import numpy as np
from tqdm import tqdm
import random

from data_preparation_scripts.data_preprocessing.utils import get_svs_path


def compute_mean_std(data_path, file_list, ratio=1.0, seed=0):
    """
    Mean and std of the LAB channels over a random sample (``ratio``) of the patches in ``file_list``.

    The sample is drawn with ``random.Random(seed)`` and ``file_list`` is not modified, so the same
    seed gives the same statistics.
    """
    total_sum = np.zeros(3)  # total sum of all pixel values in each channel
    total_square_sum = np.zeros(3)
    num_pixel = 0  # total num of all pixels

    N = len(file_list)
    print(f"Total files in list: {N}")
    rng = random.Random(seed)
    files_to_be_processed = rng.sample(list(file_list), int(ratio * N))
    print(f"Files to be processed: {len(files_to_be_processed)}")

    for file_name in tqdm(files_to_be_processed, "patches are being processed for getting the mean and std"):
        img_name = os.path.join(data_path, file_name)
        img = io.imread(img_name)
        if len(img.shape) != 3 or img.shape[2] < 3:
            continue
        img = img[:, :, :3]

        img = color.rgb2lab(img)

        total_sum += img.sum(axis=(0, 1))
        total_square_sum += (img ** 2).sum(axis=(0, 1))
        num_pixel += img.shape[0] * img.shape[1]

    # compute the mean values of each channel
    mean_values = total_sum / num_pixel
    # compute the standard deviation
    std_values = np.sqrt(total_square_sum / num_pixel - mean_values ** 2)

    return mean_values, std_values


def filter_tumor_patches(tumor_patches_indices, ref_patches_list):
    # Convert all relevant indices into the corresponding filenames
    target_filenames = {f"{idx}.jpg" for idx in tumor_patches_indices}
    # Keep only patches whose names match the target filenames
    filtered_ref_patches = [
        fname for fname in ref_patches_list
        if fname in target_filenames
    ]
    return filtered_ref_patches

def get_color_normalizer(ref_case_dir, seed=0):
    ref_slide_patches_path = os.path.join(ref_case_dir, "patches")
    tumor_patches_indices = get_tumor_patches(ref_case_dir)
    # compute the mean and std of the reference slide
    mean_std_file_path = os.path.join(ref_case_dir, "color_norm_mean_std.npy")
    ref_patches_list = os.listdir(ref_slide_patches_path)
    filtered_ref_patches = filter_tumor_patches(tumor_patches_indices, ref_patches_list)
    means, stds = compute_mean_std(ref_slide_patches_path, filtered_ref_patches, ratio=0.2, seed=seed)
    np.save(mean_std_file_path, np.array([means, stds]))
    # initialize the normalizer
    normalizer = stainNorm_Reinhard.normalizer(means, stds)
    return mean_std_file_path, normalizer

def get_tumor_patches(ref_case_dir):
    all_patches_indices_path = os.path.join(ref_case_dir, "clustering_results", "npy_files", "cluster.npy")
    all_patches_indices = np.load(all_patches_indices_path, allow_pickle=True).item()
    tumor_patches_indices = all_patches_indices['1']
    return tumor_patches_indices

def normalize_patches(case_dir, normalizer: stainNorm_Reinhard.normalizer, seed=0):
    slide_patches_path = os.path.join(case_dir, "patches")
    target_dir = os.path.join(case_dir, "20xNormalizedPatches")
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    case_patches_list = os.listdir(slide_patches_path)
    tumor_patches_indices = get_tumor_patches(case_dir)
    filtered_patches = filter_tumor_patches(tumor_patches_indices, case_patches_list)
    print(f"Calculating mean and std for: {os.path.basename(case_dir)}")
    slide_mean, slide_std = compute_mean_std(slide_patches_path, filtered_patches, ratio=0.2, seed=seed)

    for i in tqdm(range(len(filtered_patches)), desc="Normalizing patches"):
        img_name = filtered_patches[i]
        img_path = os.path.join(case_dir, "patches", str(img_name))
        img = io.imread(img_path)
        img = img[:, :, :3]
        # perform reinhard color normalization
        img_normalized = normalizer.transform(img, slide_mean, slide_std)
        img_index = img_name[:-4]
        target_file_path = os.path.join(case_dir, "20xNormalizedPatches", f"{img_index}.jpg")
        io.imsave(target_file_path, img_normalized, quality=95)

def normalize_slide_dir(svs_dir, normalizer, seed=0):
    """Normalize one slide folder; a marker file lets an interrupted run skip slides that are finished."""
    done = Path(svs_dir) / "20xNormalizedPatches" / ".done"
    if done.exists():
        print(f"already normalized: {svs_dir}")
        return
    print(f"Processing svs_dir: {svs_dir}")
    normalize_patches(svs_dir, normalizer, seed=seed)
    done.write_text("done")


if __name__ == '__main__':
    from functools import partial

    from data_preparation_scripts.data_preprocessing.utils import run_over_slides

    parser = argparse.ArgumentParser(description="Reinhard stain normalization of the tissue patches.")
    parser.add_argument("--ref-case-dir", required=True,
                        help="Slide folder of the reference slide (the paper used TCGA-12-0819-01Z-00-DX1).")
    parser.add_argument("--base-dir", required=True, help="raw_data folder with <case_id>/<slide_name>/ folders.")
    parser.add_argument("--seed", type=int, default=0,
                        help="Seed of the random sample of patches used to estimate the LAB statistics.")
    parser.add_argument("--ref-stats", default=None,
                        help="precomputed reference statistics (color_norm_mean_std.npy) to use instead of "
                             "computing them from the reference slide.")
    parser.add_argument("--workers", type=int, default=1, help="number of slides processed in parallel")
    args = parser.parse_args()

    if args.ref_stats:
        means, stds = np.load(args.ref_stats)
        normalizer = stainNorm_Reinhard.normalizer(means, stds)
    else:
        _, normalizer = get_color_normalizer(args.ref_case_dir, seed=args.seed)

    run_over_slides(partial(normalize_slide_dir, normalizer=normalizer, seed=args.seed), args.base_dir, args.workers)
