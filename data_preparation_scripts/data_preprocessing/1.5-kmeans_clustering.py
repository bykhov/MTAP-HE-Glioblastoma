import os
import pandas as pd
import numpy as np
from skimage import io, color
from pathlib import Path
from PIL import Image, PngImagePlugin
from tqdm import tqdm
from data_preparation_scripts.data_preprocessing.utils import get_svs_path, run_over_slides
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

PngImagePlugin.MAX_TEXT_CHUNK = 2 * 1024 * 1024
Image.MAX_IMAGE_PIXELS = None

def main(svs_dir, slide_path):
    slide_name = Path(slide_path).name.split('.')[0]
    thumbnail_path = os.path.join(svs_dir, slide_name + "_thumbnail.png")
    dest_dir = os.path.join(svs_dir, "clustering_results")
    slide_info_path = os.path.join(svs_dir, "slide_size_info.pickle")
    slide_info_pickle = pd.read_pickle(slide_info_path)
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)

    k_means_clustering(svs_dir, '{:s}/npy_files'.format(dest_dir))

    # project clustering results
    filename = '{:s}/npy_files/cluster.npy'.format(dest_dir)
    clustering_results = np.load(filename, allow_pickle=True).item()
    thumbnail = io.imread(thumbnail_path)
    project_clustering_results(slide_name, clustering_results, thumbnail, '{:s}/png_files'.format(dest_dir), slide_info_pickle)

def k_means_clustering(case_dir, dest_dir):
    os.makedirs(dest_dir, exist_ok=True)

    # ----- perform k-means clustering ----- #
    file_list = os.listdir('{:s}/{:s}'.format(str(case_dir), "patches"))

    features = []
    index_list = []
    for file in file_list:
        index = int(file.split('.')[0])
        image = Image.open('{:s}/{:s}/{:s}'.format(str(case_dir), "patches", file))
        image2 = image.resize((128, 128))   # for 512x512 patch
        image2 = np.array(image2)[:, :, :3]
        
        # Convert to HSV (Hue, Saturation, Value)
        # Saturation is the best feature to separate white background (low S) from tissue (high S)
        hsv_image = color.rgb2hsv(image2)
        
        means = np.mean(hsv_image, axis=(0, 1))
        stds = np.std(hsv_image, axis=(0, 1))
        features.append(np.concatenate((means, stds)))
        index_list.append(index)

    features_array = np.array(features)
    features_normalized = StandardScaler().fit_transform(features_array)
    
    # K=3 to capture: 1. Background, 2. Light Tissue, 3. Dark Tissue
    kmeans = KMeans(n_clusters=3, random_state=0).fit(features_normalized)
    labels = kmeans.labels_
    
    # Logic to identify which cluster is background:
    # The cluster with the LOWEST Mean Saturation (index 1 in HSV features) is the background.
    cluster_saturations = []
    for i in range(3):
        # Get all features belonging to cluster i
        cluster_feats = features_array[labels == i]
        # Handle edge case if a cluster is empty (rare)
        mean_s = np.mean(cluster_feats[:, 1]) if len(cluster_feats) > 0 else 999
        cluster_saturations.append((i, mean_s))
    
    # Sort by saturation. The lowest saturation is the background.
    cluster_saturations.sort(key=lambda x: x[1])
    bg_label = cluster_saturations[0][0]

    index_list = np.array(index_list)
    
    # Group the two tissue clusters together (keep), and background separate (discard)
    cluster_bg = index_list[labels == bg_label]
    cluster_tissue = index_list[labels != bg_label]

    # '0' will be background (dim in visualization), '1' will be tissue (bright in visualization)
    cluster_results = {'0': cluster_bg, '1': cluster_tissue}
    np.save('{:s}/cluster.npy'.format(dest_dir), cluster_results)

def project_clustering_results(slide_name, clustering_results, thumbnail, dest_dir, slide_info_pickle, on_thumbnail=True, color=(255, 255, 255)):
    os.makedirs(dest_dir, exist_ok=True)
    patch_size = 512  # the size of patches extracted from 20x svs image
    target_mag = 20
    slide_info = slide_info_pickle[slide_info_pickle.Slide_name == slide_name]
    mag, h, w = slide_info['Mag'].values[0], slide_info['Height'].values[0], slide_info['Width'].values[0]
    extract_patch_size = int(patch_size * mag / target_mag)

    N_patch_row = h // extract_patch_size
    stride = round(float(thumbnail.shape[0]) / N_patch_row)

    color_mask = np.ones(thumbnail.shape, dtype=float) * 50.0 / 255
    for j in range(0, thumbnail.shape[1], stride):
        for i in range(0, thumbnail.shape[0], stride):
            index = (j//stride) * N_patch_row + (i//stride) + 1
            if index in clustering_results['0']:
                color_mask[i:i + stride, j:j + stride, :] = np.array(color) * 0.5 / 255
            elif index in clustering_results['1']:
                color_mask[i:i + stride, j:j + stride, :] = np.array(color) / 255

    if on_thumbnail:
        thumbnail = thumbnail.astype(float) / 255
        result = thumbnail * color_mask
    else:
        result = color_mask
    io.imsave('{:s}/cluster.png'.format(dest_dir), (result*255).astype(np.uint8))


def is_already_processed(svs_dir: str) -> bool:
    """
    Check whether the svs_dir has already been processed.

    Requirements:
      svs_dir/
         clustering_results/
             npy_files/cluster.npy
             png_files/cluster.png
    """
    clustering_folder = os.path.join(svs_dir, "clustering_results")

    npy_file = os.path.join(clustering_folder, "npy_files", "cluster.npy")
    png_file = os.path.join(clustering_folder, "png_files", "cluster.png")

    return os.path.exists(npy_file) and os.path.exists(png_file)


def process_slide_dir(svs_dir):
    """Cluster the patches of one slide folder (skipped when already processed)."""
    svs_dir = str(svs_dir)
    if is_already_processed(svs_dir):
        print(f"Already processed: {svs_dir}")
        return
    print(f"Processing svs_dir: {svs_dir}")
    main(svs_dir, get_svs_path(svs_dir))


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="k-means (k = 3) tissue selection for every slide under --base-dir.")
    parser.add_argument("--base-dir", required=True, help="raw_data folder with <case_id>/<slide_name>/ folders")
    parser.add_argument("--workers", type=int, default=1, help="number of slides processed in parallel")
    args = parser.parse_args()
    run_over_slides(process_slide_dir, args.base_dir, args.workers)
