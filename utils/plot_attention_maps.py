"""Attention map of one slide: per-tile attention scores (attention_maps/<patient_id>.csv written during testing)
drawn at the tile positions of the coordinate table.

    python utils/plot_attention_maps.py --base_dir <slide folder with the .svs> \
        --attention_scores_path <outputs/<run>/fold<k>/attention_maps/<patient_id>.csv> [--output_dir <dir>]
"""
import pandas as pd
import openslide
import matplotlib.pyplot as plt
import numpy as np
import argparse
from pathlib import Path
import glob
import os


def plot_attention_map(slide_path, coordinates_path, attention_scores_path, output_path):
    """
    Plots the attention map on a slide, dynamically adjusting patch size based on magnification.
    """
    # Load data
    slide = openslide.OpenSlide(slide_path)
    slide_id = Path(slide_path).stem.split('.')[0]
    print(f"Processing slide: {slide_id}")

    # --- Data Loading and Filtering ---
    all_patch_coords = pd.read_parquet(coordinates_path)
    patch_coords = all_patch_coords[all_patch_coords['slide_id'] == slide_id].copy()
    print(f"Found {len(patch_coords)} coordinates for this slide in the parquet file.")
    if patch_coords.empty:
        print("Warning: No coordinates found for this slide. Cannot generate heatmap.")
        return

    all_attention_scores = pd.read_csv(attention_scores_path)
    attention_scores = all_attention_scores[all_attention_scores['slide_id'] == slide_id].copy()

    if attention_scores.empty:
        print("Warning: No attention scores found for this slide in the CSV. All patches will be black.")

    # --- ID Parsing ---
    if not pd.api.types.is_numeric_dtype(attention_scores['patch_id']):
        attention_scores['patch_id'] = attention_scores['patch_id'].str.split('-').str[-1].astype(int)

    # --- Deduplication ---
    attention_scores.drop_duplicates(subset=['patch_id'], keep='first', inplace=True)

    # --- Merging DataFrames ---
    df = pd.merge(patch_coords, attention_scores, on='patch_id', how='left')
    df['attention_score'].fillna(0, inplace=True)

    if df.empty:
        print("Warning: The merge resulted in an empty DataFrame. No patches to plot.")
        return

    # --- DYNAMIC PATCH SIZE LOGIC ---
    # Try to get objective power from metadata
    mag = slide.properties.get('openslide.objective-power')
    mpp = slide.properties.get('openslide.mpp-x')

    if mag == '40' or (mpp and 0.2 < float(mpp) < 0.3):
        patch_width, patch_height = 1024, 1024
        detected_mag = "40x"
    elif mag == '20' or (mpp and 0.4 < float(mpp) < 0.6):
        patch_width, patch_height = 512, 512
        detected_mag = "20x"
    else:
        # Default fallback
        patch_width, patch_height = 512, 512
        detected_mag = f"Unknown ({mag}x). Defaulting to 20x (512)"

    print(f"Detected magnification: {detected_mag} -> Using patch size: {patch_width}x{patch_height}")

    # --- Heatmap Generation ---
    target_max_dim = 4096
    best_level = 0
    for i, dims in enumerate(slide.level_dimensions):
        if max(dims) <= target_max_dim:
            best_level = i
            break
    else:
        best_level = slide.level_count - 1

    level = best_level
    width, height = slide.level_dimensions[level]
    downsample = slide.level_downsamples[level]
    print(f"Using slide level {level} (downsample: {downsample:.2f}x) for heatmap.")

    heatmap = np.zeros((height, width), dtype=np.float32)

    # Calculate scaled patch dimensions once
    scaled_patch_w = int(patch_width / downsample)
    scaled_patch_h = int(patch_height / downsample)

    for _, row in df.iterrows():
        x, y, attention = row['x'], row['y'], row['attention_score']

        # Coordinates x,y are always relative to Level 0 (magnification-independent)
        scaled_x = int(x / downsample)
        scaled_y = int(y / downsample)

        y_start, y_end = scaled_y, scaled_y + scaled_patch_h
        x_start, x_end = scaled_x, scaled_x + scaled_patch_w

        if y_start < height and x_start < width:
            # min() ensures we don't index outside the heatmap array
            heatmap[y_start:min(y_end, height), x_start:min(x_end, width)] = attention

    # --- Plotting ---
    slide_image = slide.read_region((0, 0), level, (width, height)).convert("RGB")

    # --- Plotting with Transparency Mask ---
    slide_image = slide.read_region((0, 0), level, (width, height)).convert("RGB")

    # Create a masked array: areas with 0 attention will be transparent
    masked_heatmap = np.ma.masked_where(heatmap == 0, heatmap)

    plt.figure(figsize=(20, 20))
    plt.imshow(slide_image)

    # Use hot, but only on the masked data
    # Set vmin/vmax to ensure 0 is the center if you want blue-to-red
    # Or just use vmin=0 if you want 0 to be the 'low' point
    plt.imshow(masked_heatmap, cmap='hot', alpha=0.5, vmin=df['attention_score'].min(), vmax=df['attention_score'].max())

    plt.axis('off')
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
    plt.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plot attention maps for a given slide.')
    parser.add_argument('--base_dir', type=str, required=True,
                        help='Directory of a patient which contains the svs file.')
    parser.add_argument('--attention_scores_path', type=str, required=True,
                        help='Path to the attention scores csv file.')
    parser.add_argument('--coordinates_path', type=str,
                        default=str(Path(__file__).resolve().parent / 'patch_coordinates_combined.parquet'),
                        help='Tile coordinate table (default: utils/patch_coordinates_combined.parquet).')
    parser.add_argument('--output_dir', type=str, help='Directory to save output.')

    args = parser.parse_args()

    output_dir = Path(args.output_dir) if args.output_dir else Path(__file__).resolve().parent / 'attention_maps'
    output_dir.mkdir(parents=True, exist_ok=True)

    svs_files = glob.glob(os.path.join(args.base_dir, '*.svs'))
    if not svs_files:
        raise FileNotFoundError(f"No .svs file found in {args.base_dir}")

    slide_path = svs_files[0]
    coordinates_path = args.coordinates_path
    output_path = output_dir / f'{Path(slide_path).stem.split(".")[0]}_attention.png'

    plot_attention_map(slide_path, coordinates_path, args.attention_scores_path, output_path)