from pathlib import Path
import pandas as pd
from openslide import open_slide
import numpy as np
from utils import get_svs_path, run_over_slides

# --- Tiling parameters (same as 1.3-extract_patches.py) ---
thresh_high = 220.0 / 255  # Threshold to remove blank patches
patch_size = 512  # The size of patches extracted from 20x svs image
target_mag = 20


def slide_coordinates(svs_dir):
    """Rows (patient_id, slide_id, patch_id, x, y) of the tissue patches of one slide folder; [] if it cannot be read."""
    svs_dir = Path(svs_dir)
    patient_id = svs_dir.parent.name
    try:
        svs_path = get_svs_path(str(svs_dir))
        slide_name = Path(svs_path).name.split('.')[0]
        print(f"\n  -> Processing slide: {slide_name} for patient {patient_id}")

        slide = open_slide(svs_path)
        magnification = float(slide.properties.get('aperio.AppMag', target_mag))

        extract_patch_size = int(patch_size * magnification / target_mag)
        w, h = slide.level_dimensions[0]

        # Align width and height to the patch size grid
        w = w // extract_patch_size * extract_patch_size
        h = h // extract_patch_size * extract_patch_size

        patch_count = 0
        rows = []
        for i in range(0, w, extract_patch_size):
            for j in range(0, h, extract_patch_size):
                patch_count += 1
                patch = slide.read_region((i, j), level=0, size=[extract_patch_size, extract_patch_size])

                # Downsample to target patch size for tissue check
                patch_resized = patch.resize([patch_size, patch_size])

                # Check if the patch contains tissue
                patch_gray = patch_resized.convert('1')
                ave_pixel_val = np.array(patch_gray).mean()

                if ave_pixel_val < thresh_high:
                    # If it's a valid tissue patch, store its info
                    rows.append({
                        'patient_id': patient_id,
                        'slide_id': slide_name,
                        'patch_id': patch_count,
                        'x': i,
                        'y': j
                    })

        slide.close()
        print(f"     - Found {len(rows)} valid patches for slide {slide_name}.")
        return rows

    except Exception as e:
        print(f"     - ERROR: Could not process slide in {svs_dir}. Reason: {e}")
        return []


def extract_patch_coordinates(base_dir, workers=1):
    """
    Iterates through a directory structure, calculates patch coordinates without
    saving patch images, and stores the coordinates in a single Parquet file.
    ``workers`` > 1 processes that many slides in parallel (the rows are the same, in the same order).
    """
    output_parquet_path = Path(base_dir) / "patch_coordinates.parquet"

    print(f"Scanning base directory: {base_dir}")
    print(f"Output will be saved to: {output_parquet_path}")

    all_patch_data = [row for rows in run_over_slides(slide_coordinates, base_dir, workers) for row in rows]

    # --- Write all collected data to a single Parquet file ---
    if not all_patch_data:
        print("\nNo valid patches were found. Parquet file will not be created.")
        return

    print(f"\nFound {len(all_patch_data)} total valid patches across all slides. Writing to Parquet...")
    df = pd.DataFrame(all_patch_data)
    df.to_parquet(output_parquet_path, index=False)

    print(f"Successfully created {output_parquet_path}")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="Write patch_coordinates.parquet for all slides under --base-dir.")
    parser.add_argument("--base-dir", required=True,
                        help="folder with <patient_id>/<slide_name>/<slide_name>.svs (all patients under one folder)")
    parser.add_argument("--workers", type=int, default=1, help="number of slides processed in parallel")
    args = parser.parse_args()
    if not Path(args.base_dir).exists():
        parser.error(f"base directory not found: {args.base_dir}")
    extract_patch_coordinates(args.base_dir, args.workers)
