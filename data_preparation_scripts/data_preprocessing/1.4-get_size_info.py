import argparse
import json
from pathlib import Path
from openslide import open_slide
import pandas as pd
from utils import get_svs_path, run_over_slides

PATCH_SIZE = 512  # the size of patches extracted from 20x svs image
TARGET_MAG = 20


def get_size_info(case_dir, slide_path):
    slide_name = Path(slide_path).name.split('.')[0]
    slide_size_info = pd.DataFrame(columns=['Slide_name', 'Mag', 'Height', 'Width'])
    slide = open_slide(slide_path)
    magnification = float(slide.properties['aperio.AppMag'])
    w, h = slide.level_dimensions[0]
    slide_size_info = pd.concat([
        slide_size_info,
        pd.DataFrame([{
            'Slide_name': slide_name,
            'Mag': int(magnification),
            'Height': int(h),
            'Width': int(w)
        }])
    ], ignore_index=True)

    slide_size_info.to_pickle('{:s}/slide_size_info.pickle'.format(str(case_dir)))
    write_slide_properties(case_dir, slide_name, slide, magnification)
    print(slide_size_info.shape)


def write_slide_properties(case_dir, slide_name, slide, magnification):
    """
    Record what the tiling stage works from: the nominal magnification, the microns per pixel the scanner reports
    and the region read at level 0 for one 512 px tile. (Stage 1.3 always reads level 0 and resizes the
    ``512 * AppMag / 20`` px region to 512 px, so a 40x slide is downsampled 2x.) Informational only.
    """
    properties = slide.properties
    mpp = [float(properties[k]) if k in properties else None for k in ('openslide.mpp-x', 'openslide.mpp-y')]
    region = int(PATCH_SIZE * magnification / TARGET_MAG)
    info = {
        'slide_name': slide_name,
        'aperio.AppMag': magnification,
        'mpp_x': mpp[0], 'mpp_y': mpp[1],
        'level_dimensions': [list(d) for d in slide.level_dimensions],
        'level_downsamples': [float(d) for d in slide.level_downsamples],
        'level_used': 0,
        'region_read_px': region,
        'tile_px': PATCH_SIZE,
        'resize_factor': region / PATCH_SIZE,
        'effective_um_per_px_of_tile': None if mpp[0] is None else mpp[0] * region / PATCH_SIZE,
    }
    with open(Path(case_dir) / 'slide_properties.json', 'w', newline='\n') as f:
        json.dump(info, f, indent=2)
        f.write('\n')


def process_slide_dir(svs_dir):
    print(f"svs_dir: {svs_dir}")
    get_size_info(svs_dir, get_svs_path(str(svs_dir)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Write slide_size_info.pickle (and slide_properties.json) per slide.")
    parser.add_argument("--base-dir", required=True, help="raw_data folder with <case_id>/<slide_name>/<slide_name>.svs")
    parser.add_argument("--workers", type=int, default=1, help="number of slides processed in parallel")
    args = parser.parse_args()
    run_over_slides(process_slide_dir, args.base_dir, args.workers)
