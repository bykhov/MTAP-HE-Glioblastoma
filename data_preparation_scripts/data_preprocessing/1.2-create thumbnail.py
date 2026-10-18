import argparse
import os
from openslide import open_slide
from pathlib import Path

from utils import get_svs_path, run_over_slides


def create_thumbnail(slide_path, case_dir):
    patch_size = 512  # the size of patches extracted from 20x svs image
    target_mag = 20

    slide = open_slide(slide_path)
    if 'aperio.AppMag' in slide.properties:
        slide_name = Path(slide_path).name.split('.')[0]
        magnification = float(slide.properties['aperio.AppMag'])
        extract_patch_size = int(patch_size * magnification / target_mag)
        w, h = slide.level_dimensions[0]
        th_w = int(w / extract_patch_size * 10)
        th_h = int(h / extract_patch_size * 10)
        thumbnail = slide.get_thumbnail((th_w, th_h))
        thumbnail_name = '{:s}/{:s}_thumbnail.png'.format(case_dir, slide_name)
        thumbnail.save(thumbnail_name)
        print(f"thumbnail saved for: {slide_name}")
    else:
        print(f"Slide: {slide_path} does not have magnification information.")


def process_slide_dir(svs_dir):
    svs_dir = str(svs_dir)
    print(f"svs_dir: {svs_dir}")
    create_thumbnail(get_svs_path(svs_dir), svs_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Create a thumbnail for every slide folder under --base-dir.")
    parser.add_argument("--base-dir", required=True, help="raw_data folder with <case_id>/<slide_name>/<slide_name>.svs")
    parser.add_argument("--workers", type=int, default=1, help="number of slides processed in parallel")
    args = parser.parse_args()
    run_over_slides(process_slide_dir, args.base_dir, args.workers)
