import argparse
import time
from pathlib import Path
from skimage import io
import os
from openslide import open_slide
import numpy as np
from tqdm import tqdm
from utils import get_svs_path, run_over_slides

thresh_high = 220.0/255  # threshod to remove blank patches
patch_size = 512    # the size of patches extracted from 20x svs image
target_mag = 20
resize_flag = False    # True: downsample the patch from 40x image by 2


def extract_patches(case_dir, slide_path, thresh_high = 220.0/255, patch_size = 512, target_mag = 20):
    slide_name = Path(slide_path).name.split('.')[0]

    # Get magnification
    slide = open_slide(slide_path)
    magnification = float(slide.properties['aperio.AppMag'])

    # create designated folder for patches
    patches_folder = os.path.join(case_dir, "patches")
    if not os.path.exists(patches_folder):
        os.makedirs(patches_folder)

    extract_patch_size = int(patch_size * magnification / target_mag)
    scale = 20.0 / patch_size

    w, h = slide.level_dimensions[0]

    # remove some pixels on the bottom and right edges of the image
    w = w // extract_patch_size * extract_patch_size
    h = h // extract_patch_size * extract_patch_size

    patch_mask = np.zeros((int(h*scale), int(w*scale)))  # the mask to indicate which parts are extracted

    count = 0
    num_patch = 0

    time_slide_reading = 0
    time_resize = 0
    time_write_image = 0
    time_all = 0
    for i in tqdm(range(0, w, extract_patch_size)):
        for j in range(0, h, extract_patch_size):
            t1 = time.time()
            patch = slide.read_region((i, j), level=0, size=[extract_patch_size, extract_patch_size])
            count += 1

            t2 = time.time()
            # downsample to target patch size
            patch = patch.resize([patch_size, patch_size])

            t3 = time.time()
            # check if the patch contains tissue
            patch_gray = patch.convert('1')
            ave_pixel_val = np.array(patch_gray).mean()
            if ave_pixel_val < thresh_high:
                num_patch += 1
                patch_name = os.path.join(patches_folder, f"{count}.jpg")
                patch.convert('RGB').save(patch_name, 'jpeg', quality=95)
                c1 = int(i*scale)
                c2 = int((i+extract_patch_size)*scale)
                r1 = int(j*scale)
                r2 = int((j+extract_patch_size)*scale)
                patch_mask[r1:r2, c1:c2] = 1

            t4 = time.time()

            time_slide_reading += t2 - t1
            time_resize += t3 - t2
            time_write_image += t4 - t3
            time_all += t4 - t1

    print('\n\tTotal time: {:.2f}'.format(time_all))
    print('\tTime to load slides: {:.2f} ({:.2f} %)'.format(time_slide_reading, time_slide_reading/time_all * 100))
    print('\tTime to downsample patches: {:.2f} ({:.2f} %)'.format(time_resize, time_resize/time_all * 100))
    print('\tTime to write images: {:.2f} ({:.2f} %)'.format(time_write_image, time_write_image/time_all * 100))
    print('\tTotal number: {:d}'.format(num_patch))

    # create mask image folder if it doesn't exist
    mask_folder = os.path.join(case_dir, "mask")
    if not os.path.exists(mask_folder):
        os.makedirs(mask_folder)

    # save mask images
    mask_name = '{:s}/{:s}_mask.png'.format(mask_folder, slide_name)
    patch_mask_to_save = (patch_mask * 255).astype(np.uint8)
    io.imsave(mask_name, patch_mask_to_save)
    return num_patch


def rgb2gray(image):
    gray = np.zeros((image.shape[0], image.shape[1]))  # init 2D numpy array
    for i in range(len(image)):
        for j in range(len(image[i])):
            pixel = image[i][j]
            gray[i][j] = 0.299*pixel[0] + 0.587*pixel[1] + 0.114*pixel[2]
    return gray


def process_slide_dir(svs_dir):
    """Extract the tissue patches of one slide folder (skipped when its mask already exists)."""
    svs_path = get_svs_path(str(svs_dir))
    slide_name = Path(svs_path).name.split('.')[0]
    if os.path.exists(os.path.join(svs_dir, "mask", f"{slide_name}_mask.png")):
        print(f"already extracted: {svs_dir}")
        return None
    print(f"svs_dir: {svs_dir}")
    return extract_patches(svs_dir, svs_path, thresh_high = 220.0/255, patch_size = 512, target_mag = 20)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Extract 512 px tissue patches at 20x from every slide under --base-dir.")
    parser.add_argument("--base-dir", required=True, help="raw_data folder with <case_id>/<slide_name>/<slide_name>.svs")
    parser.add_argument("--workers", type=int, default=1, help="number of slides processed in parallel")
    args = parser.parse_args()
    run_over_slides(process_slide_dir, args.base_dir, args.workers)
