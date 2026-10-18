"""Move every .svs image into a folder of its own: <base_dir>/<case_id>/<slide_name>/<slide_name>.svs.

Run on a base folder whose sub-directories are case ids (TCGA-00-0000) that contain the .svs images.

    python svs_organizer.py --base-dir <raw_data>
"""
import argparse
import os
import shutil


def organize(base_dir):
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file.lower().endswith(".svs"):
                svs_path = os.path.join(root, file)

                # Folder name = filename without extension
                slide_name = os.path.basename(svs_path).split('.', 1)[0]
                case_folder_name = os.path.basename(root)
                dest_folder = os.path.join(base_dir, case_folder_name, slide_name)
                os.makedirs(dest_folder, exist_ok=True)

                dest_path = os.path.join(dest_folder, file)
                print(f"Moving {svs_path} -> {dest_path}")
                shutil.move(svs_path, dest_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base-dir", required=True, help="raw_data folder with one sub-folder per case")
    organize(parser.parse_args().base_dir)
