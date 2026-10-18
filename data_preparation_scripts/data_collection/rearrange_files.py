"""Rearrange a GDC download into one sub-directory per case: <raw_data>/<case_id>/.

Download the slide images (or the gene-level copy-number TSV files) with gdc-client into a temporary folder that
also holds the GDC portal metadata.json, then run this script; repeat for the second download.

    python rearrange_files.py --download-dir <download folder> --raw-data <raw_data>
"""
import argparse
import json
import os
import re
import shutil
from tqdm import tqdm


def create_case_directory(case_id, raw_data_directory):
    target_directory = os.path.join(raw_data_directory, case_id)
    if not os.path.exists(target_directory):
        os.makedirs(target_directory)
        print(f"Created directory: {target_directory}")
    else:
        print(f"Directory already exists: {target_directory}")
    return target_directory


def get_case_id(submitter_id: str):
    match = re.match(r'TCGA-\w+-\w+', submitter_id)
    if match:
        result = match.group(0)
        print(f"Found: {result}")
        return result
    print("Returns None")
    return None

def read_metadata_file(file_path):
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Error: The file at {file_path} was not found.")
        return None
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from the file at {file_path}.")
        return None

def rearrange(download_dir, raw_data_directory):
    metadata = read_metadata_file(os.path.join(download_dir, 'metadata.json'))
    for file_metadata in tqdm(metadata, desc="Ordering files"):
        file_id = file_metadata['file_id']
        submitter_id = file_metadata['submitter_id']
        file_name = file_metadata['file_name']
        case_id = get_case_id(submitter_id)  # TCGA-00-0000
        dst_dir = create_case_directory(case_id, raw_data_directory)
        src_file = os.path.join(download_dir, file_id, file_name)
        try:
            shutil.move(src_file, dst_dir)
            print(f"Moved: {src_file} to {dst_dir}")
        except FileNotFoundError:
            print(f"Error: The file at {src_file} was not found.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--download-dir", required=True, help="gdc-client download folder with metadata.json")
    parser.add_argument("--raw-data", default="raw_data", help="target folder (one sub-folder per case)")
    args = parser.parse_args()
    rearrange(args.download_dir, args.raw_data)
