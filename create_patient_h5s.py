import argparse
import os

import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm

# Columns of the slide-level manifest written by extract_features.py
PATIENT_COL = "case_id"
SLIDE_ID_COL = "slide_id"
SLIDE_LABEL_COL = "slide_label"
SLIDE_H5_COL = "reference_to_slide_h5"

# Columns of the patient-level manifest read by MILDataModule
OUT_PATIENT_COL = "patient_id"
OUT_LABEL_COL = "patient_label"


def create_patient_h5_files(slide_manifest_path, out_dir, manifest_out):
    """
    Converts slide-level H5 feature files into patient-level H5 files.

    Args:
        slide_manifest_path: slide-level manifest parquet (columns case_id, slide_id,
            reference_to_slide_h5, slide_label), as written by extract_features.py.
        out_dir: directory that receives one ``<patient_id>.h5`` per patient.
        manifest_out: path of the patient-level manifest parquet to write (a file, not a directory).

    The manifest has the columns the trainer needs: patient_id, patient_label, n_patches,
    n_slides, patient_h5_path.
    """
    print(f"Loading slide feature manifest from: {slide_manifest_path}")
    df = pd.read_parquet(slide_manifest_path)

    os.makedirs(out_dir, exist_ok=True)
    manifest_dir = os.path.dirname(os.fspath(manifest_out))
    if manifest_dir:
        os.makedirs(manifest_dir, exist_ok=True)
    print(f"Patient-level H5 files will be saved to: {out_dir}")

    patient_groups = df.groupby(PATIENT_COL)
    new_manifest_data = []

    print(f"Processing {len(patient_groups)} patients...")
    for patient_id, patient_df in tqdm(patient_groups, desc="Creating Patient H5s"):
        patient_features_list = []
        patient_patch_ids_list = []
        n_slides_read = 0

        patient_df = patient_df.sort_values(by=SLIDE_ID_COL).reset_index(drop=True)

        # All slides of a patient must carry the same label: never silently take the first one
        labels = patient_df[SLIDE_LABEL_COL].unique()
        if len(labels) != 1:
            raise ValueError(
                f"Patient {patient_id} has slides with different labels {sorted(labels.tolist())}."
            )

        for _, row in patient_df.iterrows():
            slide_h5_path = row[SLIDE_H5_COL]
            slide_id = row[SLIDE_ID_COL]
            try:
                with h5py.File(slide_h5_path, 'r') as f:
                    slide_features = []
                    slide_patch_ids = []
                    for patch_id in f.keys():
                        slide_features.append(f[patch_id][()])
                        slide_patch_ids.append(f"{slide_id}-{patch_id}")
            except Exception as e:
                print(f"Warning: Could not read {slide_h5_path} for patient {patient_id}. Error: {e}. Skipping slide.")
                continue
            patient_features_list.extend(slide_features)
            patient_patch_ids_list.extend(slide_patch_ids)
            n_slides_read += 1

        if not patient_features_list:
            print(f"Warning: No features found for patient {patient_id}. Skipping patient.")
            continue

        patient_features_array = np.array(patient_features_list)
        patient_patch_ids_array = np.array(patient_patch_ids_list, dtype=h5py.string_dtype(encoding='utf-8'))

        patient_h5_path = os.path.join(out_dir, f"{patient_id}.h5")
        with h5py.File(patient_h5_path, 'w') as f:
            f.create_dataset('features', data=patient_features_array)
            f.create_dataset('patch_ids', data=patient_patch_ids_array)

        new_manifest_data.append({
            OUT_PATIENT_COL: patient_id,
            OUT_LABEL_COL: labels[0],
            'n_patches': len(patient_features_array),
            'n_slides': n_slides_read,
            'patient_h5_path': patient_h5_path,
        })

    new_manifest_df = pd.DataFrame(new_manifest_data)
    new_manifest_df.to_parquet(manifest_out, index=False)

    print("\nPatient H5 file creation complete.")
    print(f"New manifest saved to: {manifest_out}")
    return new_manifest_df


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=create_patient_h5_files.__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--slide-manifest", required=True,
                        help="Slide-level manifest parquet from extract_features.py "
                             "(e.g. source/precomputed_features/uni/feature_manifest.parquet).")
    parser.add_argument("--out-dir", required=True,
                        help="Directory for the patient-level H5 files "
                             "(e.g. source/precomputed_features/uni-patient-level).")
    parser.add_argument("--manifest-out", required=True,
                        help="Path of the patient-level manifest parquet to write "
                             "(e.g. source/precomputed_features/uni-patient-level/feature_manifest.parquet).")
    return parser.parse_args(argv)


if __name__ == '__main__':
    args = parse_args()
    create_patient_h5_files(args.slide_manifest, args.out_dir, args.manifest_out)
