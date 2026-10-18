import pandas as pd
import argparse
from pathlib import Path

from tqdm import tqdm


def label_from_tsv(tsv_path):
    """
    Label of a case from its gene-level copy-number TSV: 1 if the MTAP copy number is 0 (homozygous deletion),
    otherwise 0. Raises if the file cannot be read.
    """
    # Determine Label: Default to 0, switch to 1 if MTAP copy_number is 0
    label = 0
    df_meta = pd.read_csv(tsv_path, sep='\t')

    # Filter for gene_name == 'MTAP'
    mtap_row = df_meta[df_meta['gene_name'] == 'MTAP']

    if not mtap_row.empty:
        copy_number = mtap_row.iloc[0]['copy_number']
        if copy_number == 0:
            label = 1
    else:
        print(f"Warning: 'MTAP' gene not found in {Path(tsv_path).name}. Defaulting to Group 0.")
    return label


def slide_records(case_id, slide_dir, label):
    """One record per normalized patch of a slide folder (empty if the slide has no 20xNormalizedPatches folder)."""
    slide_dir = Path(slide_dir)
    patches_dir = slide_dir / "20xNormalizedPatches"
    records = []
    if patches_dir.exists():
        # Collect all patches
        for patch_file in patches_dir.iterdir():
            if patch_file.is_file() and not patch_file.name.startswith('.'):
                records.append({
                    'case_id': case_id,
                    'slide_id': slide_dir.name,
                    'patch_id': patch_file.stem,
                    'file_path': str(patch_file),
                    'is_mtap_mutation': label # If patient has MTAP mutation (homozygous deletion), then the label = 1. Otherwise, label = 0.
                })
    return records


def generate_slide_summary(data_root, output_file):
    """
    Scans the data_root for cases, parses TSV files for labels,
    and aggregates patch IDs into a Parquet file.
    """
    root_path = Path(data_root)
    records = []

    print(f"Scanning directory: {root_path}")

    # Iterate over items in the root directory
    for case_dir in tqdm(root_path.iterdir(), desc="Processing cases", total=len(list(root_path.iterdir()))):
        if not case_dir.is_dir():
            continue

        case_id = case_dir.name

        # Find the TSV file in the case main directory
        tsv_files = list(case_dir.glob("*.tsv"))
        if not tsv_files:
            print(f"Warning: No TSV file found for case {case_id}. Skipping.")
            continue

        # Assuming the first TSV found is the correct one
        tsv_path = tsv_files[0]

        try:
            label = label_from_tsv(tsv_path)
        except Exception as e:
            print(f"Error reading TSV for {case_id}: {e}")
            continue

        # Iterate over slide directories within the case directory
        for slide_dir in case_dir.iterdir():
            if not slide_dir.is_dir():
                continue
            records += slide_records(case_id, slide_dir, label)

    # Create DataFrame and save to Parquet
    if records:
        df_final = pd.DataFrame(records)
        Path(output_file).mkdir(parents=True, exist_ok=True)
        df_final.to_parquet(Path(output_file) / "slides_summary.parquet", index=False)
        print(f"Successfully created {Path(output_file) / 'slides_summary.parquet'} with {len(df_final)} rows.")
    else:
        print("No records found to save.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create Parquet summary of slide patches.")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Path to the root folder containing case directories.")
    parser.add_argument("--output", type=str, default=".",
                        help="Output directory; slides_summary.parquet is written there.")

    args = parser.parse_args()

    generate_slide_summary(args.data_dir, args.output)
