"""
Cohort summary: patients, slides and tiles of the analysed cohort, and the per-patient cohort table.

Reads the patient-level manifest (233 patients), the slide-level table (``dataset_v1.parquet``, one row per
tile that survived tissue selection) and, optionally, the tile coordinate table (one row per tile before
tissue selection). Writes ``outputs/cohort_summary.json`` and ``data/cohort.csv``
(``patient_id, label, n_slides, n_tiles, tss``).

    python scripts/cohort_summary.py \
        --manifest source/precomputed_features/uni-patient-level/feature_manifest.parquet \
        --slide-table source/dataset_v1.parquet \
        --coords utils/patch_coordinates_combined.parquet
"""
import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

def tss_from_barcode(barcode):
    """TCGA tissue source site: TCGA-<TSS>-<participant> -> <TSS>."""
    m = re.match(r"^TCGA-([0-9A-Za-z]{2})-", str(barcode))
    return m.group(1) if m else "UNK"


def _spread(x):
    x = np.asarray(x, float)
    q1, med, q3 = np.percentile(x, [25, 50, 75])
    return {"n": int(len(x)), "median": float(med), "iqr": [float(q1), float(q3)], "mean": float(x.mean()),
            "min": float(x.min()), "max": float(x.max())}


def build_cohort(manifest, slide_table):
    """One row per patient: label and tile count from the manifest, slide count from the slide table."""
    slides = slide_table.groupby("case_id").slide_id.nunique().rename("n_slides")
    cohort = pd.DataFrame({
        "patient_id": manifest["patient_id"].to_numpy(),
        "label": manifest["patient_label"].astype(int).to_numpy(),
        "n_tiles": manifest["n_patches"].astype(int).to_numpy(),
    })
    cohort = cohort.merge(slides, left_on="patient_id", right_index=True, how="left")
    cohort["n_slides"] = cohort["n_slides"].fillna(0).astype(int)
    cohort["tss"] = cohort["patient_id"].map(tss_from_barcode)
    return cohort.sort_values("patient_id").reset_index(drop=True)[["patient_id", "label", "n_slides", "n_tiles", "tss"]]


def summarize(manifest, slide_table, coords=None):
    """Returns ``(summary dict, cohort DataFrame)``."""
    cohort = build_cohort(manifest, slide_table)
    in_cohort = slide_table[slide_table["case_id"].isin(cohort["patient_id"])]
    outside = slide_table[~slide_table["case_id"].isin(cohort["patient_id"])]

    # consistency of the manifest and the slide table for the cohort patients
    table_tiles = in_cohort.groupby("case_id").size()
    table_label = in_cohort.groupby("case_id").is_mtap_mutation.agg(["min", "max"])
    c = cohort.set_index("patient_id")
    checks = {
        "patients_missing_from_slide_table": int((~c.index.isin(in_cohort["case_id"])).sum()),
        "tile_counts_equal_manifest": bool((table_tiles.reindex(c.index) == c["n_tiles"]).all()),
        "labels_equal_manifest": bool((table_label["max"].reindex(c.index) == c["label"]).all()),
        "labels_constant_within_patient": bool((table_label["min"] == table_label["max"]).all()),
        "manifest_patient_ids_unique": bool(cohort["patient_id"].is_unique),
    }

    pos, neg = cohort[cohort.label == 1], cohort[cohort.label == 0]
    observed = {
        "n_patients": int(len(cohort)),
        "n_HD": int(len(pos)),
        "n_neutral": int(len(neg)),
        "n_slides": int(cohort["n_slides"].sum()),
        "n_tiles_after_selection": int(cohort["n_tiles"].sum()),
        "tiles_median_per_patient": float(cohort["n_tiles"].median()),
    }

    # tiles and slides at each stage of the tables
    steps = []
    if coords is not None:
        cc = coords[coords["patient_id"].isin(cohort["patient_id"])]
        steps.append({"step": "tile coordinate table, all patients (tiles before tissue selection)",
                      "patients": int(coords["patient_id"].nunique()), "slides": int(coords["slide_id"].nunique()),
                      "tiles": int(len(coords))})
        observed["n_tiles_before_selection_all_patients_in_coordinate_table"] = int(len(coords))
        observed["n_tiles_before_selection_cohort_patients"] = int(len(cc))
        observed["n_slides_cohort_patients_in_coordinate_table"] = int(cc["slide_id"].nunique())
    steps.append({"step": "slide table, all patients (tiles after tissue selection)",
                  "patients": int(slide_table["case_id"].nunique()), "slides": int(slide_table["slide_id"].nunique()),
                  "tiles": int(len(slide_table))})
    steps.append({"step": "analysed cohort (patient manifest)", "patients": observed["n_patients"],
                  "slides": observed["n_slides"], "tiles": observed["n_tiles_after_selection"]})
    outside_label = outside.groupby("case_id").is_mtap_mutation.max()

    summary = {
        "n_patients": observed["n_patients"], "n_HD": observed["n_HD"], "n_neutral": observed["n_neutral"],
        "n_slides": observed["n_slides"], "n_tiles": observed["n_tiles_after_selection"],
        "tiles_per_patient": _spread(cohort["n_tiles"]),
        "tiles_per_patient_HD": _spread(pos["n_tiles"]),
        "tiles_per_patient_neutral": _spread(neg["n_tiles"]),
        "slides_per_patient": {str(k): int(v) for k, v in cohort["n_slides"].value_counts().sort_index().items()},
        "n_tissue_source_sites": int(cohort["tss"].nunique()),
        "table_consistency": checks,
        "observed": observed,
        "selection_steps": steps,
        "patients_in_slide_table_outside_the_cohort": {
            "n": int(len(outside_label)), "n_HD": int((outside_label == 1).sum()),
            "n_neutral": int((outside_label == 0).sum()), "slides": int(outside["slide_id"].nunique()),
            "tiles": int(len(outside))},
    }
    return summary, cohort


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", required=True, help="patient-level feature_manifest.parquet")
    ap.add_argument("--slide-table", required=True, help="dataset_v1.parquet (case_id, slide_id, patch_id, ...)")
    ap.add_argument("--coords", default=None, help="optional patch_coordinates_combined.parquet")
    ap.add_argument("--out-json", default=str(ROOT / "outputs" / "cohort_summary.json"))
    ap.add_argument("--out-csv", default=str(ROOT / "data" / "cohort.csv"))
    args = ap.parse_args()

    manifest = pd.read_parquet(args.manifest)
    slide_table = pd.read_parquet(args.slide_table)
    coords = pd.read_parquet(args.coords) if args.coords else None
    summary, cohort = summarize(manifest, slide_table, coords)

    for out in (args.out_json, args.out_csv):
        Path(out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_json, "w", newline="\n") as f:
        json.dump(summary, f, indent=2)
        f.write("\n")
    cohort.to_csv(args.out_csv, index=False, lineterminator="\n")

    print(f"{summary['n_patients']} patients ({summary['n_HD']} HD, {summary['n_neutral']} neutral), "
          f"{summary['n_slides']} slides, {summary['n_tiles']:,} tiles")
    print(f"wrote {args.out_json} and {args.out_csv}")


if __name__ == "__main__":
    main()
