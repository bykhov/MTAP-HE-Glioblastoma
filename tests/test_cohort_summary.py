"""scripts/cohort_summary.py on a synthetic manifest, slide table and coordinate table."""
import importlib.util
import json

import numpy as np
import pandas as pd

import pytest

from conftest import ROOT, REAL_MANIFEST, REAL_SLIDE_TABLE


def load_script():
    spec = importlib.util.spec_from_file_location("cohort_summary", ROOT / "scripts" / "cohort_summary.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def synthetic_tables():
    """Four cohort patients (2 sites) plus two patients that are only in the slide table."""
    slide_rows, coord_rows, manifest_rows = [], [], []
    tables = {  # patient: (label, tiles per slide)
        "TCGA-02-0001": (1, [5, 3]), "TCGA-02-0002": (0, [4]), "TCGA-06-0003": (1, [10]), "TCGA-06-0004": (0, [2, 2, 2]),
        "TCGA-06-0005": (0, [6]), "TCGA-12-0006": (1, [7, 1]),
    }
    for patient, (label, tiles) in tables.items():
        for s, n in enumerate(tiles):
            slide = f"{patient}-01Z-00-DX{s + 1}"
            slide_rows += [{"case_id": patient, "slide_id": slide, "patch_id": str(i), "file_path": "x",
                            "is_mtap_mutation": label} for i in range(n)]
            coord_rows += [{"patient_id": patient, "slide_id": slide, "patch_id": i, "x": 0, "y": 0}
                           for i in range(2 * n)]  # twice as many tiles before tissue selection
    slide_table, coords = pd.DataFrame(slide_rows), pd.DataFrame(coord_rows)
    cohort_ids = list(tables)[:4]
    for p in cohort_ids:
        manifest_rows.append({"patient_id": p, "patient_label": tables[p][0], "n_patches": sum(tables[p][1]),
                              "patient_h5_path": "x", "bag_size": "small"})
    return pd.DataFrame(manifest_rows), slide_table, coords


def test_counts_and_medians():
    module = load_script()
    manifest, slide_table, coords = synthetic_tables()
    summary, cohort = module.summarize(manifest, slide_table, coords)

    assert list(cohort.columns) == ["patient_id", "label", "n_slides", "n_tiles", "tss"]
    assert cohort.set_index("patient_id").loc["TCGA-02-0001"].to_dict() == {"label": 1, "n_slides": 2, "n_tiles": 8, "tss": "02"}
    assert (summary["n_patients"], summary["n_HD"], summary["n_neutral"]) == (4, 2, 2)
    assert (summary["n_slides"], summary["n_tiles"]) == (7, 8 + 4 + 10 + 6)
    assert summary["tiles_per_patient"]["median"] == 7.0  # median of 4, 6, 8, 10
    assert summary["tiles_per_patient"]["iqr"] == [5.5, 8.5]
    assert summary["tiles_per_patient_HD"]["median"] == 9.0 and summary["tiles_per_patient_neutral"]["median"] == 5.0
    assert summary["slides_per_patient"] == {"1": 2, "2": 1, "3": 1}
    assert summary["table_consistency"]["tile_counts_equal_manifest"]
    assert summary["table_consistency"]["labels_equal_manifest"]
    # the patients outside the cohort and the coordinate table (twice the tiles) are reported, not mixed in
    assert summary["patients_in_slide_table_outside_the_cohort"] == {"n": 2, "n_HD": 1, "n_neutral": 1, "slides": 3, "tiles": 6 + 8}
    assert summary["observed"]["n_tiles_before_selection_cohort_patients"] == 2 * 28
    steps = {s["step"].split(",")[0]: s for s in summary["selection_steps"]}
    assert steps["tile coordinate table"]["patients"] == 6
    assert summary["observed"]["n_slides"] == 7


def test_detects_tile_count_mismatch_between_tables():
    module = load_script()
    manifest, slide_table, _ = synthetic_tables()
    manifest.loc[0, "n_patches"] += 1
    summary, _ = module.summarize(manifest, slide_table)
    assert summary["table_consistency"]["tile_counts_equal_manifest"] is False


def test_cli_writes_json_and_csv(tmp_path, monkeypatch):
    module = load_script()
    manifest, slide_table, coords = synthetic_tables()
    paths = {name: tmp_path / f"{name}.parquet" for name in ("manifest", "slides", "coords")}
    manifest.to_parquet(paths["manifest"]); slide_table.to_parquet(paths["slides"]); coords.to_parquet(paths["coords"])
    out_json, out_csv = tmp_path / "out" / "cohort_summary.json", tmp_path / "data" / "cohort.csv"
    monkeypatch.setattr("sys.argv", ["cohort_summary.py", "--manifest", str(paths["manifest"]),
                                     "--slide-table", str(paths["slides"]), "--coords", str(paths["coords"]),
                                     "--out-json", str(out_json), "--out-csv", str(out_csv)])
    module.main()
    assert json.loads(out_json.read_text())["n_patients"] == 4
    written = pd.read_csv(out_csv)
    assert written["patient_id"].is_unique and written["n_tiles"].sum() == 28
    assert b"\r\n" not in out_csv.read_bytes()


@pytest.mark.skipif(not (REAL_MANIFEST.exists() and REAL_SLIDE_TABLE.exists()), reason="parquet files not found")
def test_manifest_is_consistent_with_the_slide_table():
    """The repository tables: 233 / 146 / 87 patients, 1,115,734 tiles, median 3,080 tiles per patient."""
    module = load_script()
    summary, cohort = module.summarize(pd.read_parquet(REAL_MANIFEST), pd.read_parquet(REAL_SLIDE_TABLE))
    assert (summary["n_patients"], summary["n_HD"], summary["n_neutral"]) == (233, 146, 87)
    assert summary["n_tiles"] == 1_115_734 and summary["tiles_per_patient"]["median"] == 3080
    assert all(summary["table_consistency"][k] for k in
               ("tile_counts_equal_manifest", "labels_equal_manifest", "manifest_patient_ids_unique"))
    assert summary["table_consistency"]["patients_missing_from_slide_table"] == 0
    assert np.isfinite(cohort["n_slides"]).all() and (cohort["n_slides"] >= 1).all()
