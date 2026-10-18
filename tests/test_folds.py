"""Cross-validation folds: folds.csv, the median patch-count bucket, and the reference split of the 233 patients."""
import numpy as np
import pandas as pd
import pytest
from omegaconf import OmegaConf

from conftest import ROOT, REAL_MANIFEST, FIXTURES, make_cfg
from dataset.mil_dataset import MILDataModule, patch_buckets, make_fold_indices, folds_table
from train_k_fold import write_folds_csv

needs_manifest = pytest.mark.skipif(not REAL_MANIFEST.exists(), reason="patient manifest not present")


def real_cfg():
    cfg = OmegaConf.load(ROOT / "configs" / "train_config.yaml")
    cfg.data.parquet_path = str(REAL_MANIFEST)
    return cfg


@needs_manifest
def test_median_of_the_cohort_is_3080():
    df = pd.read_parquet(REAL_MANIFEST)
    assert len(df) == 233
    assert np.median(df["n_patches"]) == 3080
    np.testing.assert_array_equal(patch_buckets(df), (df["n_patches"] > 3080).astype(int))


@needs_manifest
@pytest.mark.parametrize("fold_index", range(5))
def test_splits_equal_the_reference_split(fold_index):
    """train/val/test patient lists (and their order) equal the reference split (seed 42)."""
    baseline = pd.read_csv(FIXTURES / "folds_reference.csv")
    baseline = baseline[baseline["fold_index"] == fold_index]
    dm = MILDataModule(real_cfg(), fold_index=fold_index)
    with pytest.warns(UserWarning, match="Could not open any patient H5"):
        dm.setup("fit")  # the patient H5 feature files are not part of the repository
    for split, dataset in [("train", dm.train_dataset), ("val", dm.val_dataset), ("test", dm.test_dataset)]:
        expected = list(baseline.loc[baseline["split"] == split, "patient_id"])
        assert list(dataset.df["patient_id"]) == expected


@needs_manifest
def test_folds_table_matches_the_reference_test_sets():
    df = pd.read_parquet(REAL_MANIFEST)
    cfg = real_cfg()
    table = folds_table(df, "patient_id", "patient_label", cfg.data.k_folds, cfg.seed)
    assert list(table.columns) == ["patient_id", "fold", "label", "n_patches"]
    assert table["patient_id"].is_unique and len(table) == 233
    baseline = pd.read_csv(FIXTURES / "folds_reference.csv")
    tested = baseline[baseline["split"] == "test"].set_index("patient_id")["fold_index"]
    # a patient's `fold` is the run in which the reference split tests that patient
    assert (table.set_index("patient_id")["fold"] == tested.reindex(table["patient_id"]).values).all()
    # labels and patch counts are carried over unchanged
    pd.testing.assert_series_equal(table["label"], df["patient_label"], check_names=False)
    pd.testing.assert_series_equal(table["n_patches"], df["n_patches"], check_names=False)


@needs_manifest
def test_write_folds_csv_is_deterministic(tmp_path):
    cfg = real_cfg()
    first = write_folds_csv(cfg, tmp_path / "a" / "folds.csv")
    second = write_folds_csv(cfg, tmp_path / "b" / "folds.csv")
    assert (tmp_path / "a" / "folds.csv").read_bytes() == (tmp_path / "b" / "folds.csv").read_bytes()
    pd.testing.assert_frame_equal(first, pd.read_csv(tmp_path / "a" / "folds.csv"))
    assert sorted(first["fold"].unique()) == [0, 1, 2, 3, 4]


def test_folds_are_a_partition_and_val_test_train_are_disjoint(cohort):
    _, df = cohort
    folds = make_fold_indices(df, "patient_label", 3, seed=42)
    assert sorted(np.concatenate(folds)) == list(range(len(df)))
    for a in range(3):
        for b in range(a + 1, 3):
            assert not set(folds[a]) & set(folds[b])


def test_median_is_computed_from_the_data(cohort):
    """No cohort constant: a manifest with another median gets its own split point."""
    _, df = cohort
    shifted = df.assign(n_patches=df["n_patches"] + 5000)
    assert patch_buckets(shifted).sum() == patch_buckets(df).sum()
    assert patch_buckets(shifted).sum() == (shifted["n_patches"] > np.median(shifted["n_patches"])).sum()


def test_k_folds_below_three_rejected(cohort):
    _, df = cohort
    with pytest.raises(ValueError, match="at least 3 folds"):
        make_fold_indices(df, "patient_label", 2, seed=42)
