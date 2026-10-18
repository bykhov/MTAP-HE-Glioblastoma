"""data.folds_csv: the split is read from a folds file instead of recomputed from the manifest (tile counts that
differ slightly could move a patient across the median bucket and change the folds)."""
import numpy as np
import pandas as pd
import pytest
from omegaconf import OmegaConf

from conftest import ROOT, REAL_MANIFEST, FIXTURES, make_cfg
from dataset.mil_dataset import MILDataModule, fold_indices_from_csv, folds_table, make_fold_indices
from train_k_fold import train_k_fold

needs_manifest = pytest.mark.skipif(not REAL_MANIFEST.exists(), reason="patient manifest not present")


def test_default_config_does_not_fix_the_folds():
    cfg = OmegaConf.load(ROOT / "configs" / "train_config.yaml")
    assert cfg.data.folds_csv is None


@needs_manifest
@pytest.mark.parametrize("fold_index", range(5))
def test_fixed_folds_keep_the_reference_split_when_tile_counts_change(tmp_path, fold_index):
    manifest = pd.read_parquet(REAL_MANIFEST)
    folds = folds_table(manifest, "patient_id", "patient_label", 5, 42)
    folds.to_csv(tmp_path / "folds.csv", index=False)

    # every tile count moves by up to 15%, so the median buckets (and the recomputed folds) change
    rng = np.random.default_rng(0)
    changed = manifest.copy()
    changed["n_patches"] = (changed["n_patches"] * rng.uniform(0.85, 1.15, len(changed))).astype(int)
    changed.to_parquet(tmp_path / "manifest.parquet", index=False)
    recomputed = folds_table(changed, "patient_id", "patient_label", 5, 42)
    assert (recomputed["fold"] != folds["fold"]).any(), "the perturbation should be large enough to move patients"

    cfg = OmegaConf.load(ROOT / "configs" / "train_config.yaml")
    cfg.data.parquet_path = str(tmp_path / "manifest.parquet")
    cfg.data.folds_csv = str(tmp_path / "folds.csv")
    datamodule = MILDataModule(cfg, fold_index=fold_index)
    with pytest.warns(UserWarning, match="Could not open any patient H5"):
        datamodule.setup("fit")
    baseline = pd.read_csv(FIXTURES / "folds_reference.csv")
    baseline = baseline[baseline["fold_index"] == fold_index]
    for split, dataset in [("train", datamodule.train_dataset), ("val", datamodule.val_dataset), ("test", datamodule.test_dataset)]:
        assert list(dataset.df["patient_id"]) == list(baseline.loc[baseline["split"] == split, "patient_id"])


def test_fold_indices_from_csv_equal_the_computed_ones(cohort, tmp_path):
    manifest_path, df = cohort
    folds_table(df, "patient_id", "patient_label", 3, 42).to_csv(tmp_path / "folds.csv", index=False)
    fixed = fold_indices_from_csv(df, "patient_id", "patient_label", 3, tmp_path / "folds.csv")
    computed = make_fold_indices(df, "patient_label", 3, 42)
    for a, b in zip(fixed, computed):
        np.testing.assert_array_equal(np.sort(a), np.sort(b))


@pytest.mark.parametrize("damage, message", [
    ("patient", "patients differ"),
    ("label", "labels differ"),
    ("fold", "expected fold values"),
    ("column", "missing column"),
])
def test_inconsistent_folds_files_are_rejected(cohort, tmp_path, damage, message):
    manifest_path, df = cohort
    table = folds_table(df, "patient_id", "patient_label", 3, 42)
    if damage == "patient":
        table = table.iloc[1:]
    elif damage == "label":
        table.loc[0, "label"] = 1 - table.loc[0, "label"]
    elif damage == "fold":
        table["fold"] = table["fold"].clip(upper=1)
    elif damage == "column":
        table = table.drop(columns="label")
    table.to_csv(tmp_path / "folds.csv", index=False)
    with pytest.raises(ValueError, match=message):
        fold_indices_from_csv(df, "patient_id", "patient_label", 3, tmp_path / "folds.csv")


def test_training_with_a_fixed_split_keeps_a_copy_and_does_not_write_folds_csv(cohort, tmp_path):
    manifest_path, df = cohort
    folds_table(df, "patient_id", "patient_label", 3, 42).to_csv(tmp_path / "given_folds.csv", index=False)
    cfg = make_cfg(manifest_path, 8, tmp_path / "outputs")
    cfg.trainer.max_epochs = 1
    cfg.data.folds_csv = str(tmp_path / "given_folds.csv")
    train_k_fold(cfg)
    run = tmp_path / "outputs" / cfg.logging.run_name
    assert (run / "folds_used.csv").read_text() == (tmp_path / "given_folds.csv").read_text()
    assert not (tmp_path / "outputs" / "folds.csv").exists()
    pooled = pd.concat([pd.read_csv(run / f"fold{k}_test_predictions.csv") for k in range(3)])
    assert sorted(pooled["patient_id"]) == sorted(df["patient_id"])
