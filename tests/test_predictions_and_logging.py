"""Local run outputs, prediction CSVs and the single threshold, on a tiny CPU run."""
import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

from conftest import make_cfg
from dataset.mil_dataset import MILDataModule
from lightning_modules.mil_classifier import PREDICTION_COLUMNS
from train_k_fold import train_one_fold, train_k_fold


def split_ids(cfg, fold_idx):
    dm = MILDataModule(cfg, fold_index=fold_idx)
    dm.setup("fit")
    return {name: list(ds.df["patient_id"]) for name, ds in
            [("train", dm.train_dataset), ("val", dm.val_dataset), ("test", dm.test_dataset)]}


@pytest.fixture
def two_fold_run(tmp_path, cohort):
    """Run two folds of a 3-fold cross-validation (the code needs at least 3 folds)."""
    manifest, _ = cohort
    cfg = make_cfg(manifest, dim=8, output_dir=tmp_path / "outputs")
    exp_dir = tmp_path / "outputs" / cfg.logging.run_name
    for fold_idx in (0, 1):
        train_one_fold(cfg, fold_idx, exp_dir)
    return cfg, exp_dir


def test_both_prediction_files_written_with_schema(two_fold_run):
    cfg, exp_dir = two_fold_run
    for fold_idx in (0, 1):
        ids = split_ids(cfg, fold_idx)
        for split in ("test", "val"):
            path = exp_dir / f"fold{fold_idx}_{split}_predictions.csv"
            assert path.exists(), path
            df = pd.read_csv(path)
            assert list(df.columns) == PREDICTION_COLUMNS
            assert sorted(df["patient_id"]) == sorted(ids[split])
            assert (df["fold"] == fold_idx).all()
            assert df["true_label"].isin([0, 1]).all()
            assert df["probability"].between(0, 1).all()


def test_prediction_and_probability_follow_the_threshold(two_fold_run):
    cfg, exp_dir = two_fold_run
    df = pd.read_csv(exp_dir / "fold0_test_predictions.csv")
    np.testing.assert_array_equal(df["prediction"], (df["probability"] > cfg.lit_module.threshold).astype(int))
    np.testing.assert_allclose(df["probability"], 1 / (1 + np.exp(-df["logit"])), atol=1e-3)


def test_val_file_belongs_to_best_validation_epoch(two_fold_run):
    """fold{k}_val_predictions.csv must come from the epoch with the highest val/auroc."""
    _, exp_dir = two_fold_run
    for fold_idx in (0, 1):
        val = pd.read_csv(exp_dir / f"fold{fold_idx}_val_predictions.csv")
        metrics = pd.read_csv(exp_dir / f"fold{fold_idx}" / "metrics.csv")
        best_logged = metrics["val/auroc"].dropna().max()
        assert roc_auc_score(val["true_label"], val["probability"]) == pytest.approx(best_logged, abs=1e-4)


def test_local_run_outputs(two_fold_run):
    cfg, exp_dir = two_fold_run
    fold_dir = exp_dir / "fold0"
    assert list((fold_dir / "checkpoints").glob("*.ckpt"))
    assert (fold_dir / "metrics.csv").exists()
    assert (fold_dir / "hparams.yaml").exists()
    # the threshold is recorded with every fold
    assert "threshold: 0.5" in (fold_dir / "hparams.yaml").read_text()
    # attention outputs per fold and per patient
    test_ids = split_ids(cfg, 0)["test"]
    attn = pd.read_csv(fold_dir / "attention_analysis.csv")
    assert sorted(attn["patient_id"]) == sorted(test_ids)
    per_patient = pd.read_csv(fold_dir / "attention_maps" / f"{test_ids[0]}.csv")
    assert list(per_patient.columns) == ["slide_id", "patch_id", "attention_score"]
    assert per_patient["attention_score"].sum() == pytest.approx(1.0, abs=1e-4)


def test_configured_threshold_is_used_for_saved_predictions(tmp_path, cohort):
    manifest, _ = cohort
    cfg = make_cfg(manifest, dim=8, output_dir=tmp_path / "outputs", threshold=0.3)
    exp_dir = tmp_path / "outputs" / cfg.logging.run_name
    train_one_fold(cfg, 0, exp_dir)
    df = pd.read_csv(exp_dir / "fold0_test_predictions.csv")
    np.testing.assert_array_equal(df["prediction"], (df["probability"] > 0.3).astype(int))
    assert "threshold: 0.3" in (exp_dir / "fold0" / "hparams.yaml").read_text()


def test_train_k_fold_writes_folds_csv_and_all_folds(tmp_path, cohort):
    manifest, df = cohort
    cfg = make_cfg(manifest, dim=8, output_dir=tmp_path / "outputs")
    cfg.trainer.max_epochs = 1
    train_k_fold(cfg)
    folds = pd.read_csv(tmp_path / "outputs" / "folds.csv")
    assert list(folds.columns) == ["patient_id", "fold", "label", "n_patches"]
    assert sorted(folds["patient_id"]) == sorted(df["patient_id"])
    exp_dir = tmp_path / "outputs" / cfg.logging.run_name
    pooled = pd.concat([pd.read_csv(exp_dir / f"fold{k}_test_predictions.csv") for k in range(3)])
    # every patient is tested exactly once, in the run that folds.csv names
    assert pooled["patient_id"].is_unique and len(pooled) == len(df)
    merged = pooled.merge(folds, on="patient_id")
    assert (merged["fold_x"] == merged["fold_y"]).all()
