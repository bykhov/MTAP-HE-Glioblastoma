"""Shared helpers: tiny synthetic cohorts and a small CPU config. No real data is needed."""
# Environment quirk (Windows, torchvision 0.26 with Pillow 12.2): importing torchvision, which timm does,
# fails with "DLL load failed while importing _imaging" unless Pillow was loaded before it.
import PIL.Image  # noqa: F401  (must stay the first import)

import os
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pytest
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
# The parquet tables (patient manifest, slide table, tile coordinates) are read from the repository, or from
# the folder named by MTAP_DATA_ROOT if it is set. Tests that need them skip if they are absent.
DATA_ROOT = Path(os.environ.get("MTAP_DATA_ROOT", ROOT))
REAL_MANIFEST = DATA_ROOT / "source" / "precomputed_features" / "uni-patient-level" / "feature_manifest.parquet"
REAL_SLIDE_TABLE = DATA_ROOT / "source" / "dataset_v1.parquet"
REAL_COORDS = DATA_ROOT / "utils" / "patch_coordinates_combined.parquet"
FIXTURES = Path(__file__).resolve().parent / "fixtures"


def write_feature_h5(path, n_patches, dim, rng, shift=0.0, slide_id="TCGA-AA-0000-01Z-00-DX1"):
    """A patient/slide H5 file with the layout the code reads: 'features' and 'patch_ids'."""
    features = rng.normal(shift, 1.0, size=(n_patches, dim)).astype(np.float32)
    patch_ids = np.array([f"{slide_id}-{i}" for i in range(n_patches)], dtype=h5py.string_dtype(encoding="utf-8"))
    with h5py.File(path, "w") as f:
        f.create_dataset("features", data=features)
        f.create_dataset("patch_ids", data=patch_ids)


def make_cohort(tmp_path, n_patients=24, dim=8, seed=0):
    """
    Patient-level cohort with the columns the trainer needs. Four equally sized strata
    (label x small/large bag), so a 3-fold stratified split is always possible.
    Returns the manifest path and the DataFrame.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_patients):
        label = i % 2
        n_patches = 10 if (i // 2) % 2 == 0 else 30
        patient_id = f"TCGA-00-{i:04d}"
        h5_path = tmp_path / f"{patient_id}.h5"
        write_feature_h5(h5_path, n_patches, dim, rng, shift=0.8 * label, slide_id=patient_id + "-01Z-00-DX1")
        rows.append({"patient_id": patient_id, "patient_label": label, "n_patches": n_patches,
                     "patient_h5_path": str(h5_path)})
    df = pd.DataFrame(rows)
    manifest = tmp_path / "feature_manifest.parquet"
    df.to_parquet(manifest, index=False)
    return manifest, df


def make_cfg(manifest, dim, output_dir, k_folds=3, **lit_overrides):
    """A small CPU version of configs/train_config.yaml, with model.params set the way run.py does."""
    cfg = OmegaConf.create({
        "seed": 42,
        "pipeline": "lightning",
        "data": {
            "parquet_path": str(manifest),
            "case_id_col": "patient_id",
            "bag_id_col": "patient_id",
            "label_col": "patient_label",
            "bag_size_col": "bag_size",
            "feature_h5_path_col": "patient_h5_path",
            "val_split": 0.10,
            "test_split": 0.10,
            "k_folds": k_folds,
            "max_bag_size_for_train": 64,
        },
        "datamodule": {"batch_size": 4, "num_workers": 0, "persistent_workers": False,
                       "use_weighted_sampler": True},
        "model": {
            "name": "clam_sb",
            "clam_sb": {"input_dim": dim, "n_classes": 1, "gated": True, "att_dim": 4, "k_sample": 2},
        },
        "lit_module": {
            "pos_weight": 0.5959,
            "threshold": 0.5,
            "optimizer": {"name": "AdamW", "lr": 0.001, "weight_decay": 0.01},
            "scheduler": {"name": "CosineAnnealingLR", "params": {"T_max": 4}},
            "patch_dropout_rate": 0.0,
            "patch_dropout_small_bag_threshold": 500,
        },
        "trainer": {"max_epochs": 3, "accelerator": "cpu", "devices": 1, "log_every_n_steps": 1,
                    "gradient_clip_val": 0.5, "precision": 32, "enable_progress_bar": False,
                    "enable_model_summary": False},
        "early_stopping": {"monitor": "val/auroc", "patience": 50, "mode": "max"},
        "logging": {"run_name": "test_run", "output_dir": str(output_dir)},
    })
    cfg.lit_module.update(lit_overrides)
    cfg.model.params = cfg.model[cfg.model.name]
    return cfg


@pytest.fixture
def cohort(tmp_path):
    manifest, df = make_cohort(tmp_path)
    return manifest, df
