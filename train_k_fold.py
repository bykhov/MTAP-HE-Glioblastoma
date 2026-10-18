import copy
import shutil
from pathlib import Path

import pandas as pd
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import CSVLogger

from dataset.mil_dataset import MILDataModule, fold_indices_from_csv, folds_table
from lightning_modules.mil_classifier import MILClassifier
from models import get_model


def write_folds_csv(cfg, path):
    """
    Write the fold assignment of every patient to ``path`` (``patient_id, fold, label, n_patches``).
    ``fold`` is the run index in which the patient is in the test set; all runs reuse this file.
    """
    manifest = pd.read_parquet(cfg.data.parquet_path)
    table = folds_table(
        manifest,
        patient_id_col=cfg.data.case_id_col,
        label_col=cfg.data.label_col,
        k_folds=cfg.data.get('k_folds', 5),
        seed=cfg.seed,
    )
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(path, index=False)
    return table


def train_one_fold(cfg, fold_idx, exp_dir):
    """
    Train fold ``fold_idx`` and test it at the best-validation-AUROC checkpoint.
    Writes into ``exp_dir``: fold{k}_test_predictions.csv, fold{k}_val_predictions.csv and
    a fold{k}/ directory with the checkpoint, the CSV log and the attention outputs.
    """
    exp_dir = Path(exp_dir)
    fold_dir = exp_dir / f"fold{fold_idx}"

    datamodule = MILDataModule(cfg, fold_index=fold_idx)
    model = get_model(cfg.model.name, **cfg.model.params)
    lit_module = MILClassifier(model, copy.deepcopy(cfg), fold_index=fold_idx, output_dir=exp_dir)

    # --- Local logger: metrics.csv and hparams.yaml (includes the classification threshold) ---
    logger = CSVLogger(save_dir=str(exp_dir), name=f"fold{fold_idx}", version="")

    # --- Callbacks ---
    checkpoint_callback = ModelCheckpoint(
        dirpath=str(fold_dir / "checkpoints"),
        monitor="val/auroc",
        mode="max",
        filename=f"fold_{fold_idx}-{{epoch}}-{{val_auroc:.4f}}",
        save_top_k=1
    )
    early_stop_callback = EarlyStopping(**cfg.early_stopping)

    # --- Trainer ---
    trainer = pl.Trainer(
        **cfg.trainer,
        logger=logger,
        callbacks=[checkpoint_callback, early_stop_callback],
    )

    # --- Training ---
    trainer.fit(lit_module, datamodule=datamodule)

    # --- Testing ---
    # After training, load the best checkpoint and run the test loop. The checkpoint was written by this
    # process and stores the OmegaConf config; torch >= 2.6 needs weights_only=False to load it.
    print("--- Testing on hold-out set ---")
    return trainer.test(datamodule=datamodule, ckpt_path='best', weights_only=False)


def train_k_fold(cfg):
    """
    Main training routine for K-Fold Cross Validation using PyTorch Lightning.
    """
    pl.seed_everything(cfg.seed)

    k_folds = cfg.data.get('k_folds', 5)
    run_name = cfg.logging.run_name or "lightning_run"
    output_root = Path(cfg.logging.output_dir)
    exp_dir = output_root / run_name
    print(f"Starting {k_folds}-Fold Cross Validation for {cfg.model.name}...")
    print(f"Outputs go to {exp_dir}")

    # Fold assignment, written before the folds are trained
    if cfg.data.get('folds_csv'):
        # fixed split: validate the file against the manifest and keep a copy next to the predictions
        manifest = pd.read_parquet(cfg.data.parquet_path)
        fold_indices_from_csv(manifest, cfg.data.case_id_col, cfg.data.label_col, k_folds, cfg.data.folds_csv)
        exp_dir.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(cfg.data.folds_csv, exp_dir / "folds_used.csv")
    else:
        write_folds_csv(cfg, output_root / "folds.csv")

    for fold_idx in range(k_folds):
        print(f"\n===== Starting Fold {fold_idx + 1}/{k_folds} =====")
        train_one_fold(cfg, fold_idx, exp_dir)
