"""configs/train_config.yaml itself drives a run (with only the data and the hardware overridden)."""
import pandas as pd
from omegaconf import OmegaConf

from conftest import ROOT, make_cohort
from train_k_fold import train_k_fold


def test_train_config_runs_end_to_end(tmp_path):
    manifest, df = make_cohort(tmp_path, dim=1024)  # UNI-sized features
    cfg = OmegaConf.load(ROOT / "configs" / "train_config.yaml")
    cfg.model.params = cfg.model[cfg.model.name]  # what run.py does
    # overrides: synthetic data, CPU, short run. Everything else is the repository config.
    cfg.data.parquet_path = str(manifest)
    cfg.data.k_folds = 3
    cfg.datamodule.num_workers = 0
    cfg.datamodule.persistent_workers = False
    cfg.trainer.update(accelerator="cpu", precision=32, max_epochs=1, enable_progress_bar=False,
                       enable_model_summary=False)
    cfg.logging.output_dir = str(tmp_path / "outputs")

    assert cfg.lit_module.threshold == 0.5
    train_k_fold(cfg)

    exp_dir = tmp_path / "outputs" / cfg.logging.run_name
    pooled = pd.concat([pd.read_csv(exp_dir / f"fold{k}_test_predictions.csv") for k in range(3)])
    assert sorted(pooled["patient_id"]) == sorted(df["patient_id"])
    assert (tmp_path / "outputs" / "folds.csv").exists()
