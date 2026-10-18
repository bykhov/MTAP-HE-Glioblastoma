"""The extraction config selects UNI, and a mismatching feature dimension fails loudly."""
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from conftest import ROOT, make_cohort, make_cfg
from dataset.mil_dataset import MILDataModule, expected_input_dim


def test_extract_config_selects_uni_and_matches_the_trainer():
    extract = OmegaConf.load(ROOT / "configs" / "extract_config.yaml")
    train = OmegaConf.load(ROOT / "configs" / "train_config.yaml")
    assert extract.feature_extractor.name == "uni"
    assert extract.feature_extractor.feature_dim == 1024
    assert train.model[train.model.name].input_dim == extract.feature_extractor.feature_dim
    # create_patient_h5s.py writes <slide features dir>-patient-level, which is what the trainer reads
    assert Path(train.data.parquet_path).parent.name == Path(extract.output_dir).name + "-patient-level"
    assert Path(train.data.parquet_path).parent.parent == Path(extract.output_dir).parent


def test_768d_features_with_1024d_model_raise_a_clear_error(tmp_path):
    manifest, _ = make_cohort(tmp_path, dim=768)
    cfg = make_cfg(manifest, dim=1024, output_dir=tmp_path / "out")
    datamodule = MILDataModule(cfg, fold_index=0)
    with pytest.raises(ValueError, match=r"stores 768-d features but model.params.input_dim is 1024"):
        datamodule.setup("fit")


def test_matching_dimension_passes(tmp_path):
    manifest, _ = make_cohort(tmp_path, dim=16)
    cfg = make_cfg(manifest, dim=16, output_dir=tmp_path / "out")
    MILDataModule(cfg, fold_index=0).setup("fit")


def test_expected_dim_falls_back_to_the_named_model_block():
    cfg = OmegaConf.create({"model": {"name": "clam_sb", "clam_sb": {"input_dim": 1024}}})
    assert expected_input_dim(cfg) == 1024
    assert expected_input_dim(OmegaConf.create({})) is None
