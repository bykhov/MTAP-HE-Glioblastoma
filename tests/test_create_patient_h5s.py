"""create_patient_h5s.py end to end on synthetic slides."""
import h5py
import numpy as np
import pandas as pd
import pytest

from conftest import make_cfg
from create_patient_h5s import create_patient_h5_files, parse_args
from dataset.mil_dataset import MILDataModule, MILDataset, collate_bags

DIM = 8


def write_slide_h5(path, patch_ids, rng):
    """Slide-level file as written by extract_features.py: one dataset per patch id."""
    with h5py.File(path, "w") as f:
        for patch_id in patch_ids:
            f.create_dataset(str(patch_id), data=rng.normal(size=DIM).astype(np.float32))


def make_slide_inputs(tmp_path, labels, patches_per_slide=5, slides_per_patient=2):
    rng = np.random.default_rng(1)
    rows = []
    for i, label in enumerate(labels):
        for s in range(slides_per_patient):
            slide_id = f"TCGA-00-{i:04d}-0{s}Z-00-DX1"
            h5_path = tmp_path / f"{slide_id}.h5"
            write_slide_h5(h5_path, range(1, patches_per_slide + 1), rng)
            rows.append({"case_id": f"TCGA-00-{i:04d}", "slide_id": slide_id,
                         "reference_to_slide_h5": str(h5_path), "slide_label": label})
    manifest = tmp_path / "slides" / "feature_manifest.parquet"
    manifest.parent.mkdir()
    pd.DataFrame(rows).to_parquet(manifest, index=False)
    return manifest


def run(tmp_path, slide_manifest):
    out_dir = tmp_path / "patients"
    manifest_out = tmp_path / "manifests" / "feature_manifest.parquet"  # a different folder than out_dir
    df = create_patient_h5_files(slide_manifest, out_dir, manifest_out)
    return out_dir, manifest_out, df


def test_manifest_has_the_columns_the_trainer_needs(tmp_path):
    out_dir, manifest_out, df = run(tmp_path, make_slide_inputs(tmp_path, labels=[0, 1, 1]))
    assert manifest_out.is_file() and out_dir.is_dir()
    loaded = pd.read_parquet(manifest_out)
    assert list(loaded.columns) == ["patient_id", "patient_label", "n_patches", "n_slides", "patient_h5_path"]
    assert list(loaded["patient_id"]) == ["TCGA-00-0000", "TCGA-00-0001", "TCGA-00-0002"]
    assert list(loaded["patient_label"]) == [0, 1, 1]
    assert (loaded["n_patches"] == 10).all() and (loaded["n_slides"] == 2).all()


def test_patient_h5_concatenates_the_slides_in_slide_order(tmp_path):
    slide_manifest = make_slide_inputs(tmp_path, labels=[0, 1, 1])
    out_dir, _, df = run(tmp_path, slide_manifest)
    row = df.iloc[0]
    with h5py.File(row["patient_h5_path"], "r") as f:
        features = f["features"][:]
        patch_ids = [p.decode() for p in f["patch_ids"][:]]
    assert features.shape == (10, DIM)
    assert patch_ids[0].startswith("TCGA-00-0000-00Z-00-DX1-") and patch_ids[-1].startswith("TCGA-00-0000-01Z-00-DX1-")
    # the first row is the first feature of the first slide
    slides = pd.read_parquet(slide_manifest)
    with h5py.File(slides.sort_values("slide_id").iloc[0]["reference_to_slide_h5"], "r") as f:
        first_key = list(f.keys())[0]
        np.testing.assert_array_equal(features[0], f[first_key][()])
        assert patch_ids[0].endswith("-" + first_key)


def test_slides_with_different_labels_are_rejected(tmp_path):
    slide_manifest = make_slide_inputs(tmp_path, labels=[0, 1, 1])
    slides = pd.read_parquet(slide_manifest)
    slides.loc[1, "slide_label"] = 1  # second slide of patient 0 disagrees with the first
    slides.to_parquet(slide_manifest, index=False)
    with pytest.raises(ValueError, match="TCGA-00-0000 has slides with different labels"):
        run(tmp_path, slide_manifest)


def test_unreadable_slide_is_skipped_and_counted(tmp_path, capsys):
    slide_manifest = make_slide_inputs(tmp_path, labels=[0, 1, 1])
    slides = pd.read_parquet(slide_manifest)
    slides.loc[0, "reference_to_slide_h5"] = str(tmp_path / "missing.h5")
    slides.to_parquet(slide_manifest, index=False)
    _, _, df = run(tmp_path, slide_manifest)
    first = df.iloc[0]
    assert first["n_slides"] == 1 and first["n_patches"] == 5
    assert "Skipping slide" in capsys.readouterr().out


def test_command_line_arguments():
    args = parse_args(["--slide-manifest", "a.parquet", "--out-dir", "patients", "--manifest-out", "m.parquet"])
    assert (args.slide_manifest, args.out_dir, args.manifest_out) == ("a.parquet", "patients", "m.parquet")
    with pytest.raises(SystemExit):
        parse_args([])


def test_manifest_loads_in_the_data_module_without_modification(tmp_path):
    """3 patients x 2 slides are too few for a stratified split, so this uses 6 (labels 0,0,0,1,1,1)."""
    _, manifest_out, _ = run(tmp_path, make_slide_inputs(tmp_path, labels=[0, 0, 0, 1, 1, 1]))
    cfg = make_cfg(manifest_out, DIM, tmp_path / "out")
    datamodule = MILDataModule(cfg, fold_index=0)
    datamodule.setup("fit")
    assert len(datamodule.train_dataset) + len(datamodule.val_dataset) + len(datamodule.test_dataset) == 6
    instances, labels, mask, bag_ids, patch_ids = next(iter(datamodule.test_dataloader()))
    assert instances.shape[1:] == (10, DIM) and mask.all()
    assert len(bag_ids) == len(labels) == instances.shape[0]


def test_dataset_reads_the_three_patient_manifest(tmp_path):
    _, manifest_out, df = run(tmp_path, make_slide_inputs(tmp_path, labels=[0, 1, 1]))
    cfg = make_cfg(manifest_out, DIM, tmp_path / "out")
    dataset = MILDataset(pd.read_parquet(manifest_out), cfg.data)
    instances, labels, mask, bag_ids, _ = collate_bags([dataset[i] for i in range(3)])
    assert instances.shape == (3, 10, DIM) and list(labels) == [0, 1, 1]
