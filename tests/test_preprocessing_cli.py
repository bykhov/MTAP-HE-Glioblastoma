"""Preprocessing stages 1.2 to 1.6 on a fake slide object that stands in for OpenSlide, so no WSI is needed."""
import importlib.util
import json
import sys

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from conftest import ROOT

PREPROCESSING = ROOT / "data_preparation_scripts" / "data_preprocessing"
TILE = 512


class FakeSlide:
    """A slide whose 512-tile grid has tissue (pink noise) on a checkerboard and white background elsewhere."""

    def __init__(self, mag, tiles_w=3, tiles_h=2, mpp=0.5):
        self.region = int(TILE * mag / 20)
        self.level_dimensions = [(tiles_w * self.region + 37, tiles_h * self.region + 11)]  # edge remainder is dropped
        self.level_downsamples = [1.0]
        self.properties = {"aperio.AppMag": str(mag), "openslide.mpp-x": str(mpp), "openslide.mpp-y": str(mpp)}
        self.calls = []

    def read_region(self, location, level, size):
        self.calls.append((location, level, tuple(size)))
        i, j = location[0] // self.region, location[1] // self.region
        rng = np.random.default_rng(1000 * i + j)
        if (i + j) % 2 == 0:  # tissue
            pixels = np.clip(rng.normal([200, 90, 150], 25, (size[1], size[0], 3)), 0, 255).astype(np.uint8)
        else:
            pixels = np.full((size[1], size[0], 3), 255, np.uint8)
        return Image.fromarray(pixels).convert("RGBA")

    def get_thumbnail(self, size):
        return Image.new("RGB", size, (230, 200, 220))

    def close(self):
        pass


def load_stage(file_name, module_name):
    """Load a stage script whose name has dots/hyphens; its sibling `utils` module must win over the root package."""
    sys.path.insert(0, str(PREPROCESSING))
    saved = sys.modules.pop("utils", None)
    try:
        spec = importlib.util.spec_from_file_location(module_name, PREPROCESSING / file_name)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        sys.path.remove(str(PREPROCESSING))
        sys.modules.pop("utils", None)
        if saved is not None:
            sys.modules["utils"] = saved
    return module


@pytest.fixture
def raw_data(tmp_path):
    """raw_data/<case>/<slide>/<slide>.svs (an empty file: the fake slide replaces OpenSlide)."""
    base = tmp_path / "raw_data"
    for case, slide in [("TCGA-AA-0001", "TCGA-AA-0001-01Z-00-DX1"), ("TCGA-BB-0002", "TCGA-BB-0002-01Z-00-DX1")]:
        folder = base / case / slide
        folder.mkdir(parents=True)
        (folder / f"{slide}.svs").write_bytes(b"")
    return base


def test_tiling_grid_and_resume(raw_data, monkeypatch):
    stage = load_stage("1.3-extract_patches.py", "stage13")
    fakes = {}
    monkeypatch.setattr(stage, "open_slide", lambda path: fakes.setdefault(path, FakeSlide(20)))
    slide_dir = raw_data / "TCGA-AA-0001" / "TCGA-AA-0001-01Z-00-DX1"
    assert stage.process_slide_dir(slide_dir) == 3          # 3 x 2 grid, checkerboard: tiles 1, 4 and 5 hold tissue
    names = sorted(int(p.stem) for p in (slide_dir / "patches").glob("*.jpg"))
    # patch id = running count over the grid, columns first (x outer loop, y inner loop), starting at 1
    assert names == [1, 4, 5]
    assert Image.open(slide_dir / "patches" / "1.jpg").size == (TILE, TILE)
    assert (slide_dir / "mask" / "TCGA-AA-0001-01Z-00-DX1_mask.png").exists()
    reads = len(next(iter(fakes.values())).calls)
    assert stage.process_slide_dir(slide_dir) is None        # finished slides are skipped
    assert len(next(iter(fakes.values())).calls) == reads


def test_forty_x_slides_read_a_1024_px_region_and_resize_it_to_512(raw_data, monkeypatch):
    stage = load_stage("1.3-extract_patches.py", "stage13")
    slide = FakeSlide(40)
    monkeypatch.setattr(stage, "open_slide", lambda path: slide)
    slide_dir = raw_data / "TCGA-AA-0001" / "TCGA-AA-0001-01Z-00-DX1"
    stage.process_slide_dir(slide_dir)
    assert {call[2] for call in slide.calls} == {(1024, 1024)} and {call[1] for call in slide.calls} == {0}
    assert Image.open(slide_dir / "patches" / "1.jpg").size == (TILE, TILE)


def test_size_info_and_slide_properties(raw_data, monkeypatch):
    stage = load_stage("1.4-get_size_info.py", "stage14")
    monkeypatch.setattr(stage, "open_slide", lambda path: FakeSlide(40, mpp=0.25))
    slide_dir = raw_data / "TCGA-AA-0001" / "TCGA-AA-0001-01Z-00-DX1"
    stage.process_slide_dir(slide_dir)
    info = pd.read_pickle(slide_dir / "slide_size_info.pickle")
    assert info.columns.tolist() == ["Slide_name", "Mag", "Height", "Width"] and info.loc[0, "Mag"] == 40
    props = json.loads((slide_dir / "slide_properties.json").read_text())
    assert props["aperio.AppMag"] == 40.0 and props["region_read_px"] == 1024 and props["resize_factor"] == 2.0
    assert props["level_used"] == 0 and props["effective_um_per_px_of_tile"] == pytest.approx(0.5)


def test_thumbnail_and_coordinates(raw_data, monkeypatch):
    thumb = load_stage("1.2-create thumbnail.py", "stage12")
    monkeypatch.setattr(thumb, "open_slide", lambda path: FakeSlide(20))
    thumb.run_over_slides(thumb.process_slide_dir, raw_data)
    assert (raw_data / "TCGA-AA-0001" / "TCGA-AA-0001-01Z-00-DX1" / "TCGA-AA-0001-01Z-00-DX1_thumbnail.png").exists()

    coords = load_stage("1.3.1-extract_patch_coordinates.py", "stage131")
    monkeypatch.setattr(coords, "open_slide", lambda path: FakeSlide(20))
    coords.extract_patch_coordinates(str(raw_data))
    table = pd.read_parquet(raw_data / "patch_coordinates.parquet")
    assert sorted(table[table.slide_id == "TCGA-AA-0001-01Z-00-DX1"]["patch_id"]) == [1, 4, 5]


def test_coordinates_follow_the_patch_id_order(raw_data, monkeypatch):
    coords = load_stage("1.3.1-extract_patch_coordinates.py", "stage131")
    monkeypatch.setattr(coords, "open_slide", lambda path: FakeSlide(20))
    coords.extract_patch_coordinates(str(raw_data))
    table = pd.read_parquet(raw_data / "patch_coordinates.parquet")
    one = table[table.slide_id == "TCGA-AA-0001-01Z-00-DX1"].sort_values("patch_id")
    # grid of 3 columns x 2 rows, y runs fastest: id 1 -> (0,0), 2 -> (0,512), 3 -> (512,0), 4 -> (512,512), 5 -> (1024,0)
    assert dict(zip(one["patch_id"], map(tuple, one[["x", "y"]].values))) == {1: (0, 0), 4: (512, 512), 5: (1024, 0)}


def test_kmeans_selects_the_tissue_patches(raw_data, monkeypatch):
    tiling = load_stage("1.3-extract_patches.py", "stage13")
    monkeypatch.setattr(tiling, "open_slide", lambda path: FakeSlide(20, tiles_w=6, tiles_h=4))
    size = load_stage("1.4-get_size_info.py", "stage14")
    monkeypatch.setattr(size, "open_slide", lambda path: FakeSlide(20, tiles_w=6, tiles_h=4))
    slide_dir = raw_data / "TCGA-AA-0001" / "TCGA-AA-0001-01Z-00-DX1"
    tiling.process_slide_dir(slide_dir)
    size.process_slide_dir(slide_dir)
    # add faint-tissue patches next to the dense ones so that k = 3 has a background cluster to remove
    for index in (100, 101, 102):
        Image.fromarray(np.full((TILE, TILE, 3), 248, np.uint8)).save(slide_dir / "patches" / f"{index}.jpg")
    Image.new("RGB", (60, 40), (230, 200, 220)).save(slide_dir / "TCGA-AA-0001-01Z-00-DX1_thumbnail.png")
    kmeans = load_stage("1.5-kmeans_clustering.py", "stage15")
    kmeans.process_slide_dir(slide_dir)
    result = np.load(slide_dir / "clustering_results" / "npy_files" / "cluster.npy", allow_pickle=True).item()
    all_ids = sorted(int(p.stem) for p in (slide_dir / "patches").glob("*.jpg"))
    assert sorted(result["0"].tolist() + result["1"].tolist()) == all_ids  # every patch is in exactly one cluster
    assert {100, 101, 102} <= set(result["0"].tolist())                      # near-white patches are background
    assert (slide_dir / "clustering_results" / "png_files" / "cluster.png").exists()
    kmeans.process_slide_dir(slide_dir)                                     # second call: skipped, no error


def test_run_over_slides_visits_every_slide_folder_and_supports_workers(raw_data):
    sys.path.insert(0, str(PREPROCESSING))
    saved = sys.modules.pop("utils", None)
    try:
        import utils as stage_utils
    finally:
        sys.path.remove(str(PREPROCESSING))
        sys.modules.pop("utils", None)
        if saved is not None:
            sys.modules["utils"] = saved
    folders = [p.name for p in stage_utils.iter_slide_dirs(raw_data)]
    assert folders == ["TCGA-AA-0001-01Z-00-DX1", "TCGA-BB-0002-01Z-00-DX1"]
    assert stage_utils.run_over_slides(str, raw_data) == [str(p) for p in stage_utils.iter_slide_dirs(raw_data)]
    assert stage_utils.run_over_slides(str, raw_data, workers=2) == [str(p) for p in stage_utils.iter_slide_dirs(raw_data)]


def test_coordinates_workers_are_passed_on_and_rows_are_unchanged(raw_data, monkeypatch):
    coords = load_stage("1.3.1-extract_patch_coordinates.py", "stage131")
    monkeypatch.setattr(coords, "open_slide", lambda path: FakeSlide(20))
    coords.extract_patch_coordinates(str(raw_data))
    sequential = pd.read_parquet(raw_data / "patch_coordinates.parquet")
    (raw_data / "patch_coordinates.parquet").unlink()

    seen = []
    original = coords.run_over_slides
    monkeypatch.setattr(coords, "run_over_slides", lambda func, base, workers=1: seen.append(workers) or original(func, base, 1))
    coords.extract_patch_coordinates(str(raw_data), workers=4)
    assert seen == [4]
    pd.testing.assert_frame_equal(pd.read_parquet(raw_data / "patch_coordinates.parquet"), sequential)
