"""The LAB statistics of the stain normalization are seeded and do not mutate the caller's list."""
import importlib.util
import sys

import numpy as np
import pytest
from skimage import io

from conftest import ROOT

PREPROCESSING = ROOT / "data_preparation_scripts" / "data_preprocessing"


@pytest.fixture(scope="module")
def color_norm():
    """The script name has a dot and a hyphen, so it cannot be imported by name."""
    sys.path.insert(0, str(PREPROCESSING))  # for `import stainNorm_Reinhard`
    try:
        spec = importlib.util.spec_from_file_location("color_normalization", PREPROCESSING / "1.6-color_normalization.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        sys.path.remove(str(PREPROCESSING))
    return module


@pytest.fixture
def patches(tmp_path):
    rng = np.random.default_rng(0)
    names = []
    for i in range(20):
        name = f"{i}.png"
        io.imsave(tmp_path / name, rng.integers(0, 256, size=(8, 8, 3), dtype=np.uint8), check_contrast=False)
        names.append(name)
    return tmp_path, names


def test_same_seed_gives_identical_statistics(color_norm, patches):
    path, names = patches
    first = color_norm.compute_mean_std(path, names, ratio=0.5, seed=7)
    second = color_norm.compute_mean_std(path, names, ratio=0.5, seed=7)
    np.testing.assert_array_equal(first[0], second[0])
    np.testing.assert_array_equal(first[1], second[1])


def test_default_seed_is_zero_and_other_seeds_differ(color_norm, patches):
    path, names = patches
    default = color_norm.compute_mean_std(path, names, ratio=0.5)
    zero = color_norm.compute_mean_std(path, names, ratio=0.5, seed=0)
    other = color_norm.compute_mean_std(path, names, ratio=0.5, seed=1)
    np.testing.assert_array_equal(default[0], zero[0])
    assert not np.array_equal(zero[0], other[0])


def test_callers_list_is_not_modified(color_norm, patches):
    path, names = patches
    before = list(names)
    color_norm.compute_mean_std(path, names, ratio=0.5, seed=3)
    assert names == before


def test_command_line_has_a_seed_argument_defaulting_to_zero():
    source = (PREPROCESSING / "1.6-color_normalization.py").read_text(encoding="utf-8")
    assert '"--seed", type=int, default=0' in source
