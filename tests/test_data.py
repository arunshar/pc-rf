"""Unit tests for the dataset (data.py)."""
import torch

from pcrf import data as d


def test_schema_and_shapes():
    ds = d.SyntheticDownscaleDataset(n=4, fine=32, factor=4, sat_bands=4)
    s = ds[0]
    assert {"fine", "cond", "coarse_mass"}.issubset(s)
    assert s["fine"].shape == (3, 32, 32)
    assert s["cond"].shape == (3 + 4, 32, 32)   # upsampled coarse (3) + sat (4)
    assert s["coarse_mass"].ndim == 0


def test_truth_precip_is_nonnegative():
    ds = d.SyntheticDownscaleDataset(n=1, fine=32, factor=4)
    assert (ds[0]["fine"][0] >= 0).all()


def test_reproducible_with_seed():
    a = d.SyntheticDownscaleDataset(n=1, fine=32, seed=7)[0]["fine"]
    b = d.SyntheticDownscaleDataset(n=1, fine=32, seed=7)[0]["fine"]
    assert torch.equal(a, b)


def test_len():
    assert len(d.SyntheticDownscaleDataset(n=5, fine=16)) == 5


def test_fine_and_cond_same_resolution():
    s = d.SyntheticDownscaleDataset(n=1, fine=48, factor=4)[0]
    assert s["fine"].shape[-2:] == s["cond"].shape[-2:]
