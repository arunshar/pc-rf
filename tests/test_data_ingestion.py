"""Tests for the real-data loaders (data.py).

The spectral-index math is pure and always runs. The ERA5 loader is exercised
end to end against a tiny synthetic NetCDF; those tests skip where xarray /
netCDF4 are not installed (e.g. a minimal CI image).
"""
import numpy as np
import pytest

from pcrf import data as d


def test_spectral_indices_known_values():
    b3, b4, b8 = np.array([[0.2]]), np.array([[0.1]]), np.array([[0.5]])
    ndvi, ndwi, savi = d.spectral_indices(b3, b4, b8)
    assert np.allclose(ndvi, 0.4 / 0.6, atol=1e-4)
    assert np.allclose(ndwi, -0.3 / 0.7, atol=1e-4)
    assert np.allclose(savi, 1.5 * 0.4 / 1.1, atol=1e-4)


def test_spectral_indices_ndvi_in_range():
    rng = np.random.default_rng(0)
    b3, b4, b8 = (rng.random((8, 8)) for _ in range(3))
    ndvi, _, _ = d.spectral_indices(b3, b4, b8)
    assert np.all(ndvi >= -1.001) and np.all(ndvi <= 1.001)


def test_era5_loader_schema(tmp_path):
    pytest.importorskip("xarray")
    pytest.importorskip("netCDF4")
    cpath, tpath = tmp_path / "coarse.nc", tmp_path / "target.nc"
    d.write_toy_era5(str(cpath), str(tpath), n_time=6, hc=16, wc=16, hf=64, wf=64)

    ds = d.ERA5Dataset(str(cpath), str(tpath), split="train", fine=64)
    assert len(ds) >= 1
    s = ds[0]
    assert s["fine"].shape == (3, 64, 64)
    assert s["cond"].shape == (3, 64, 64)        # sat_bands=0
    assert s["coarse_mass"].ndim == 0


def test_era5_split_partitions_time(tmp_path):
    pytest.importorskip("xarray")
    pytest.importorskip("netCDF4")
    cpath, tpath = tmp_path / "c.nc", tmp_path / "t.nc"
    d.write_toy_era5(str(cpath), str(tpath), n_time=6, hf=48, wf=48)
    lens = [len(d.ERA5Dataset(str(cpath), str(tpath), sp, fine=48))
            for sp in ("train", "val", "test")]
    assert sum(lens) == 6


def test_era5_loader_with_sat_bands(tmp_path):
    pytest.importorskip("xarray")
    pytest.importorskip("netCDF4")
    cpath, tpath = tmp_path / "c.nc", tmp_path / "t.nc"
    d.write_toy_era5(str(cpath), str(tpath), n_time=4, hf=48, wf=48)
    ds = d.ERA5Dataset(str(cpath), str(tpath), "train", fine=48, sat_bands=4)
    assert ds[0]["cond"].shape == (3 + 4, 48, 48)
