"""
CPU gate for the Sentinel-2 super-resolution path: the SAM metric, the
rasterio-backed Sentinel2Dataset loader (fake GeoTIFF tiles), and the offline
pieces of pcrf.s2_ingest (no network).
"""
import numpy as np
import pytest
import torch

from pcrf.metrics import sam


def test_sam_identical_is_near_zero():
    # arccos near 1.0 amplifies float32 noise, so identical fields give a tiny
    # but nonzero angle (~0.01 deg); that is physically zero (real SAM is degrees).
    x = torch.rand(2, 3, 8, 8) + 0.1
    assert sam(x, x) < 0.05


def test_sam_orthogonal_is_90_degrees():
    pred = torch.zeros(1, 2, 4, 4); pred[:, 0] = 1.0
    tgt = torch.zeros(1, 2, 4, 4); tgt[:, 1] = 1.0
    assert abs(sam(pred, tgt) - 90.0) < 1e-2


def _write_tif(path, arr):
    import rasterio
    a = np.asarray(arr, dtype="float32")[None]
    with rasterio.open(path, "w", driver="GTiff", height=a.shape[1],
                       width=a.shape[2], count=1, dtype="float32") as dst:
        dst.write(a)


def test_sentinel2_dataset_shapes(tmp_path):
    pytest.importorskip("rasterio")
    from pcrf.data import Sentinel2Dataset
    rng = np.random.default_rng(0)
    scenes = []
    for k in range(3):
        rec = {"target": []}
        for b in ("b3", "b4", "b8"):
            c = tmp_path / f"{k}_{b}_20m.tif"
            t = tmp_path / f"{k}_{b}_10m.tif"
            _write_tif(c, rng.random((16, 16), dtype="float32"))   # 20m coarse
            _write_tif(t, rng.random((32, 32), dtype="float32"))   # 10m target
            rec[b] = str(c)
            rec["target"].append(str(t))
        scenes.append(rec)
    ds = Sentinel2Dataset(scenes, tile=32)
    assert len(ds) == 3
    s = ds[0]
    assert tuple(s["fine"].shape) == (3, 32, 32)      # 3 target bands
    assert tuple(s["cond"].shape) == (6, 32, 32)      # 3 upsampled bands + 3 indices
    assert torch.isfinite(s["fine"]).all()
    assert torch.isfinite(s["cond"]).all()


def test_s2_ingest_block_avg_and_io(tmp_path):
    pytest.importorskip("rasterio")
    from pcrf.s2_ingest import block_avg2, write_tif
    a = np.arange(64, dtype="float32").reshape(8, 8)
    c = block_avg2(a)
    assert c.shape == (4, 4)
    assert abs(float(c[0, 0]) - (0 + 1 + 8 + 9) / 4.0) < 1e-5   # top-left 2x2 mean
    p = tmp_path / "x.tif"
    write_tif(p, c)
    import rasterio
    with rasterio.open(p) as ds:
        back = ds.read(1)
    assert np.allclose(back, c, atol=1e-5)
