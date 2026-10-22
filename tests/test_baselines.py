"""
CPU gate for the PC-RF Table-2 baselines. Uses the toy ERA5 writer (no download)
to train each baseline for one epoch and evaluate it, asserting the five reported
metrics are present and finite and that the JSON is written under the canonical
row label make_tables.py expects.
"""
import math
from types import SimpleNamespace

import pytest

pytest.importorskip("xarray")
pytest.importorskip("netCDF4")

import pcrf.data as d
from pcrf.baselines import METHODS, LABEL, train_baseline, evaluate_baseline


def _args(coarse, target, out, jout, method):
    return SimpleNamespace(
        mode="train", baseline=method, data="era5", root="",
        coarse=coarse, target=target, scenes="", n=8, fine=32, factor=4,
        sat_bands=4, epochs=1, batch=2, lr=1e-3, base=16, ddpm_T=20,
        steps=4, eta=1.0, ensemble=2, adv_weight=1e-3, out=out, resume=False,
        baseline_ckpt="", json_out=jout, json_method="", json_dataset="era5",
        device="cpu",
    )


@pytest.mark.parametrize("method", METHODS)
def test_baseline_trains_and_scores(method, tmp_path):
    cpath = tmp_path / "coarse.nc"
    tpath = tmp_path / "target.nc"
    d.write_toy_era5(str(cpath), str(tpath), n_time=20, hc=16, wc=16, hf=32, wf=32)

    out = tmp_path / f"run_{method}"
    jout = tmp_path / f"metrics_{method}.json"
    args = _args(str(cpath), str(tpath), str(out), str(jout), method)

    train_baseline(method, args)          # learned: 1 epoch; bicubic: no-op
    means = evaluate_baseline(args)

    for k in ["PSNR", "SSIM", "CRPS", "DE", "NVR", "MCE"]:
        assert k in means, f"{method}: missing metric {k}"
        assert math.isfinite(means[k]), f"{method}: non-finite {k}={means[k]}"

    import json
    blob = json.loads(jout.read_text())
    assert LABEL[method] in blob["era5"], f"{method}: json missing {LABEL[method]}"
    assert set(["PSNR", "SSIM", "CRPS", "DE", "NVR", "MCE"]).issubset(
        blob["era5"][LABEL[method]])


def test_deterministic_crps_is_mae_like():
    """A single-member (deterministic) ensemble yields CRPS == mean|pred-target|."""
    import torch
    from pcrf.metrics import crps_ensemble
    pred = torch.randn(2, 3, 8, 8)
    target = torch.randn(2, 3, 8, 8)
    crps = crps_ensemble(pred.unsqueeze(0), target)
    mae = (pred - target).abs().mean().item()
    assert abs(crps - mae) < 1e-5
