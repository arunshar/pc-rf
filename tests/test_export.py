"""Export test: build a tiny PC-RF, export to ONNX, and assert the ONNX graph
matches torch at a batch size different from the export example (catches an
export that baked in static shapes).
"""
import os
from pathlib import Path

import pytest

# These tests need the full export stack (torch.onnx.export needs the `onnx`
# package, plus onnxruntime to verify). Skip cleanly where any is absent (e.g.
# the python-core CI job installs serve deps but not [export]); the dedicated
# export-parity job installs [export] and runs them.
pytest.importorskip("torch")
pytest.importorskip("onnx")
pytest.importorskip("onnxruntime")
import export_onnx as ex  # noqa: E402


def test_export_roundtrip_crossshape(tmp_path):
    m, f, p, d = ex.load_pcrf_modules()
    ds = d.SyntheticDownscaleDataset(n=8, fine=32, factor=4, sat_bands=4)
    in_ch = ds[0]["fine"].shape[0]
    cond_ch = ds[0]["cond"].shape[0]

    net = ex.build_model(m, in_ch, cond_ch, base=16, attn=False)
    rf = f.RectifiedFlow(net, p.PhysicsLoss(p.ChannelSpec(**ex.CHANNEL_SPEC)))
    ex.quick_train(rf, ds, epochs=1, batch=4, device="cpu")

    out = tmp_path / "m.onnx"
    err = ex.export_velocity(net, in_ch, cond_ch, fine=32, out=out)

    assert out.exists()
    assert out.with_suffix(".json").exists()
    assert err is not None, "onnxruntime missing; cannot verify export"
    assert err < 1e-3, f"onnx/torch drift too large: {err}"


def test_metadata_records_tile(tmp_path):
    import json
    m, f, p, d = ex.load_pcrf_modules()
    ds = d.SyntheticDownscaleDataset(n=4, fine=32, factor=4, sat_bands=4)
    in_ch = ds[0]["fine"].shape[0]
    cond_ch = ds[0]["cond"].shape[0]
    net = ex.build_model(m, in_ch, cond_ch, base=16, attn=False)
    out = tmp_path / "m.onnx"
    ex.export_velocity(net, in_ch, cond_ch, fine=32, out=out)
    meta = json.loads(out.with_suffix(".json").read_text())
    assert meta["tile"] == 32
    assert meta["channel_spec"] == {"precip": 0, "u": 1, "v": 2}
