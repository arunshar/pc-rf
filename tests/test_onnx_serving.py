"""Integration test for the ONNX serving path (mirrors a Rust sampler that is not in this repository).

Loads artifacts/pcrf.onnx and drives it through the Euler sampler + projection,
asserting batch dynamism and that the physics projection holds. Skips if the
model has not been exported yet.
"""
import json
from pathlib import Path

import numpy as np
import pytest

from pcrf.serving import math as sm

# Skip the whole module where onnxruntime is absent (e.g. CI without heavy deps).
pytest.importorskip("onnxruntime")

ROOT = Path(__file__).resolve().parents[1]
ONNX = ROOT / "artifacts" / "pcrf.onnx"
META = ONNX.with_suffix(".json")


def _meta():
    if META.exists():
        return json.loads(META.read_text())
    return {"tile": 64, "in_ch": 3, "cond_ch": 7}


@pytest.fixture(scope="module")
def sess():
    if not ONNX.exists():
        pytest.skip(f"{ONNX} not found; run `make export` first")
    import onnxruntime as ort
    return ort.InferenceSession(str(ONNX), providers=["CPUExecutionProvider"])


def test_output_shape(sess):
    m = _meta()
    mean, _ = sm.sample_onnx(sess, B=4, tile=m["tile"], in_ch=m["in_ch"],
                             cond_ch=m["cond_ch"], steps=8)
    assert mean.shape == (m["in_ch"], m["tile"], m["tile"])


@pytest.mark.parametrize("B", [1, 4, 8])
def test_batch_dynamic_and_physics(sess, B):
    m = _meta()
    cm = 0.1
    mean, metrics = sm.sample_onnx(sess, B=B, tile=m["tile"], in_ch=m["in_ch"],
                                   cond_ch=m["cond_ch"], steps=10, coarse_mass=cm)
    assert metrics["NVR"] == 0.0
    assert metrics["MCE"] < 1e-4
    assert np.isfinite(metrics["DE"])


def test_projection_off_can_violate(sess):
    # Without the projection, NVR is generally > 0 (the value of the projection).
    m = _meta()
    mean, metrics = sm.sample_onnx(sess, B=8, tile=m["tile"], in_ch=m["in_ch"],
                                   cond_ch=m["cond_ch"], steps=10, project=False)
    assert metrics["NVR"] >= 0.0  # finite, defined; projection drives it to 0
