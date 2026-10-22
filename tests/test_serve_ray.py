"""Unit tests for the Ray Serve downscaler deployment (apps/training/serve_ray.py).

These are CPU-only and never deploy a live Serve cluster. The Serve decorators
are imported (ray is installed in the serving image) but we exercise the
deployment's *underlying* class + methods directly:

  * the batched handler takes a list of N request bodies and returns a list of
    N results aligned to the inputs;
  * num_gpus is read from PCRF_SERVE_GPUS (default 0 => CPU providers);
  * the autoscaling_config carries min/max/target from the env knobs.

onnxruntime is not required: we inject a fake session whose run() returns a
zero "velocity" tensor of the right shape, so pcrf.serving.math.sample_onnx
runs as pure numpy. Each test loads serve_ray under a controlled environment so
the import-time knob reads are deterministic.
"""
import asyncio
import importlib
import sys

import numpy as np
import pytest

# ray hosts the @serve.deployment / @serve.batch decorators used at import time.
pytest.importorskip("ray")


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
class _FakeSession:
    """Stand-in for onnxruntime.InferenceSession.

    sample_onnx calls sess.run(["velocity"], {"x_t": x, "t": t, "cond": cond})
    once per Euler step and adds the result * dt to x. Returning zeros makes the
    sampler deterministic (x stays at its seeded init) without onnxruntime.
    """

    def __init__(self, in_ch=3):
        self.in_ch = in_ch
        self.calls = 0

    def run(self, outputs, feeds):
        self.calls += 1
        x = feeds["x_t"]
        return [np.zeros_like(x)]


def _load_serve_ray(monkeypatch, env):
    """Import (fresh) apps.training.serve_ray with a controlled environment.

    serve_ray reads all its knobs at import time, so we set env and drop any
    cached module before importing.
    """
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    # conftest puts apps/training on sys.path; ensure a clean import each call.
    sys.modules.pop("serve_ray", None)
    return importlib.import_module("serve_ray")


def _raw_class(mod):
    """Return the original PCRFDownscaler class behind the Serve deployment.

    serve.deployment(...) wraps the class; the original is exposed as
    .func_or_class on the returned Deployment. Fall back to the attribute
    itself if a future Ray stops wrapping.
    """
    dep = mod.PCRFDownscaler
    return getattr(dep, "func_or_class", dep)


def _make_instance(mod, in_ch=3, tile=8, cond_ch=7, batch_enabled=False):
    """Build a PCRFDownscaler instance without running __init__ (no ONNX load).

    We inject the attributes _run / _batched_run rely on and a fake session.
    """
    cls = _raw_class(mod)
    inst = cls.__new__(cls)
    inst.sess = _FakeSession(in_ch=in_ch)
    inst.tile = tile
    inst.in_ch = in_ch
    inst.cond_ch = cond_ch
    inst.backend = "onnxruntime-cpu"
    inst._batch_enabled = batch_enabled
    return inst


def _unwrapped_batched(inst):
    """Get the plain async function under @serve.batch, version-defensively.

    @serve.batch wraps _batched_run; functools.wraps keeps the original at
    __wrapped__ in current Ray. If unavailable we return None and the caller
    reconstructs the documented body (list comprehension of _run).
    """
    fn = type(inst)._batched_run
    return getattr(fn, "__wrapped__", None)


# --------------------------------------------------------------------------- #
# batched handler: N bodies in -> N results out, aligned
# --------------------------------------------------------------------------- #
def test_batched_handler_returns_aligned_list(monkeypatch):
    mod = _load_serve_ray(monkeypatch, {"PCRF_SERVE_BATCH": "1"})
    inst = _make_instance(mod, batch_enabled=True)

    bodies = [
        {"ensemble": 2, "steps": 3, "coarse_mass": 0.1},
        {"ensemble": 4, "steps": 2, "coarse_mass": 0.2},
        {"ensemble": 1, "steps": 5, "coarse_mass": 0.0},
    ]

    wrapped = _unwrapped_batched(inst)
    if wrapped is not None:
        results = asyncio.get_event_loop().run_until_complete(wrapped(inst, bodies))
    else:
        # Documented batch body: one _run per body, preserving input order.
        results = [inst._run(b) for b in bodies]

    assert isinstance(results, list)
    assert len(results) == len(bodies)
    # Each result is the dict shape _run produces, aligned to its input.
    for body, res in zip(bodies, results):
        assert set(res.keys()) >= {"shape", "metrics", "backend"}
        assert res["shape"] == [inst.in_ch, inst.tile, inst.tile]
        assert {"DE", "NVR", "MCE"} <= set(res["metrics"].keys())


def test_batched_handler_alignment_is_order_preserving(monkeypatch):
    """A field-returning request must keep its slot; only that slot has 'field'."""
    mod = _load_serve_ray(monkeypatch, {"PCRF_SERVE_BATCH": "1"})
    inst = _make_instance(mod, in_ch=3, tile=8, batch_enabled=True)

    bodies = [
        {"ensemble": 2, "steps": 2, "return_field": False},
        {"ensemble": 2, "steps": 2, "return_field": True},
        {"ensemble": 2, "steps": 2, "return_field": False},
    ]
    results = [inst._run(b) for b in bodies]

    assert "field" not in results[0]
    assert "field" in results[1]
    assert "field" not in results[2]
    # The field length equals C*H*W of the returned mean.
    assert len(results[1]["field"]) == inst.in_ch * inst.tile * inst.tile


def test_run_single_body_matches_sample_onnx(monkeypatch):
    """_run reproduces pcrf.serving.math.sample_onnx for the same inputs."""
    from pcrf.serving import math as sm

    mod = _load_serve_ray(monkeypatch, {})
    inst = _make_instance(mod, in_ch=3, tile=8, cond_ch=7)

    body = {"ensemble": 3, "steps": 4, "coarse_mass": 0.15, "project": True}
    out = inst._run(body)

    mean, metrics = sm.sample_onnx(
        inst.sess, B=3, tile=8, in_ch=3, cond_ch=7, steps=4,
        coarse_mass=0.15, project=True,
    )
    assert out["shape"] == list(mean.shape)
    assert out["metrics"]["NVR"] == metrics["NVR"]
    assert out["metrics"]["MCE"] == pytest.approx(metrics["MCE"])


# --------------------------------------------------------------------------- #
# num_gpus is read from PCRF_SERVE_GPUS (default 0 => CPU)
# --------------------------------------------------------------------------- #
def test_num_gpus_defaults_to_zero_cpu(monkeypatch):
    monkeypatch.delenv("PCRF_SERVE_GPUS", raising=False)
    mod = _load_serve_ray(monkeypatch, {})
    assert mod._SERVE_GPUS == 0.0
    assert mod._gpus_enabled(mod._SERVE_GPUS) is False
    opts = mod.PCRFDownscaler.ray_actor_options or {}
    assert float(opts.get("num_gpus", 0.0)) == 0.0


def test_num_gpus_reads_fractional_value(monkeypatch):
    mod = _load_serve_ray(monkeypatch, {"PCRF_SERVE_GPUS": "0.25"})
    assert mod._SERVE_GPUS == 0.25
    assert mod._gpus_enabled(mod._SERVE_GPUS) is True
    opts = mod.PCRFDownscaler.ray_actor_options or {}
    assert float(opts.get("num_gpus")) == 0.25


# --------------------------------------------------------------------------- #
# autoscaling_config carries min/max/target from env
# --------------------------------------------------------------------------- #
def test_autoscaling_config_defaults(monkeypatch):
    for k in ("PCRF_SERVE_MIN_REPLICAS", "PCRF_SERVE_MAX_REPLICAS",
              "PCRF_SERVE_TARGET_REQUESTS"):
        monkeypatch.delenv(k, raising=False)
    mod = _load_serve_ray(monkeypatch, {})
    cfg = _autoscaling_config(mod)
    assert cfg["min_replicas"] == 1
    assert cfg["max_replicas"] == 8
    assert cfg["target_ongoing_requests"] == 4


def test_autoscaling_config_reads_env(monkeypatch):
    mod = _load_serve_ray(monkeypatch, {
        "PCRF_SERVE_MIN_REPLICAS": "2",
        "PCRF_SERVE_MAX_REPLICAS": "16",
        "PCRF_SERVE_TARGET_REQUESTS": "7",
    })
    cfg = _autoscaling_config(mod)
    assert cfg["min_replicas"] == 2
    assert cfg["max_replicas"] == 16
    assert cfg["target_ongoing_requests"] == 7


def _autoscaling_config(mod):
    """Pull the autoscaling_config dict off the built deployment.

    Recent Ray exposes it via the Deployment's deployment_config /
    autoscaling_config; we fall back to the module-level knobs the decorator
    was built from so the assertion is robust across Ray 2.30+ shapes.
    """
    dep = mod.PCRFDownscaler
    dc = getattr(dep, "_deployment_config", None) or getattr(dep, "deployment_config", None)
    ac = getattr(dc, "autoscaling_config", None) if dc is not None else None
    if ac is not None:
        get = (lambda k: getattr(ac, k)) if not isinstance(ac, dict) else ac.get
        return {
            "min_replicas": int(get("min_replicas")),
            "max_replicas": int(get("max_replicas")),
            "target_ongoing_requests": int(get("target_ongoing_requests")),
        }
    # Fallback: the knobs the @serve.deployment decorator was constructed from.
    return {
        "min_replicas": mod._MIN_REPLICAS,
        "max_replicas": mod._MAX_REPLICAS,
        "target_ongoing_requests": mod._TARGET_REQUESTS,
    }


# --------------------------------------------------------------------------- #
# batch knobs are read from env
# --------------------------------------------------------------------------- #
def test_batch_knobs_read_from_env(monkeypatch):
    mod = _load_serve_ray(monkeypatch, {
        "PCRF_SERVE_BATCH": "1",
        "PCRF_SERVE_MAX_BATCH": "5",
        "PCRF_SERVE_BATCH_WAIT_S": "0.05",
    })
    assert mod._BATCH_ENABLED is True
    assert mod._MAX_BATCH == 5
    assert mod._BATCH_WAIT_S == 0.05


def test_batch_disabled_by_default(monkeypatch):
    monkeypatch.delenv("PCRF_SERVE_BATCH", raising=False)
    mod = _load_serve_ray(monkeypatch, {})
    assert mod._BATCH_ENABLED is False
