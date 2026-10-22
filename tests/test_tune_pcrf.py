"""Unit tests for the Ray Tune HPO helpers in apps/training/tune_pcrf.py.

These test the PURE helpers only (param-space builder, the PhysicsLoss
monkeypatch class factory, and best-config extraction). They do NOT launch a
real Tuner, do NOT init ray, and run on CPU with tiny synthetic objects.

conftest.py puts apps/training on sys.path, so `import tune_pcrf` resolves the
same way `from train_ray import ...` does for the real sweep. The param-space
test injects a stub sampler so ray.tune does not need to be importable here.
"""
import importlib

import tune_pcrf as t
from pcrf import physics as p


# ---------------------------------------------------------------------------
# build_param_space: keys + ranges (no ray import, via injected sampler)
# ---------------------------------------------------------------------------
class _RecordingSampler:
    """Stand-in for ray.tune.loguniform that records its (lo, hi) bounds."""
    def __init__(self, lo, hi):
        self.lo, self.hi = lo, hi

    def __repr__(self):  # pragma: no cover - debug aid only
        return f"_RecordingSampler({self.lo}, {self.hi})"


def test_build_param_space_keys_and_ranges():
    base = dict(data="synthetic", fine=64, base=16, batch=4, epochs=3)
    ps = t.build_param_space(base, lr_low=1e-4, lr_high=5e-4,
                             sampler=_RecordingSampler)

    # The swept keys live under train_loop_config and merge with the base config.
    assert set(ps.keys()) == {"train_loop_config"}
    tlc = ps["train_loop_config"]
    for k in ("l_div", "l_nn", "l_mass", "lr"):
        assert k in tlc, f"missing swept key {k}"
    # base config keys are preserved (merged in, not dropped)
    for k, v in base.items():
        assert tlc[k] == v

    # Ranges bracket the sweep center (1.0 / 0.1 / 0.05) and the lr bounds.
    assert (tlc["l_div"].lo, tlc["l_div"].hi) == (0.1, 10.0)
    assert (tlc["l_nn"].lo, tlc["l_nn"].hi) == (0.01, 1.0)
    assert (tlc["l_mass"].lo, tlc["l_mass"].hi) == (0.005, 0.5)
    assert (tlc["lr"].lo, tlc["lr"].hi) == (1e-4, 5e-4)


def test_build_param_space_honors_lr_bounds_override():
    base = dict(data="era5")
    ps = t.build_param_space(base, lr_low=2e-4, lr_high=8e-4,
                             sampler=_RecordingSampler)
    lr = ps["train_loop_config"]["lr"]
    assert (lr.lo, lr.hi) == (2e-4, 8e-4)


def test_sweep_center_and_library_defaults_are_distinct():
    # Guards the corrected docstring claim: library defaults are 1.0/1.0/1.0,
    # the sweep centers on the target 1.0/0.1/0.05. They must NOT be equal.
    assert t.PHYSICS_LIBRARY_DEFAULTS == {"l_div": 1.0, "l_nn": 1.0, "l_mass": 1.0}
    assert t.SWEEP_CENTER == {"l_div": 1.0, "l_nn": 0.1, "l_mass": 0.05}
    assert t.PHYSICS_LIBRARY_DEFAULTS != t.SWEEP_CENTER
    # And the source-of-truth constant matches the real library signature.
    import inspect
    sig = inspect.signature(p.PhysicsLoss.__init__)
    for k in ("l_div", "l_nn", "l_mass"):
        assert sig.parameters[k].default == t.PHYSICS_LIBRARY_DEFAULTS[k]


# ---------------------------------------------------------------------------
# make_tuned_physics_loss_cls: the monkeypatch injection
# ---------------------------------------------------------------------------
def test_tuned_physics_loss_injects_weights():
    spec = p.ChannelSpec(precip=0, u=1, v=2)
    Cls = t.make_tuned_physics_loss_cls(p, l_div=2.5, l_nn=0.2, l_mass=0.07)
    # Construct exactly the way train_ray.py does: positional spec, no weights.
    loss = Cls(spec)
    assert isinstance(loss, p.PhysicsLoss)        # still a PhysicsLoss subclass
    assert loss.l_div == 2.5
    assert loss.l_nn == 0.2
    assert loss.l_mass == 0.07


def test_tuned_physics_loss_default_differs_from_library_default():
    spec = p.ChannelSpec(precip=0)
    # Library default would give 1.0/1.0/1.0; the tuned class must override.
    Cls = t.make_tuned_physics_loss_cls(p, l_div=0.5, l_nn=0.05, l_mass=0.01)
    tuned = Cls(spec)
    plain = p.PhysicsLoss(spec)
    assert (plain.l_div, plain.l_nn, plain.l_mass) == (1.0, 1.0, 1.0)
    assert (tuned.l_div, tuned.l_nn, tuned.l_mass) == (0.5, 0.05, 0.01)


def test_tuned_physics_loss_honors_explicit_override():
    spec = p.ChannelSpec(precip=0, u=1, v=2)
    Cls = t.make_tuned_physics_loss_cls(p, l_div=2.0, l_nn=0.2, l_mass=0.02)
    # A future caller that DOES pass a weight must win over the injected default.
    loss = Cls(spec, l_div=9.0)
    assert loss.l_div == 9.0          # explicit override respected
    assert loss.l_nn == 0.2           # others fall back to injected values
    assert loss.l_mass == 0.02


def test_tuned_physics_loss_forward_uses_injected_weights():
    # End-to-end check: a larger l_nn must raise L_total on a clearly-negative
    # precip field (so the injected weight actually flows into the objective).
    import torch
    spec = p.ChannelSpec(precip=0, u=1, v=2)
    x = torch.full((2, 3, 8, 8), -1.0)   # all-negative precip -> nonzero L_nn
    cmass = torch.zeros(2)
    low = t.make_tuned_physics_loss_cls(p, 1.0, 0.1, 0.05)(spec)
    high = t.make_tuned_physics_loss_cls(p, 1.0, 5.0, 0.05)(spec)
    total_low, _ = low(x, cmass)
    total_high, _ = high(x, cmass)
    assert float(total_high) > float(total_low)


# ---------------------------------------------------------------------------
# extract_best_config: read the winner from a fake ResultGrid-like object
# ---------------------------------------------------------------------------
class _FakeResult:
    def __init__(self, config, metrics):
        self.config = config
        self.metrics = metrics


class _FakeResultGrid:
    """Mimics the surface of ray.tune.ResultGrid used by extract_best_config."""
    def __init__(self, best):
        self._best = best
        self.calls = []

    def get_best_result(self, metric, mode):
        self.calls.append((metric, mode))
        return self._best


def test_extract_best_config_returns_weights_and_metric():
    best = _FakeResult(
        config={"train_loop_config": {
            "l_div": 1.3, "l_nn": 0.08, "l_mass": 0.04, "lr": 3e-4,
            "data": "era5", "epochs": 30}},
        metrics={"L_total": 0.123, "L_div": 0.01})
    grid = _FakeResultGrid(best)

    out = t.extract_best_config(grid, metric="L_total", mode="min")
    assert grid.calls == [("L_total", "min")]
    assert out["l_div"] == 1.3
    assert out["l_nn"] == 0.08
    assert out["l_mass"] == 0.04
    assert out["lr"] == 3e-4
    assert out["metric_name"] == "L_total"
    assert out["metric_value"] == 0.123
    # non-swept config keys must NOT leak into the extracted weight dict
    assert "data" not in out and "epochs" not in out


def test_extract_best_config_flat_config_fallback():
    # If a result stores the loop config flat (no train_loop_config wrapper),
    # extract_best_config still finds the weights.
    best = _FakeResult(
        config={"l_div": 2.0, "l_nn": 0.5, "l_mass": 0.1, "lr": 1e-4},
        metrics={"val_crps": 0.9})
    out = t.extract_best_config(_FakeResultGrid(best), metric="val_crps", mode="min")
    assert out == {"l_div": 2.0, "l_nn": 0.5, "l_mass": 0.1, "lr": 1e-4,
                   "metric_name": "val_crps", "metric_value": 0.9}


def test_module_importable_without_ray():
    # The pure helpers must be importable without ray on the path (ray imports
    # are deferred into main() / the worker loop). Re-importing must not raise.
    importlib.reload(t)
    assert hasattr(t, "build_param_space")
    assert hasattr(t, "make_tuned_physics_loss_cls")
    assert hasattr(t, "extract_best_config")
