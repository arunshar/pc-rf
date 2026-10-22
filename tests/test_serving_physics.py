"""Unit tests for the serving-side physics (pcrf.serving.math).

These are pure-numpy and fast: they pin the exact projection + metric behavior
that a Rust sampler (not included in this repository) mirrors.
"""
import numpy as np
from pcrf.serving import math as sm


def test_projection_enforces_nonnegativity():
    rng = np.random.default_rng(0)
    x = rng.standard_normal((4, 3, 16, 16)).astype(np.float32)
    y = sm.project_precip(x, precip=0, coarse_mass=0.0)
    assert (y[:, 0] >= 0).all()
    assert sm.nonneg_violation_rate(y.mean(0)) == 0.0


def test_projection_conserves_mass_per_sample():
    rng = np.random.default_rng(1)
    x = rng.standard_normal((4, 3, 16, 16)).astype(np.float32)
    cm = 0.25
    y = sm.project_precip(x, precip=0, coarse_mass=cm)
    per_sample_mean = y[:, 0].mean(axis=(1, 2))
    assert np.allclose(per_sample_mean, cm, atol=1e-5)
    assert sm.mass_conservation_error(y.mean(0), cm) < 1e-4


def test_projection_handles_all_negative_precip():
    x = -np.ones((2, 3, 8, 8), dtype=np.float32)
    # clamp makes precip all-zero; rescale guards against divide-by-zero
    y = sm.project_precip(x, precip=0, coarse_mass=0.3)
    assert np.isfinite(y).all()
    assert (y[:, 0] >= 0).all()


def test_divergence_free_field_has_near_zero_DE():
    # Analytic stream function psi = sin(kx x) cos(ky y); u = dpsi/dy, v = -dpsi/dx
    # gives an analytically divergence-free field.
    H = W = 48
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float64)
    kx, ky = 2 * np.pi / W, 2 * np.pi / H
    u = -ky * np.sin(kx * xx) * np.sin(ky * yy)
    v = -kx * np.cos(kx * xx) * np.cos(ky * yy)
    mean = np.stack([np.zeros_like(u), u, v]).astype(np.float32)
    div_free = sm.divergence_error(mean)

    # A radial outflow field is strongly divergent (du/dx + dv/dy = 2).
    u2 = (xx - W / 2)
    v2 = (yy - H / 2)
    mean2 = np.stack([np.zeros_like(u2), u2, v2]).astype(np.float32)
    divergent = sm.divergence_error(mean2)

    assert div_free < 0.05
    assert divergent > 0.5
    assert div_free < 0.05 * divergent


def test_mce_zero_when_no_coarse_mass():
    mean = np.random.default_rng(3).standard_normal((3, 8, 8)).astype(np.float32)
    assert sm.mass_conservation_error(mean, 0.0) == 0.0
