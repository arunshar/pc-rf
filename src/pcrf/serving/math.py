"""
Numpy version of the serving-side physics: the projection and the three metrics
used by the Python serving path (serve_mock, tests, visuals). A Rust service that
is not part of this repository implemented the same steps. The Euler sampler step
itself is trivial (x += v * dt) and lives at the call sites.

This projection differs from pcrf.physics.project_physics in two ways. With
coarse_mass <= 0 it applies only the clamp, and the training-side function
rescales to zero. Its MCE returns 0.0 whenever the coarse mass is below 1e-8,
which hides any error in that case.

Channel order matches the model: [precip, u, v].
"""
from __future__ import annotations
import numpy as np


def _ddx(f: np.ndarray) -> np.ndarray:
    """d/dx via central difference, replicate edges."""
    fp = np.pad(f, ((0, 0), (1, 1)), mode="edge")
    return (fp[:, 2:] - fp[:, :-2]) * 0.5


def _ddy(f: np.ndarray) -> np.ndarray:
    fp = np.pad(f, ((1, 1), (0, 0)), mode="edge")
    return (fp[2:, :] - fp[:-2, :]) * 0.5


def project_precip(x: np.ndarray, precip: int = 0, coarse_mass: float = 0.0) -> np.ndarray:
    """Clamp precip >= 0, then multiplicative mass rescale to the coarse total.

    x: [B, C, H, W]. Multiplicative rescale preserves non-negativity. Returns a
    copy. When coarse_mass <= 0 only the non-negativity clamp is applied.
    """
    x = x.copy()
    p = np.clip(x[:, precip:precip + 1], 0.0, None)
    if coarse_mass > 0.0:
        cur = np.clip(p.mean(axis=(1, 2, 3), keepdims=True), 1e-8, None)
        p = p * (coarse_mass / cur)
    x[:, precip:precip + 1] = p
    return x


def divergence_error(mean: np.ndarray, u: int = 1, v: int = 2) -> float:
    """DE: mean |du/dx + dv/dy| on the ensemble-mean field [C, H, W]."""
    return float(np.abs(_ddx(mean[u]) + _ddy(mean[v])).mean())


def nonneg_violation_rate(mean: np.ndarray, precip: int = 0) -> float:
    """NVR: fraction of precip cells < 0."""
    p = mean[precip]
    return float((p < 0).mean())


def mass_conservation_error(mean: np.ndarray, coarse_mass: float, precip: int = 0) -> float:
    """MCE: |fine_mass - coarse_mass| / |coarse_mass|."""
    if abs(coarse_mass) < 1e-8:
        return 0.0
    fine_mass = float(mean[precip].mean())
    return abs(fine_mass - coarse_mass) / abs(coarse_mass)


def sample_onnx(sess, B: int, tile: int, in_ch: int = 3, cond_ch: int = 7,
                steps: int = 20, coarse_mass: float = 0.1, seed: int = 0,
                cond: np.ndarray | None = None, project: bool = True):
    """Drive an ONNX velocity session through the Euler sampler + projection.

    Returns (ensemble_mean[C,H,W], metrics dict). Uses project_precip and the metrics above.
    """
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((B, in_ch, tile, tile), dtype=np.float32)
    if cond is None:
        cond = (rng.standard_normal((B, cond_ch, tile, tile), dtype=np.float32)) * 0.5
    dt = 1.0 / steps
    for i in range(steps):
        t = np.full((B,), i * dt, dtype=np.float32)
        x = x + sess.run(["velocity"], {"x_t": x, "t": t, "cond": cond})[0] * dt
    if project:
        x = project_precip(x, 0, coarse_mass)
    mean = x.mean(0)
    metrics = {
        "DE": divergence_error(mean),
        "NVR": nonneg_violation_rate(mean),
        "MCE": mass_conservation_error(mean, coarse_mass),
    }
    return mean, metrics
