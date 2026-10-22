"""Pure-NumPy synthetic downscaling item generator for the Ray Data ingest path.

WHY THIS EXISTS (the torch._C worker bug, correctly diagnosed):
  The --ingest ray path runs a flat_map UDF in Ray Data worker processes. The
  earlier UDF did ``from pcrf.data import SyntheticDownscaleDataset``, and
  pcrf/data.py imports torch at module top. Importing torch inside a Ray Data
  worker (a process the raylet forks/spawns) intermittently fails with

      NameError: name '_C' is not defined        # torch/__init__.py: for __name in dir(_C)

  i.e. torch's compiled _C extension is not initialized in that worker process.
  The root cause was NOT a torch tensor crossing the Arrow boundary (the earlier
  theory baked into the old docstrings); it was ``import torch`` ITSELF failing
  in the worker. The robust, deterministic fix is to keep the whole UDF
  torch-free so no Ray Data worker ever imports torch. This module therefore has
  NO torch import on purpose -- numpy only.

The arrays mirror SyntheticDownscaleDataset (pcrf.data) in STRUCTURE -- a
divergence-free wind from a smoothed stream function, non-negative precip blobs,
a block-mean coarse conditioning field, and smooth "satellite" bands -- but the
RNG and the coarse upsampling differ from the torch dataset (numpy Generator vs
torch.Generator; nearest-block upsample vs bicubic). That is intentional and
harmless: ``--ingest ray`` is a data-ingest / throughput path exercised on
SYNTHETIC data, not a fidelity comparison against the torch dataloader, so
bit-identical fields are neither needed nor claimed. Channel order of "fine" is
[precip, u, v] (physics.ChannelSpec(0, 1, 2)), matching the torch dataset and
RectifiedFlow.training_loss.
"""
from __future__ import annotations
import numpy as np


def _smooth_noise_np(h, w, scale, rng):
    """NumPy port of pcrf.data._smooth_noise.

    White noise low-passed by a Gaussian filter in the Fourier domain, then
    standardized to zero mean / unit std. No torch.
    """
    n = rng.standard_normal((h, w))
    fx = np.fft.fftfreq(w)[None, :]
    fy = np.fft.fftfreq(h)[:, None]
    flt = np.exp(-(fx ** 2 + fy ** 2) * (scale ** 2) * 20.0)
    out = np.real(np.fft.ifft2(np.fft.fft2(n) * flt))
    return (out - out.mean()) / (out.std() + 1e-8)


def synth_item_numpy(idx, fine, factor=4, sat_bands=4, seed=0):
    """Build one synthetic downscaling item as a pure-NumPy row.

    Returns a dict keyed fine/cond/coarse_mass:
      fine        float32 [3, fine, fine]              (precip, u, v)
      cond        float32 [3 + sat_bands, fine, fine]  (upsampled coarse + sat)
      coarse_mass float32 [1]                           (domain-mean coarse precip)

    No torch anywhere. coarse_mass is a [1]-shaped array so it collates to [B, 1]
    and is reduced to [B] by PhysicsLoss.mass_penalty via ``.view(-1)`` -- the
    same contract _item_to_numpy_row emits for the torch path. ``fine`` must be a
    multiple of ``factor`` (always true for the synthetic configs: 16/32/64/128
    with factor 4).
    """
    H = W = int(fine)
    cf = int(factor)
    sat_bands = int(sat_bands)
    # Per-item deterministic RNG (mirrors the torch dataset's per-idx seeding so
    # a given idx is reproducible across epochs/workers).
    rng = np.random.default_rng(int(seed) * 100003 + int(idx))

    # divergence-free wind from a smoothed stream function (central differences)
    psi = _smooth_noise_np(H, W, 7.0, rng)
    u = 0.5 * (np.roll(psi, -1, axis=0) - np.roll(psi, 1, axis=0))     # d/dy
    v = -0.5 * (np.roll(psi, -1, axis=1) - np.roll(psi, 1, axis=1))    # d/dx

    # non-negative precip from a few Gaussian blobs
    yy, xx = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    p = np.zeros((H, W), dtype=np.float64)
    for _ in range(6):
        cy = int(rng.integers(0, H))
        cx = int(rng.integers(0, W))
        r = float(rng.random()) * 10.0 + 5.0
        p += np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2.0 * r ** 2))

    fine_arr = np.stack([p, u, v], axis=0)                            # [3,H,W]

    # coarse conditioning = block-mean (avg-pool cf) then nearest upsample back.
    # nearest (np.repeat) is a deliberate torch-free stand-in for the torch
    # dataset's bicubic interpolate; shapes are identical and the coarse->fine
    # structure (hence a learnable mapping) is preserved.
    C = fine_arr.shape[0]
    coarse = fine_arr.reshape(C, H // cf, cf, W // cf, cf).mean(axis=(2, 4))  # [3,Hc,Wc]
    cond_up = np.repeat(np.repeat(coarse, cf, axis=1), cf, axis=2)            # [3,H,W]
    coarse_mass = float(coarse[0].mean())                                    # precip mass

    # smooth "satellite" bands (stand-in for Sentinel-2 at fine res)
    if sat_bands > 0:
        sat = np.stack([_smooth_noise_np(H, W, 3.0, rng) for _ in range(sat_bands)],
                       axis=0)
    else:
        sat = np.zeros((0, H, W), dtype=np.float64)
    cond_arr = np.concatenate([cond_up, sat], axis=0)                        # [3+sat,H,W]

    return {
        "fine": np.ascontiguousarray(fine_arr, dtype=np.float32),
        "cond": np.ascontiguousarray(cond_arr, dtype=np.float32),
        "coarse_mass": np.asarray([coarse_mass], dtype=np.float32),
    }
