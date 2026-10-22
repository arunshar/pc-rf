"""
Evaluation metrics for PC-RF.

Perceptual:  PSNR, SSIM, CRPS (ensemble).
Physics:     Divergence Error (DE), Non-negativity Violation Rate (NVR),
             Mass Conservation Error (MCE).

All functions accept torch tensors [B, C, H, W] and return python floats
(means over the batch). Physics metrics take a ChannelSpec.
"""
from __future__ import annotations
import torch
import torch.nn.functional as F

from pcrf.physics import ddx, ddy, ChannelSpec


# ------------------------------------------------------------- perceptual ----
def psnr(pred: torch.Tensor, target: torch.Tensor,
         data_range: float | None = None) -> float:
    if data_range is None:
        data_range = float(target.max() - target.min())
    mse = F.mse_loss(pred, target)
    return float(10.0 * torch.log10((data_range ** 2) / (mse + 1e-12)))


def _gaussian_window(ch: int, ksize: int = 11, sigma: float = 1.5,
                     device="cpu"):
    coords = torch.arange(ksize, device=device) - ksize // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = (g / g.sum())
    k2d = torch.outer(g, g)
    return k2d.expand(ch, 1, ksize, ksize).contiguous()


def ssim(pred: torch.Tensor, target: torch.Tensor,
         data_range: float | None = None) -> float:
    """Windowed SSIM (Gaussian window), averaged over channels and batch."""
    if data_range is None:
        data_range = float(target.max() - target.min())
    ch = pred.shape[1]
    w = _gaussian_window(ch, device=pred.device)
    pad = w.shape[-1] // 2
    mu_p = F.conv2d(pred, w, padding=pad, groups=ch)
    mu_t = F.conv2d(target, w, padding=pad, groups=ch)
    mu_p2, mu_t2, mu_pt = mu_p ** 2, mu_t ** 2, mu_p * mu_t
    sig_p = F.conv2d(pred * pred, w, padding=pad, groups=ch) - mu_p2
    sig_t = F.conv2d(target * target, w, padding=pad, groups=ch) - mu_t2
    sig_pt = F.conv2d(pred * target, w, padding=pad, groups=ch) - mu_pt
    c1, c2 = (0.01 * data_range) ** 2, (0.03 * data_range) ** 2
    s = ((2 * mu_pt + c1) * (2 * sig_pt + c2)) / \
        ((mu_p2 + mu_t2 + c1) * (sig_p + sig_t + c2) + 1e-12)
    return float(s.mean())


def crps_ensemble(ensemble: torch.Tensor, target: torch.Tensor) -> float:
    """Fair CRPS for an ensemble. ensemble: [M, B, C, H, W], target: [B,C,H,W].

    CRPS = E|X - y| - 0.5 E|X - X'|  (estimated over the M members).
    """
    m = ensemble.shape[0]
    term1 = (ensemble - target[None]).abs().mean()
    diff = (ensemble[:, None] - ensemble[None, :]).abs()   # [M,M,...]
    term2 = diff.sum(dim=(0, 1)) / (m * (m - 1) + 1e-12)
    return float(term1 - 0.5 * term2.mean())


# ---------------------------------------------------------------- physics ----
def divergence_error(x: torch.Tensor, spec: ChannelSpec) -> float:
    if spec.u is None or spec.v is None:
        return float("nan")
    u = x[:, spec.u:spec.u + 1]
    v = x[:, spec.v:spec.v + 1]
    return float((ddx(u) + ddy(v)).abs().mean())


def nonneg_violation_rate(x: torch.Tensor, spec: ChannelSpec) -> float:
    if spec.precip is None:
        return float("nan")
    p = x[:, spec.precip:spec.precip + 1]
    return float((p < 0).float().mean())


def mass_conservation_error(x: torch.Tensor, coarse_mass: torch.Tensor,
                            spec: ChannelSpec) -> float:
    if spec.precip is None:
        return float("nan")
    p = x[:, spec.precip:spec.precip + 1]
    fine = p.mean(dim=(1, 2, 3))
    return float(((fine - coarse_mass.view(-1)).abs()
                  / (coarse_mass.view(-1).abs() + 1e-8)).mean())


def sam(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> float:
    """Spectral Angle Mapper in degrees (lower is better), averaged over pixels
    and batch. pred/target: [B, C, H, W]; the angle is taken between the two
    C-dim spectral vectors at each pixel. The natural fidelity metric for optical
    multi-band super-resolution (invariant to per-pixel gain, sensitive to
    spectral distortion)."""
    dot = (pred * target).sum(dim=1)                       # [B, H, W]
    pn = pred.norm(dim=1)
    tn = target.norm(dim=1)
    cos = (dot / (pn * tn + eps)).clamp(-1.0, 1.0)
    return float(torch.arccos(cos).mean() * 180.0 / torch.pi)


def all_metrics(pred, target, coarse_mass, spec, data_range=None) -> dict:
    return {
        "PSNR": psnr(pred, target, data_range),
        "SSIM": ssim(pred, target, data_range),
        "SAM":  sam(pred, target),
        "DE":   divergence_error(pred, spec),
        "NVR":  nonneg_violation_rate(pred, spec),
        "MCE":  mass_conservation_error(pred, coarse_mass, spec),
    }
