"""
Physics-penalty operators for PC-RF (PyTorch, autograd-friendly).

Mirrors Section 3.3 of the paper:
  L_div  : divergence-free wind constraint
  L_nn   : non-negativity of precipitation
  L_mass : domain mass conservation

All penalties are computed on the one-step clean-field estimate x_hat_f
(see flow.py: x_hat = x_t + (1 - t) * v_theta). Channel layout is configurable
via a ChannelSpec so the same module works for precip-only or wind+precip data.
"""
from __future__ import annotations
from dataclasses import dataclass
import torch
import torch.nn as nn


# ----------------------------------------------------------------------------
# Finite-difference operators (central differences, replicate padding)
# ----------------------------------------------------------------------------
def ddx(f: torch.Tensor) -> torch.Tensor:
    """d/dx via central difference. f: [B, C, H, W]."""
    fp = torch.nn.functional.pad(f, (1, 1, 0, 0), mode="replicate")
    return (fp[..., :, 2:] - fp[..., :, :-2]) * 0.5


def ddy(f: torch.Tensor) -> torch.Tensor:
    """d/dy via central difference. f: [B, C, H, W]."""
    fp = torch.nn.functional.pad(f, (0, 0, 1, 1), mode="replicate")
    return (fp[..., 2:, :] - fp[..., :-2, :]) * 0.5


def divergence(u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Horizontal divergence du/dx + dv/dy. u, v: [B, 1, H, W]."""
    return ddx(u) + ddy(v)


@dataclass
class ChannelSpec:
    """Which channels of the field are which physical variable.

    Example (precip + u + v + temp): precip=0, u=1, v=2.
    Set u/v to None to disable the divergence penalty (precip-only data).
    """
    precip: int | None = 0
    u: int | None = None
    v: int | None = None


class PhysicsLoss(nn.Module):
    """Weighted sum of the three physics penalties.

    Args:
        spec: channel layout.
        l_div, l_nn, l_mass: penalty weights (lambda_1..3 in the paper).
    """
    def __init__(self, spec: ChannelSpec,
                 l_div: float = 1.0, l_nn: float = 1.0, l_mass: float = 1.0):
        super().__init__()
        self.spec = spec
        self.l_div, self.l_nn, self.l_mass = l_div, l_nn, l_mass

    def divergence_penalty(self, x: torch.Tensor) -> torch.Tensor:
        if self.spec.u is None or self.spec.v is None:
            return x.new_zeros(())
        u = x[:, self.spec.u:self.spec.u + 1]
        v = x[:, self.spec.v:self.spec.v + 1]
        return (divergence(u, v) ** 2).mean()

    def nonneg_penalty(self, x: torch.Tensor) -> torch.Tensor:
        if self.spec.precip is None:
            return x.new_zeros(())
        p = x[:, self.spec.precip:self.spec.precip + 1]
        return (torch.clamp(p, max=0.0) ** 2).mean()

    def mass_penalty(self, x: torch.Tensor,
                     coarse_mass: torch.Tensor) -> torch.Tensor:
        """coarse_mass: [B] or [B,1], the domain-mean precip of the coarse input."""
        if self.spec.precip is None:
            return x.new_zeros(())
        p = x[:, self.spec.precip:self.spec.precip + 1]
        fine_mass = p.mean(dim=(1, 2, 3))            # [B]
        return ((fine_mass - coarse_mass.view(-1)) ** 2).mean()

    def forward(self, x_hat: torch.Tensor,
                coarse_mass: torch.Tensor) -> tuple[torch.Tensor, dict]:
        ld = self.divergence_penalty(x_hat)
        ln = self.nonneg_penalty(x_hat)
        lm = self.mass_penalty(x_hat, coarse_mass)
        total = self.l_div * ld + self.l_nn * ln + self.l_mass * lm
        logs = {"L_div": float(ld.detach()),
                "L_nn": float(ln.detach()),
                "L_mass": float(lm.detach())}
        return total, logs


# ----------------------------------------------------------------------------
# Inference-time projection (clamp at zero, then a multiplicative mass rescale)
# ----------------------------------------------------------------------------
@torch.no_grad()
def project_physics(x: torch.Tensor, spec: ChannelSpec,
                    coarse_mass: torch.Tensor | None = None) -> torch.Tensor:
    """Clamp precipitation at zero, then rescale it to the coarse mass.

    Applied at inference. It sets NVR to 0, and MCE to about 0, when coarse_mass is
    not negative and the clamped mean is at least 1e-8. If every cell is clamped
    the field stays all zero. See KNOWN_ISSUES.md for the cases where it fails.
    """
    x = x.clone()
    if spec.precip is not None:
        p = x[:, spec.precip:spec.precip + 1].clamp(min=0.0)
        if coarse_mass is not None:
            cur = p.mean(dim=(1, 2, 3), keepdim=True).clamp(min=1e-8)
            tgt = coarse_mass.view(-1, 1, 1, 1)
            p = p * (tgt / cur)
        x[:, spec.precip:spec.precip + 1] = p
    return x
