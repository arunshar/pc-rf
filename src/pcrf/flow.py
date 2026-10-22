"""
Rectified-flow training and sampling for PC-RF (Section 3.1-3.2 of the paper).

Training objective (Eq. 6):
    L = L_RF + lambda1 L_div + lambda2 L_nn + lambda3 L_mass

The physics penalties are evaluated on the one-step clean-field estimate
    x_hat = x_t + (1 - t) * v_theta(x_t, t, c)
which is the rectified-flow analogue of the predicted data point.
"""
from __future__ import annotations
import torch
import torch.nn as nn

from pcrf.physics import PhysicsLoss, project_physics, ChannelSpec


class RectifiedFlow(nn.Module):
    """Wraps a velocity network with RF training loss and ODE sampling."""

    def __init__(self, model: nn.Module, physics: PhysicsLoss | None = None):
        super().__init__()
        self.model = model
        self.physics = physics

    # ------------------------------------------------------------------ train
    def forward(self, x1: torch.Tensor, cond: torch.Tensor,
                coarse_mass: torch.Tensor | None = None):
        """Return (loss, logs) from training_loss.

        DistributedDataParallel arms its gradient all-reduce inside its own
        forward, so a DDP-wrapped RectifiedFlow must be called as rf(...). A
        call to rf.module.training_loss(...) skips that step, and each rank
        then trains an independent replica.
        """
        return self.training_loss(x1, cond, coarse_mass)

    def training_loss(self, x1: torch.Tensor, cond: torch.Tensor,
                      coarse_mass: torch.Tensor | None = None):
        """x1: clean fine field [B,C,H,W]; cond: conditioning image [B,Cc,H,W]."""
        b = x1.shape[0]
        t = torch.rand(b, device=x1.device)
        x0 = torch.randn_like(x1)
        t_b = t.view(b, 1, 1, 1)
        x_t = (1 - t_b) * x0 + t_b * x1
        target = x1 - x0

        v = self.model(x_t, t, cond)
        loss_rf = ((v - target) ** 2).mean()

        logs = {"L_RF": float(loss_rf.detach())}
        total = loss_rf

        if self.physics is not None and coarse_mass is not None:
            x_hat = x_t + (1 - t_b) * v          # one-step clean estimate
            loss_phys, plog = self.physics(x_hat, coarse_mass)
            total = total + loss_phys
            logs.update(plog)

        logs["L_total"] = float(total.detach())
        return total, logs

    # ----------------------------------------------------------------- sample
    @torch.no_grad()
    def sample(self, shape, cond: torch.Tensor, steps: int = 20,
               device: str = "cpu", spec: ChannelSpec | None = None,
               coarse_mass: torch.Tensor | None = None,
               project: bool = True) -> torch.Tensor:
        """Integrate dx/dt = v_theta from x0 ~ N(0,I) at t=0 to t=1 (Euler)."""
        x = torch.randn(shape, device=device)
        dt = 1.0 / steps
        for i in range(steps):
            t = torch.full((shape[0],), i * dt, device=device)
            x = x + self.model(x, t, cond) * dt
        if project and spec is not None:
            x = project_physics(x, spec, coarse_mass)
        return x
