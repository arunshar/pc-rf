"""Unit tests for the physics operators and penalties (physics.py)."""
import torch

from pcrf import physics as p


def test_ddx_ddy_constant_is_zero():
    f = torch.ones(1, 1, 8, 8)
    assert p.ddx(f).abs().max() < 1e-6
    assert p.ddy(f).abs().max() < 1e-6


def test_ddx_linear_is_unit_slope():
    # f(x) = column index; central difference in the interior is ~1
    x = torch.arange(8.0).view(1, 1, 1, 8).expand(1, 1, 8, 8).contiguous()
    d = p.ddx(x)
    assert torch.allclose(d[..., 1:-1], torch.ones_like(d[..., 1:-1]), atol=1e-5)


def test_divergence_free_field_has_lower_divergence():
    torch.manual_seed(0)
    import torch.nn.functional as F
    psi = torch.randn(1, 1, 32, 32)
    psi = F.conv2d(psi, torch.ones(1, 1, 5, 5) / 25.0, padding=2)  # smooth
    u, v = p.ddy(psi), -p.ddx(psi)            # divergence-free by construction
    div_free = p.divergence(u, v).abs().mean()
    div_any = p.divergence(psi, psi).abs().mean()
    assert div_free < div_any


def test_physicsloss_returns_finite_and_logs():
    spec = p.ChannelSpec(precip=0, u=1, v=2)
    loss = p.PhysicsLoss(spec)
    x = torch.randn(2, 3, 16, 16)
    cmass = torch.rand(2)
    total, logs = loss(x, cmass)
    assert torch.isfinite(total)
    assert {"L_div", "L_nn", "L_mass"}.issubset(logs)


def test_nonneg_penalty_zero_when_positive():
    spec = p.ChannelSpec(precip=0, u=1, v=2)
    loss = p.PhysicsLoss(spec)
    assert loss.nonneg_penalty(torch.rand(2, 3, 16, 16)) == 0.0


def test_project_physics_enforces_nonneg_and_mass():
    spec = p.ChannelSpec(precip=0, u=1, v=2)
    x = torch.randn(2, 3, 16, 16)
    cmass = torch.full((2,), 0.3)
    y = p.project_physics(x, spec, cmass)
    precip = y[:, 0]
    assert (precip >= 0).all()
    assert torch.allclose(precip.mean(dim=(1, 2)), cmass, atol=1e-5)
