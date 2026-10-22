"""Unit tests for rectified-flow training + sampling (flow.py)."""
import torch

from pcrf import flow as fl
from pcrf import model as m
from pcrf import physics as p


def _make(attn=()):
    spec = p.ChannelSpec(precip=0, u=1, v=2)
    net = m.UNet(in_ch=3, cond_ch=7, base=16, ch_mults=(1, 2), attn_at=attn)
    return spec, fl.RectifiedFlow(net, p.PhysicsLoss(spec))


def test_training_loss_finite_and_backward():
    spec, rf = _make()
    x1 = torch.randn(2, 3, 32, 32)
    cond = torch.randn(2, 7, 32, 32)
    cm = torch.rand(2)
    loss, logs = rf.training_loss(x1, cond, cm)
    assert torch.isfinite(loss)
    loss.backward()
    g = sum(par.grad.abs().sum() for par in rf.parameters() if par.grad is not None)
    assert g > 0
    assert "L_RF" in logs and "L_total" in logs


def test_training_loss_without_physics():
    net = m.UNet(in_ch=3, cond_ch=7, base=16, ch_mults=(1, 2), attn_at=())
    rf = fl.RectifiedFlow(net, None)
    loss, logs = rf.training_loss(torch.randn(1, 3, 32, 32),
                                  torch.randn(1, 7, 32, 32), torch.rand(1))
    assert torch.isfinite(loss)
    assert "L_div" not in logs


def test_forward_returns_training_loss():
    # rf(...) must give the same (loss, logs) as training_loss under the same
    # noise, because train_ray.py calls the DDP wrapper, which calls forward.
    _, rf = _make()
    x1, cond, cm = torch.randn(2, 3, 32, 32), torch.randn(2, 7, 32, 32), torch.rand(2)
    torch.manual_seed(3)
    loss_a, logs_a = rf(x1, cond, cm)
    torch.manual_seed(3)
    loss_b, logs_b = rf.training_loss(x1, cond, cm)
    assert torch.equal(loss_a, loss_b)
    assert logs_a == logs_b


def test_sample_shape_and_nonnegativity():
    spec, rf = _make()
    cond = torch.randn(2, 7, 32, 32)
    cm = torch.full((2,), 0.3)
    with torch.no_grad():
        x = rf.sample((2, 3, 32, 32), cond, steps=4, device="cpu",
                      spec=spec, coarse_mass=cm, project=True)
    assert x.shape == (2, 3, 32, 32)
    assert (x[:, 0] >= -1e-6).all()
