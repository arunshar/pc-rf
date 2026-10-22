"""Unit tests for the evaluation metrics (metrics.py)."""
import torch

from pcrf import metrics as me
from pcrf import physics as p


def test_psnr_identical_is_high():
    x = torch.rand(1, 1, 16, 16)
    assert me.psnr(x, x, data_range=1.0) > 80


def test_ssim_identical_near_one():
    x = torch.rand(1, 3, 32, 32)
    assert me.ssim(x, x, data_range=1.0) > 0.99


def test_crps_ensemble_finite():
    ens = torch.randn(3, 2, 3, 16, 16)
    tgt = torch.randn(2, 3, 16, 16)
    c = me.crps_ensemble(ens, tgt)
    assert c == c  # not NaN


def test_physics_metrics_ranges():
    spec = p.ChannelSpec(precip=0, u=1, v=2)
    x = torch.randn(2, 3, 16, 16)
    cm = torch.rand(2)
    assert me.divergence_error(x, spec) >= 0
    assert 0.0 <= me.nonneg_violation_rate(x, spec) <= 1.0
    assert me.mass_conservation_error(x, cm, spec) >= 0


def test_nvr_zero_for_nonnegative_precip():
    spec = p.ChannelSpec(precip=0, u=1, v=2)
    x = torch.rand(1, 3, 16, 16)  # all >= 0
    assert me.nonneg_violation_rate(x, spec) == 0.0
