"""End-to-end pipeline test: data -> model -> train -> sample -> projection ->
metrics. Asserts the physics projection drives NVR to 0 and MCE to ~0, exactly
the paper's inference-time guarantee.
"""
import torch
from torch.utils.data import DataLoader

from pcrf import data as d
from pcrf import flow as fl
from pcrf import metrics as me
from pcrf import model as m
from pcrf import physics as p


def test_end_to_end_pipeline_and_physics_projection():
    torch.manual_seed(0)
    spec = p.ChannelSpec(precip=0, u=1, v=2)
    ds = d.SyntheticDownscaleDataset(n=8, fine=32, factor=4, sat_bands=4)
    in_ch = ds[0]["fine"].shape[0]
    cond_ch = ds[0]["cond"].shape[0]

    net = m.UNet(in_ch=in_ch, cond_ch=cond_ch, base=16, ch_mults=(1, 2), attn_at=())
    rf = fl.RectifiedFlow(net, p.PhysicsLoss(spec))
    opt = torch.optim.Adam(rf.parameters(), lr=2e-3)
    dl = DataLoader(ds, batch_size=4, shuffle=True)

    losses = []
    for _ in range(3):
        for b in dl:
            opt.zero_grad()
            loss, _ = rf.training_loss(b["fine"], b["cond"], b["coarse_mass"])
            loss.backward()
            opt.step()
            losses.append(float(loss.detach()))
    assert all(x == x for x in losses)  # no NaNs through training

    fine = torch.stack([ds[i]["fine"] for i in range(4)])
    cond = torch.stack([ds[i]["cond"] for i in range(4)])
    cm = torch.stack([ds[i]["coarse_mass"] for i in range(4)])
    with torch.no_grad():
        x = rf.sample(fine.shape, cond, steps=8, device="cpu",
                      spec=spec, coarse_mass=cm, project=True)

    mts = me.all_metrics(x, fine, cm, spec)
    assert x.shape == fine.shape
    assert mts["NVR"] == 0.0
    assert mts["MCE"] < 1e-3


def test_projection_removes_nonnegativity_violations():
    torch.manual_seed(1)
    spec = p.ChannelSpec(precip=0, u=1, v=2)
    ds = d.SyntheticDownscaleDataset(n=2, fine=32, factor=4, sat_bands=4)
    net = m.UNet(in_ch=3, cond_ch=7, base=16, ch_mults=(1, 2), attn_at=())
    rf = fl.RectifiedFlow(net, p.PhysicsLoss(spec))
    cond = torch.stack([ds[i]["cond"] for i in range(2)])
    cm = torch.stack([ds[i]["coarse_mass"] for i in range(2)])
    with torch.no_grad():
        raw = rf.sample((2, 3, 32, 32), cond, steps=8, device="cpu",
                        spec=spec, coarse_mass=cm, project=False)
        proj = rf.sample((2, 3, 32, 32), cond, steps=8, device="cpu",
                         spec=spec, coarse_mass=cm, project=True)
    assert me.nonneg_violation_rate(proj, spec) == 0.0
    assert me.nonneg_violation_rate(proj, spec) <= me.nonneg_violation_rate(raw, spec)
