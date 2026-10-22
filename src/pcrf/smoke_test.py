"""
End-to-end shape/grad smoke test for the PC-RF pipeline (CPU, ~seconds).
Validates: dataset schema, model forward, training loss + backward,
sampling + physics projection, and all metrics. Tiny sizes on purpose.

    python -m pcrf.smoke_test        # or the console script: pcrf-smoke
"""
import torch
from pcrf.data import SyntheticDownscaleDataset
from pcrf.model import UNet
from pcrf.flow import RectifiedFlow
from pcrf.physics import PhysicsLoss, ChannelSpec
from pcrf.metrics import all_metrics, crps_ensemble


def main():
    torch.manual_seed(0)
    print("torch", torch.__version__)

    # 1) dataset -------------------------------------------------------------
    ds = SyntheticDownscaleDataset(n=8, fine=32, factor=4, sat_bands=4)
    s = ds[0]
    in_ch, cond_ch = s["fine"].shape[0], s["cond"].shape[0]
    print(f"[dataset] fine={tuple(s['fine'].shape)} cond={tuple(s['cond'].shape)} "
          f"coarse_mass={float(s['coarse_mass']):.4f}")
    assert s["fine"].shape[-2:] == s["cond"].shape[-2:], "fine/cond res mismatch"

    # 2) batch ---------------------------------------------------------------
    B = 2
    fine = torch.stack([ds[i]["fine"] for i in range(B)])
    cond = torch.stack([ds[i]["cond"] for i in range(B)])
    cmass = torch.stack([ds[i]["coarse_mass"] for i in range(B)])

    # 3) model + flow --------------------------------------------------------
    spec = ChannelSpec(precip=0, u=1, v=2)
    model = UNet(in_ch=in_ch, cond_ch=cond_ch, base=16, ch_mults=(1, 2), attn_at=(1,))
    physics = PhysicsLoss(spec, l_div=1.0, l_nn=1.0, l_mass=1.0)
    rf = RectifiedFlow(model, physics)
    print(f"[model] params={sum(p.numel() for p in model.parameters())/1e6:.3f}M")

    # 4) training loss + backward -------------------------------------------
    loss, logs = rf.training_loss(fine, cond, cmass)
    loss.backward()
    gnorm = sum(p.grad.abs().sum() for p in model.parameters() if p.grad is not None)
    print(f"[train] loss={logs['L_total']:.4f} logs={ {k: round(v,4) for k,v in logs.items()} }")
    assert torch.isfinite(loss), "non-finite loss"
    assert gnorm > 0, "no gradient flowed"
    print(f"[train] grad flowed OK (sum|grad|={float(gnorm):.2f})")

    # 5) sampling + projection ----------------------------------------------
    with torch.no_grad():
        x = rf.sample(fine.shape, cond, steps=5, device="cpu",
                      spec=spec, coarse_mass=cmass, project=True)
    print(f"[sample] out={tuple(x.shape)} "
          f"precip_min={float(x[:,0].min()):.4f} (should be >= 0 after projection)")
    assert x.shape == fine.shape, "sample shape mismatch"
    assert float(x[:, 0].min()) >= -1e-6, "projection failed: negative precip remains"

    # 6) metrics -------------------------------------------------------------
    m = all_metrics(x, fine, cmass, spec)
    ens = torch.stack([rf.sample(fine.shape, cond, steps=5, device="cpu",
                                 spec=spec, coarse_mass=cmass) for _ in range(3)])
    m["CRPS"] = crps_ensemble(ens, fine)
    print("[metrics]", {k: round(v, 4) for k, v in m.items()})
    for k, v in m.items():
        assert v == v, f"metric {k} is NaN"   # NaN check

    # 7) ablation path (no physics) -----------------------------------------
    rf_base = RectifiedFlow(model, None)
    lb, lg = rf_base.training_loss(fine, cond, cmass)
    print(f"[ablation] RF-base loss={float(lb):.4f} (physics off) OK")

    print("\nALL SMOKE TESTS PASSED")


if __name__ == "__main__":
    main()
