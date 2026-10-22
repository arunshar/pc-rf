"""
Render Figure 3: PC-RF physics-constrained downscaling on a REAL ERA5 test tile.

Four panels, left to right: the coarse (upsampled) precip input that conditions
the model, the PC-RF downscaled precip (ensemble mean, with the inference
projection), the ground-truth fine precip, and the downscaled wind over its
divergence map. Physics-validity metrics (DE, NVR, MCE) are annotated.

Unlike apps/training/visualize.py (which drives the ONNX graph on the synthetic
dataset), this loads the trained .pt checkpoint directly and pulls a held-out ERA5
tile, so the figure matches the paper's ERA5 tables.

  python apps/training/make_figure3.py \
      --ckpt runs/pcrf_full/pcrf_last.pt \
      --coarse .../era5_2020/coarse.nc --target .../era5_2020/target.nc \
      --out results/figure3_era5.png
"""
from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
import torch

from pcrf.data import ERA5Dataset
from pcrf.model import UNet
from pcrf.flow import RectifiedFlow
from pcrf.physics import ChannelSpec
from pcrf.metrics import all_metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--coarse", required=True)
    ap.add_argument("--target", required=True)
    ap.add_argument("--out", default="results/figure3_era5.png")
    ap.add_argument("--fine", type=int, default=128)
    ap.add_argument("--sat_bands", type=int, default=4)
    ap.add_argument("--idx", type=int, default=0, help="test-tile index")
    ap.add_argument("--ensemble", type=int, default=8)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    device = args.device
    ck = torch.load(args.ckpt, map_location=device)
    in_ch, cond_ch = ck["in_ch"], ck["cond_ch"]
    base = ck.get("args", {}).get("base", 64)
    spec = ChannelSpec(precip=0, u=1, v=2) if in_ch >= 3 else ChannelSpec(precip=0)
    model = UNet(in_ch=in_ch, cond_ch=cond_ch, base=base).to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    rf = RectifiedFlow(model, None).to(device)

    ds = ERA5Dataset(args.coarse, args.target, split="test", fine=args.fine,
                     sat_bands=args.sat_bands)
    s = ds[args.idx]
    fine = s["fine"][None].to(device)              # [1,3,H,W]
    cond = s["cond"][None].to(device)              # [1,cond_ch,H,W]
    cmass = s["coarse_mass"].reshape(1).to(device)

    with torch.no_grad():
        members = torch.stack([
            rf.sample(fine.shape, cond, steps=args.steps, device=device,
                      spec=spec, coarse_mass=cmass, project=True)
            for _ in range(args.ensemble)], dim=0)
        mean = members.mean(0)                      # [1,3,H,W]

    m = all_metrics(mean, fine, cmass, spec)

    mean_np = mean[0].cpu().numpy()
    fine_np = fine[0].cpu().numpy()
    coarse_precip = cond[0, 0].cpu().numpy()
    pred_precip, u, v = mean_np[0], mean_np[1], mean_np[2]
    tile = pred_precip.shape[-1]

    du = (np.roll(u, -1, 1) - np.roll(u, 1, 1)) * 0.5
    dv = (np.roll(v, -1, 0) - np.roll(v, 1, 0)) * 0.5
    div = du + dv

    fig, ax = plt.subplots(1, 4, figsize=(16, 4.4))
    vmax = float(max(coarse_precip.max(), pred_precip.max(), fine_np[0].max(), 1e-6))
    panels = [
        (coarse_precip, "Coarse input precip\n(upsampled, the conditioning)", "Blues"),
        (pred_precip, "PC-RF downscaled precip", "Blues"),
        (fine_np[0], "Ground-truth fine precip", "Blues"),
    ]
    for a, (img, title, cmap) in zip(ax[:3], panels):
        im = a.imshow(img, cmap=cmap, vmin=0, vmax=vmax)
        a.set_title(title, fontsize=10)
        fig.colorbar(im, ax=a, fraction=0.046, pad=0.04)

    dmax = float(np.abs(div).max() + 1e-8)
    im = ax[3].imshow(div, cmap="RdBu_r", vmin=-dmax, vmax=dmax)
    st = max(tile // 16, 1)
    yy, xx = np.mgrid[0:tile:st, 0:tile:st]
    norm = float(np.hypot(u, v).max() + 1e-8)
    ax[3].quiver(xx, yy, u[::st, ::st] / norm, v[::st, ::st] / norm,
                 scale=25, color="k", alpha=0.6)
    ax[3].set_title("Downscaled wind + divergence", fontsize=10)
    fig.colorbar(im, ax=ax[3], fraction=0.046, pad=0.04)

    for a in ax:
        a.set_xticks([])
        a.set_yticks([])

    fig.suptitle(
        f"PC-RF physics-constrained downscaling on ERA5 (held-out tile)   |   "
        f"ensemble={args.ensemble}, steps={args.steps}   |   "
        f"DE={m['DE']:.4f}   NVR={m['NVR']:.4f}   MCE={m['MCE']:.5f}   "
        f"(coarse mass={float(cmass):.4f})", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.92])

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130)
    print(f"[fig3] wrote {out}")
    print(f"[fig3] DE={m['DE']:.4f} NVR={m['NVR']:.5f} MCE={m['MCE']:.6f} "
          f"PSNR={m['PSNR']:.2f} SSIM={m['SSIM']:.4f} coarse_mass={float(cmass):.4f}")


if __name__ == "__main__":
    main()
