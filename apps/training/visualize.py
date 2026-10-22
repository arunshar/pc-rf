"""
Render a coarse-input vs PC-RF-downscaled comparison with the physics-validity
metrics annotated. Uses the synthetic dataset's conditioning so the downscaled
output is meaningful (not noise), and drives the actual ONNX graph through the
same sampler + projection the Rust service uses.

  python visualize.py \
      --onnx artifacts/pcrf.onnx --out artifacts/downscale_demo.png
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", default="artifacts/pcrf.onnx")
    ap.add_argument("--out", default="artifacts/downscale_demo.png")
    ap.add_argument("--ensemble", type=int, default=8)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--seed", type=int, default=3)
    args = ap.parse_args()

    import onnxruntime as ort
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from pcrf import data as d
    from pcrf.serving import math as sm

    meta = json.loads(Path(args.onnx).with_suffix(".json").read_text())
    tile, in_ch, cond_ch = meta["tile"], meta["in_ch"], meta["cond_ch"]

    ds = d.SyntheticDownscaleDataset(n=1, fine=tile, factor=4,
                                     sat_bands=cond_ch - 3, seed=args.seed)
    s = ds[0]
    fine = s["fine"].numpy()        # [3, H, W] ground truth (precip, u, v)
    cond = s["cond"].numpy()        # [cond_ch, H, W]
    cmass = float(s["coarse_mass"])
    coarse_precip = cond[0]         # upsampled coarse precip = the input

    sess = ort.InferenceSession(args.onnx, providers=["CPUExecutionProvider"])
    cond_b = np.repeat(cond[None].astype(np.float32), args.ensemble, axis=0)
    mean, metrics = sm.sample_onnx(
        sess, B=args.ensemble, tile=tile, in_ch=in_ch, cond_ch=cond_ch,
        steps=args.steps, coarse_mass=cmass, seed=args.seed, cond=cond_b, project=True)
    pred_precip, u, v = mean[0], mean[1], mean[2]

    du = (np.roll(u, -1, 1) - np.roll(u, 1, 1)) * 0.5
    dv = (np.roll(v, -1, 0) - np.roll(v, 1, 0)) * 0.5
    div = du + dv

    fig, ax = plt.subplots(1, 4, figsize=(16, 4.4))
    vmax = float(max(coarse_precip.max(), pred_precip.max(), fine[0].max(), 1e-6))
    panels = [
        (coarse_precip, "Coarse input precip\n(upsampled, the conditioning)", "Blues"),
        (pred_precip, "PC-RF downscaled precip", "Blues"),
        (fine[0], "Ground-truth fine precip", "Blues"),
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
        f"PC-RF physics-constrained downscaling   |   ensemble={args.ensemble}, "
        f"steps={args.steps}   |   DE={metrics['DE']:.4f}   NVR={metrics['NVR']:.4f}   "
        f"MCE={metrics['MCE']:.5f}   (coarse mass={cmass:.4f})",
        fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.92])

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130)
    print(f"[viz] wrote {out}")
    print(f"[viz] DE={metrics['DE']:.4f} NVR={metrics['NVR']:.5f} "
          f"MCE={metrics['MCE']:.6f} coarse_mass={cmass:.4f}")


if __name__ == "__main__":
    main()
