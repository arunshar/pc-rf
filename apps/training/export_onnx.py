"""
Train a small PC-RF on the synthetic dataset and export the velocity network to
ONNX.

The exported graph is the velocity field v_theta(x_t, t, cond). The ODE sampler
loop and the inference-time physics projection run in the host (the Rust service
or serve_mock.py), calling this graph once per Euler step.

The graph is batch-dynamic and fixed-tile: the bottleneck self-attention flattens
H*W, which does not export as a clean dynamic dim, so downscaling is served
tile-by-tile at a fixed size (the standard pattern for SR / downscaling).

Usage:
  python export_onnx.py \
      --epochs 2 --fine 64 --base 32 --out artifacts/pcrf.onnx
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import torch

CHANNEL_SPEC = {"precip": 0, "u": 1, "v": 2}


def load_pcrf_modules(code_dir=None):
    """Return the PC-RF model modules (m, f, p, d) from the installed `pcrf`
    package. `code_dir` is accepted for backward compatibility and ignored: the
    model is the single source of truth in `pcrf`."""
    from pcrf import data as d
    from pcrf import flow as f
    from pcrf import model as m
    from pcrf import physics as p
    return m, f, p, d


def build_model(m, in_ch: int, cond_ch: int, base: int, attn: bool):
    attn_at = (1,) if attn else ()
    return m.UNet(in_ch=in_ch, cond_ch=cond_ch, base=base,
                  ch_mults=(1, 2), attn_at=attn_at)


def quick_train(rf, ds, epochs, batch, device):
    from torch.utils.data import DataLoader
    dl = DataLoader(ds, batch_size=batch, shuffle=True)
    opt = torch.optim.Adam(rf.parameters(), lr=2e-3)
    rf.train()
    logs = {"L_total": float("nan")}
    for ep in range(epochs):
        for b in dl:
            opt.zero_grad()
            loss, logs = rf.training_loss(
                b["fine"].to(device), b["cond"].to(device),
                b["coarse_mass"].to(device))
            loss.backward()
            opt.step()
        print(f"[train] epoch {ep + 1}/{epochs} L_total={logs['L_total']:.4f}")
    return logs


class VelocityWrapper(torch.nn.Module):
    """ONNX-exportable wrapper: forward(x_t, t, cond) -> velocity."""

    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, x_t, t, cond):
        return self.net(x_t, t, cond)


def export_velocity(net, in_ch: int, cond_ch: int, fine: int, out, opset=18,
                    extra_meta: dict | None = None):
    """Export the velocity field to ONNX (batch-dynamic, fixed tile), write the
    sidecar metadata, and return the cross-shape onnx-vs-torch max abs error
    (None if onnxruntime is unavailable).
    """
    net = net.eval().to("cpu")
    wrap = VelocityWrapper(net)
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)

    x = torch.randn(1, in_ch, fine, fine)
    t = torch.zeros(1)
    cond = torch.randn(1, cond_ch, fine, fine)

    # dynamo=False -> legacy exporter, which honors dynamic batch via dynamic_axes.
    dyn = {k: {0: "B"} for k in ("x_t", "t", "cond", "velocity")}
    torch.onnx.export(
        wrap, (x, t, cond), str(out),
        input_names=["x_t", "t", "cond"], output_names=["velocity"],
        dynamic_axes=dyn, opset_version=opset, do_constant_folding=True,
        dynamo=False,
    )

    meta = {
        "in_ch": in_ch, "cond_ch": cond_ch, "fine": fine, "tile": fine,
        "channel_spec": CHANNEL_SPEC, "opset": opset,
        "note": "velocity field v(x_t,t,cond); batch dynamic, spatial fixed to "
                "tile (tile-based serving); sampler + projection run host-side",
    }
    if extra_meta:
        meta.update(extra_meta)
    out.with_suffix(".json").write_text(json.dumps(meta, indent=2))

    # cross-shape verify at a DIFFERENT batch to catch baked-in static shapes.
    try:
        import numpy as np
        import onnxruntime as ort
        vb = 3
        xv = torch.randn(vb, in_ch, fine, fine)
        tv = torch.rand(vb)
        cv = torch.randn(vb, cond_ch, fine, fine)
        sess = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])
        o = sess.run(["velocity"], {"x_t": xv.numpy(), "t": tv.numpy(),
                                    "cond": cv.numpy()})[0]
        with torch.no_grad():
            r = wrap(xv, tv, cv).numpy()
        return float(np.abs(o - r).max())
    except Exception as e:  # noqa: BLE001
        print(f"[verify] skipped ({e})")
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="artifacts/pcrf.onnx")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--fine", type=int, default=64)
    ap.add_argument("--base", type=int, default=32)
    ap.add_argument("--sat-bands", type=int, default=4)
    ap.add_argument("--attn", action="store_true",
                    help="keep self-attention (heavier ONNX graph)")
    ap.add_argument("--compile", action="store_true",
                    help="torch.compile the training loop (PyTorch 2.x graph compile)")
    ap.add_argument("--opset", type=int, default=18)
    ap.add_argument("--ckpt", default=None,
                    help="bare state_dict for the network this script builds (two levels, "
                         "no attention unless --attn); a pcrf-train checkpoint does not fit")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    m, f, p, d = load_pcrf_modules()

    ds = d.SyntheticDownscaleDataset(n=64, fine=args.fine, factor=4,
                                     sat_bands=args.sat_bands)
    sample = ds[0]
    in_ch = sample["fine"].shape[0]
    cond_ch = sample["cond"].shape[0]
    print(f"[model] in_ch={in_ch} cond_ch={cond_ch} fine={args.fine} device={device}")

    net = build_model(m, in_ch, cond_ch, args.base, args.attn).to(device)
    train_target = net
    if args.compile:
        train_target = torch.compile(net)  # shares params with net; export stays eager
        print("[compile] torch.compile enabled for the training loop")
    rf = f.RectifiedFlow(
        train_target, p.PhysicsLoss(p.ChannelSpec(**CHANNEL_SPEC))).to(device)

    if args.ckpt:
        net.load_state_dict(torch.load(args.ckpt, map_location=device))
        print(f"[load] {args.ckpt}")
    else:
        quick_train(rf, ds, args.epochs, args.batch, device)

    err = export_velocity(
        net, in_ch, cond_ch, args.fine, args.out, opset=args.opset,
        extra_meta={"base": args.base, "ch_mults": [1, 2], "attn": bool(args.attn)},
    )
    out = Path(args.out)
    print(f"[onnx] wrote {out} ({out.stat().st_size / 1e6:.2f} MB)")
    print(f"[meta] wrote {out.with_suffix('.json')}")
    if err is not None:
        status = "OK" if err < 1e-3 else "CHECK (export drift)"
        print(f"[verify] cross-shape max|onnx - torch| = {err:.2e}  {status}")


if __name__ == "__main__":
    main()
