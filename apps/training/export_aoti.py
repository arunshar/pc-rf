"""
AOTInductor export: a PyTorch-native compiled serving artifact for the PC-RF
velocity field.

torch.export captures the graph and torch._inductor compiles it to a shared
library / package you can load from Python (or a C++/Rust runtime) with no ONNX
in the loop. This is the PyTorch-native counterpart to export_onnx.py, and the
"torch.onnx vs AOTInductor" comparison for the PyTorch Conference talk lives here.

Like the ONNX path, the graph is batch-dynamic and fixed-tile (the bottleneck
attention flattens H*W).

  python export_aoti.py --epochs 8 --fine 64 --base 32 --out artifacts/pcrf_aoti
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path

import torch

import export_onnx as ex  # reuse build_model, quick_train, loaders, wrapper, specs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="artifacts/pcrf_aoti")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--fine", type=int, default=64)
    ap.add_argument("--base", type=int, default=32)
    ap.add_argument("--sat-bands", type=int, default=4)
    ap.add_argument("--compile", action="store_true",
                    help="torch.compile the training loop too")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    m, f, p, d = ex.load_pcrf_modules()
    ds = d.SyntheticDownscaleDataset(n=64, fine=args.fine, factor=4,
                                     sat_bands=args.sat_bands)
    in_ch = ds[0]["fine"].shape[0]
    cond_ch = ds[0]["cond"].shape[0]
    print(f"[model] in_ch={in_ch} cond_ch={cond_ch} fine={args.fine} device={device}")

    net = ex.build_model(m, in_ch, cond_ch, args.base, attn=False).to(device)
    train_target = torch.compile(net) if args.compile else net
    rf = f.RectifiedFlow(
        train_target, p.PhysicsLoss(p.ChannelSpec(**ex.CHANNEL_SPEC))).to(device)
    ex.quick_train(rf, ds, args.epochs, args.batch, device)

    net = net.eval().to("cpu")
    wrap = ex.VelocityWrapper(net)
    # Use a batch-2 example: a batch-1 example makes torch.export specialize the
    # batch dim to the constant 1 and the dynamic-shape constraint then fails.
    x = torch.randn(2, in_ch, args.fine, args.fine)
    t = torch.zeros(2)
    cond = torch.randn(2, cond_ch, args.fine, args.fine)
    example = (x, t, cond)

    # Dynamic batch, fixed tile (mirrors the ONNX export).
    batch = torch.export.Dim("batch", min=1, max=4096)
    dynamic_shapes = ({0: batch}, {0: batch}, {0: batch})
    ep = torch.export.export(wrap, example, dynamic_shapes=dynamic_shapes)
    print("[aoti] torch.export captured the graph")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    # The AOTInductor packaging API has moved across torch releases; try the
    # modern package API first, then the .so fallback.
    artifact = None
    try:
        pkg = str(out.with_suffix(".pt2"))
        artifact = torch._inductor.aoti_compile_and_package(ep, package_path=pkg)
        print(f"[aoti] compiled package -> {artifact}")
    except Exception as e1:  # noqa: BLE001
        print(f"[aoti] aoti_compile_and_package unavailable ({e1}); trying aot_compile")
        try:
            so = torch._inductor.aot_compile(ep.module(), example)
            artifact = str(out.with_suffix(".so"))
            os.replace(so, artifact)
            print(f"[aoti] compiled shared library -> {artifact}")
        except Exception as e2:  # noqa: BLE001
            print(f"[aoti] FAILED to compile ({e2})")
            print("[aoti] AOTInductor needs a C++ compiler (clang) and the inductor "
                  "toolchain; the export graph above is still valid.")
            return

    meta = {
        "in_ch": in_ch, "cond_ch": cond_ch, "tile": args.fine,
        "channel_spec": ex.CHANNEL_SPEC, "runtime": "aotinductor",
        "artifact": Path(artifact).name,
    }
    Path(str(out) + ".json").write_text(json.dumps(meta, indent=2))
    print(f"[meta] wrote {out}.json")

    # Verify: reload the compiled artifact and compare to eager torch at a
    # different batch size.
    try:
        if artifact.endswith(".pt2"):
            runner = torch._inductor.aoti_load_package(artifact)
        else:
            runner = torch._export.aot_load(artifact, device="cpu")
        vb = 3
        xv = torch.randn(vb, in_ch, args.fine, args.fine)
        tv = torch.rand(vb)
        cv = torch.randn(vb, cond_ch, args.fine, args.fine)
        out_aoti = runner(xv, tv, cv)
        with torch.no_grad():
            ref = wrap(xv, tv, cv)
        err = float((out_aoti - ref).abs().max())
        status = "OK" if err < 1e-3 else "CHECK"
        print(f"[verify] batch={vb} max|aoti - torch| = {err:.2e}  {status}")
    except Exception as e:  # noqa: BLE001
        print(f"[verify] skipped ({e})")


if __name__ == "__main__":
    main()
