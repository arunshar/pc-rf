"""
Serving benchmark for the PC-RF velocity network at its trained size.

Measures, for one configuration of the network, the numbers the PyTorch
Conference talk needs and the repository never recorded:

  * export parity against eager PyTorch, for ONNX Runtime and AOTInductor, at
    several batch sizes (catches an export that baked in a static shape);
  * latency (p50, p99, mean) and samples per second for one velocity-network
    call, per backend and batch size;
  * the same for a full sample: the Euler ODE loop on the host plus the
    physics projection, which is how the model is served.

Backends: eager fp32, torch.compile, AOTInductor, and ONNX Runtime (CUDA
execution provider when available, otherwise CPU; the JSON records which).

The network defaults to the trained configuration (UNet in_ch=3, cond_ch=7,
base=64, ch_mults=(1, 2, 4), self-attention at level 2 and in the bottleneck,
128 x 128 tile). Pass --ckpt to load trained weights; without it the weights
are random and the JSON says so. Latency and parity depend on the architecture,
so random weights still give valid serving numbers, labeled as such.

  python apps/training/bench_serving.py --out results/serving_bench.json
  python apps/training/bench_serving.py --ckpt pcrf_last.pt --out results/serving_bench.json

The JSON carries no file paths, only a checkpoint content hash, so it can be
pulled back from the cluster as-is.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import tempfile
import time
from pathlib import Path

import export_onnx as ex  # VelocityWrapper, CHANNEL_SPEC, load_pcrf_modules
import numpy as np
import torch

SCHEMA_VERSION = 1
BACKENDS = ("eager", "compile", "aoti", "ort")


# ----------------------------------------------------------------------------
# model
# ----------------------------------------------------------------------------
def build_net(m, in_ch: int, cond_ch: int, base: int, tiny: bool):
    """The trained configuration, or a tiny one with the same topology."""
    if tiny:
        return m.UNet(in_ch=in_ch, cond_ch=cond_ch, base=base,
                      ch_mults=(1, 2), attn_at=(1,))
    return m.UNet(in_ch=in_ch, cond_ch=cond_ch, base=base)


def load_weights(net, ckpt: str | None) -> dict:
    if not ckpt:
        return {"weights": "random", "ckpt_sha256": None}
    raw = Path(ckpt).read_bytes()
    ck = torch.load(ckpt, map_location="cpu")
    state = ck["model"] if isinstance(ck, dict) and "model" in ck else ck
    net.load_state_dict(state)
    return {"weights": "trained", "ckpt_sha256": hashlib.sha256(raw).hexdigest()}


def n_params(net) -> int:
    return int(sum(p.numel() for p in net.parameters()))


# ----------------------------------------------------------------------------
# backends: each returns a callable velocity(x_t, t, cond) -> tensor on device
# ----------------------------------------------------------------------------
def make_eager(net):
    wrap = ex.VelocityWrapper(net).eval()

    @torch.no_grad()
    def run(x, t, c):
        return wrap(x, t, c)
    return run


def make_compile(net):
    comp = torch.compile(ex.VelocityWrapper(net).eval())

    @torch.no_grad()
    def run(x, t, c):
        return comp(x, t, c)
    return run


def make_aoti(net, in_ch, cond_ch, fine, device, workdir: Path):
    wrap = ex.VelocityWrapper(net).eval()
    # A batch-1 example makes torch.export specialize the batch dim to 1.
    example = (torch.randn(2, in_ch, fine, fine, device=device),
               torch.zeros(2, device=device),
               torch.randn(2, cond_ch, fine, fine, device=device))
    batch = torch.export.Dim("batch", min=1, max=4096)
    ep = torch.export.export(wrap, example,
                             dynamic_shapes=({0: batch}, {0: batch}, {0: batch}))
    pkg = torch._inductor.aoti_compile_and_package(
        ep, package_path=str(workdir / "pcrf_velocity.pt2"))
    runner = torch._inductor.aoti_load_package(pkg)

    @torch.no_grad()
    def run(x, t, c):
        return runner(x, t, c)
    return run


def make_ort(net, in_ch, cond_ch, fine, device, workdir: Path):
    import onnxruntime as ort
    path = workdir / "pcrf_velocity.onnx"
    wrap = ex.VelocityWrapper(net).eval().to("cpu")
    dyn = {k: {0: "B"} for k in ("x_t", "t", "cond", "velocity")}
    torch.onnx.export(
        wrap, (torch.randn(1, in_ch, fine, fine), torch.zeros(1),
               torch.randn(1, cond_ch, fine, fine)),
        str(path), input_names=["x_t", "t", "cond"], output_names=["velocity"],
        dynamic_axes=dyn, opset_version=18, do_constant_folding=True, dynamo=False)
    net.to(device)
    providers = ["CPUExecutionProvider"]
    if device == "cuda" and "CUDAExecutionProvider" in ort.get_available_providers():
        # onnxruntime-gpu can reuse the CUDA and cuDNN libraries that the torch
        # wheels ship; without this the CUDA provider may silently fall back to CPU.
        if hasattr(ort, "preload_dlls"):
            ort.preload_dlls()
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    opts = ort.SessionOptions()
    # Cap threads at the CPUs Slurm granted; the default spawns one per core on the
    # node and oversubscribes the allocation.
    opts.intra_op_num_threads = int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 4))
    opts.inter_op_num_threads = 1
    sess = ort.InferenceSession(str(path), sess_options=opts, providers=providers)
    used = sess.get_providers()[0]
    if device == "cuda" and used != "CUDAExecutionProvider":
        used = f"{used} (CUDA provider unavailable, fell back)"

    def run(x, t, c):
        # Host round trip is part of this backend: numpy in, numpy out.
        o = sess.run(["velocity"], {"x_t": x.detach().cpu().numpy(),
                                    "t": t.detach().cpu().numpy(),
                                    "cond": c.detach().cpu().numpy()})[0]
        return torch.from_numpy(o).to(x.device)
    return run, used


# ----------------------------------------------------------------------------
# measurement
# ----------------------------------------------------------------------------
def _sync(device):
    if device == "cuda":
        torch.cuda.synchronize()


def time_call(fn, device, warmup: int, iters: int) -> dict:
    for _ in range(warmup):
        fn()
    _sync(device)
    lat = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        _sync(device)
        lat.append(time.perf_counter() - t0)
    a = np.asarray(lat) * 1e3
    return {"p50_ms": float(np.percentile(a, 50)), "p99_ms": float(np.percentile(a, 99)),
            "mean_ms": float(a.mean()), "iters": iters}


def sample(velocity, x0, cond, coarse_mass, steps: int, project_fn, spec):
    """Euler integration from t=0 to t=1, then the physics projection."""
    x = x0
    dt = 1.0 / steps
    b = x.shape[0]
    for i in range(steps):
        t = torch.full((b,), i * dt, device=x.device)
        x = x + velocity(x, t, cond) * dt
    return project_fn(x, spec, coarse_mass)


def run_bench(args, save=None) -> dict:
    """Run every backend. `save(report)` is called after each backend finishes, so a
    job that hits its wall-time limit still leaves the completed backends on disk."""
    save = save or (lambda _r: None)
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)
    if args.tf32 == "off":
        # TF32 matmuls and convolutions trade precision for speed on Ampere GPUs.
        # Turning both off isolates how much of the export parity gap they cause.
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False

    m, _f, p, _d = ex.load_pcrf_modules()
    in_ch, cond_ch, fine = 3, args.cond_ch, args.fine
    net = build_net(m, in_ch, cond_ch, args.base, args.tiny)
    meta_w = load_weights(net, args.ckpt)
    net = net.eval().to(device)
    spec = p.ChannelSpec(**ex.CHANNEL_SPEC)

    report = {
        "schema_version": SCHEMA_VERSION,
        "config": {"in_ch": in_ch, "cond_ch": cond_ch, "fine": fine, "base": args.base,
                   "tiny": bool(args.tiny), "params": n_params(net), "steps": args.steps,
                   "batches": args.batches, "parity_batches": args.parity_batches,
                   "warmup": args.warmup, "iters": args.iters,
                   "sample_iters": args.sample_iters, "seed": args.seed,
                   "tf32": args.tf32, "parity_only": bool(args.parity_only),
                   "allow_tf32_matmul": bool(torch.backends.cuda.matmul.allow_tf32),
                   "allow_tf32_cudnn": bool(torch.backends.cudnn.allow_tf32)},
        "weights": meta_w,
        "env": env_info(device),
        "backends": {},
    }

    workdir = Path(tempfile.mkdtemp(prefix="pcrf_bench_"))
    velocities = {"eager": make_eager(net)}
    notes = {"eager": "fp32, torch.no_grad"}
    for name in args.backends:
        if name == "eager":
            continue
        try:
            if name == "compile":
                velocities[name] = make_compile(net)
                notes[name] = "torch.compile default mode"
            elif name == "aoti":
                velocities[name] = make_aoti(net, in_ch, cond_ch, fine, device, workdir)
                notes[name] = "torch.export, dynamic batch 1..4096, aoti_compile_and_package"
            elif name == "ort":
                velocities[name], used = make_ort(net, in_ch, cond_ch, fine, device, workdir)
                notes[name] = f"onnxruntime {used}, legacy exporter, opset 18, host copies included"
        except Exception as e:  # noqa: BLE001  (record any backend failure, keep going)
            report["backends"][name] = {"status": "failed", "error": type(e).__name__,
                                        "message": str(e)[:300]}
            save(report)

    gen = torch.Generator(device="cpu").manual_seed(args.seed)
    for name, vel in velocities.items():
        try:
            report["backends"][name] = measure_backend(
                name, vel, velocities["eager"], notes[name], args, device, gen,
                in_ch, cond_ch, fine, p, spec, timing=not args.parity_only)
        except Exception as e:  # noqa: BLE001  (record any backend failure, keep going)
            report["backends"][name] = {"status": "failed", "stage": "measure",
                                        "error": type(e).__name__, "message": str(e)[:300]}
        save(report)
    return report


def measure_backend(name, vel, eager, note, args, device, gen, in_ch, cond_ch, fine, p, spec,
                    timing=True):
    """Parity against eager, then velocity-call and full-sample timing, then the
    post-projection guarantee, for one backend."""
    entry = {"status": "ok", "note": note, "parity_max_abs_err": {},
             "velocity": {}, "sample": {}}
    # parity against eager at batch sizes other than the export example
    for b in args.parity_batches:
        x = torch.randn(b, in_ch, fine, fine, generator=gen).to(device)
        t = torch.rand(b, generator=gen).to(device)
        c = torch.randn(b, cond_ch, fine, fine, generator=gen).to(device)
        ref = eager(x, t, c)
        out = vel(x, t, c)
        entry["parity_max_abs_err"][str(b)] = float((out - ref).abs().max())
    for b in (args.batches if timing else []):
        x = torch.randn(b, in_ch, fine, fine, generator=gen).to(device)
        t = torch.rand(b, generator=gen).to(device)
        c = torch.randn(b, cond_ch, fine, fine, generator=gen).to(device)
        cm = torch.full((b,), 0.1, device=device)
        v = time_call(lambda x=x, t=t, c=c: vel(x, t, c), device, args.warmup, args.iters)
        v["samples_per_s"] = b / (v["mean_ms"] / 1e3)
        entry["velocity"][str(b)] = v
        s = time_call(lambda x=x, c=c, cm=cm: sample(vel, x, c, cm, args.steps, p.project_physics, spec),
                      device, max(1, args.warmup // 5), args.sample_iters)
        s["samples_per_s"] = b / (s["mean_ms"] / 1e3)
        entry["sample"][str(b)] = s
    # the guarantee must hold after projection, whatever the backend
    b = args.batches[0]
    x = torch.randn(b, in_ch, fine, fine, generator=gen).to(device)
    c = torch.randn(b, cond_ch, fine, fine, generator=gen).to(device)
    cm = torch.full((b,), 0.1, device=device)
    out = sample(vel, x, c, cm, args.steps, p.project_physics, spec)
    pr = out[:, spec.precip]
    entry["post_projection"] = {
        "nvr": float((pr < 0).float().mean()),
        "mce_max": float(((pr.mean(dim=(1, 2)) - cm).abs() / cm).max()),
    }
    return entry


def env_info(device) -> dict:
    info = {"torch": torch.__version__, "python": platform.python_version(),
            "device": device, "gpu": None, "onnxruntime": None, "git_sha": None}
    if device == "cuda":
        info["gpu"] = torch.cuda.get_device_name(0)
    try:
        import onnxruntime as ort
        info["onnxruntime"] = ort.__version__
    except Exception:  # noqa: BLE001, S110  (optional metadata)
        pass
    try:
        info["git_sha"] = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
            cwd=Path(__file__).resolve().parent, check=True).stdout.strip()
    except Exception:  # noqa: BLE001, S110  (optional metadata)
        pass
    return info


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--out", required=True)
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    ap.add_argument("--fine", type=int, default=128)
    ap.add_argument("--base", type=int, default=64)
    ap.add_argument("--cond-ch", type=int, default=7)
    ap.add_argument("--tiny", action="store_true",
                    help="two-level test network with the same topology")
    ap.add_argument("--backends", type=lambda s: s.split(","), default=list(BACKENDS))
    ap.add_argument("--batches", type=lambda s: [int(v) for v in s.split(",")],
                    default=[1, 8, 32])
    ap.add_argument("--parity-batches", type=lambda s: [int(v) for v in s.split(",")],
                    default=[1, 4, 8])
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--iters", type=int, default=100,
                    help="timed iterations per velocity-call measurement")
    ap.add_argument("--sample-iters", type=int, default=30,
                    help="timed iterations per full-sample measurement")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tf32", choices=["default", "off"], default="default",
                    help="'off' disables TF32 matmuls and cuDNN convolutions")
    ap.add_argument("--parity-only", action="store_true",
                    help="measure parity and the post-projection check, skip timing")
    args = ap.parse_args(argv)
    bad = [b for b in args.backends if b not in BACKENDS]
    if bad:
        ap.error(f"unknown backend(s) {bad}; choose from {BACKENDS}")
    return args


def main(argv=None):
    args = parse_args(argv)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    def save(report):
        tmp = out.with_suffix(out.suffix + ".tmp")
        tmp.write_text(json.dumps(report, indent=2))
        os.replace(tmp, out)

    report = run_bench(args, save=save)
    save(report)
    for name, e in report["backends"].items():
        if e.get("status") != "ok":
            print(f"[{name}] FAILED {e['error']}")
            continue
        line = f"[{name}] parity {max(e['parity_max_abs_err'].values()):.2e}"
        b = str(args.batches[-1])
        if b in e["velocity"]:
            line += (f"  velocity@{b} p50 {e['velocity'][b]['p50_ms']:.2f} ms"
                     f"  sample@{b} {e['sample'][b]['samples_per_s']:.1f} samples/s")
        print(line)
    print(f"wrote {out}")
    return report


if __name__ == "__main__":
    main()
