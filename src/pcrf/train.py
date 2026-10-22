"""
Train PC-RF.

Smoke test (synthetic, runs on CPU in a couple of minutes):
    pcrf-train --data synthetic --epochs 2 --batch 8 --fine 64 --base 32

Real run (ERA5; loaders are implemented in data.py):
    pcrf-train --data era5 --coarse /path/coarse.nc --target /path/target.nc \
        --fine 128 --epochs 200 --batch 16 --base 64 --l_div 1.0 --l_nn 1.0 --l_mass 1.0

Ablations: set any of --l_div/--l_nn/--l_mass to 0, or --no_physics for RF-base.
"""
from __future__ import annotations
import argparse, os, time, json
import torch
from torch.utils.data import DataLoader

from pcrf.data import SyntheticDownscaleDataset
from pcrf.model import UNet, SatelliteEncoder
from pcrf.flow import RectifiedFlow
from pcrf.physics import PhysicsLoss, ChannelSpec


def build_dataset(args):
    if args.data == "synthetic":
        return SyntheticDownscaleDataset(n=args.n, fine=args.fine,
                                         factor=args.factor,
                                         sat_bands=args.sat_bands)
    if args.data == "era5":
        from pcrf.data import ERA5Dataset
        coarse = args.coarse or os.path.join(args.root, "coarse.nc")
        target = args.target or os.path.join(args.root, "target.nc")
        return ERA5Dataset(coarse, target, split="train", fine=args.fine,
                           sat_bands=args.sat_bands)
    if args.data == "sentinel2":
        from pcrf.data import Sentinel2Dataset
        with open(args.scenes) as f:
            scenes = json.load(f)
        return Sentinel2Dataset(scenes, tile=args.fine)
    raise ValueError(args.data)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="synthetic",
                    choices=["synthetic", "era5", "sentinel2"])
    ap.add_argument("--root", default="")
    ap.add_argument("--coarse", default="")   # ERA5 coarse NetCDF (or glob)
    ap.add_argument("--target", default="")   # ERA5 high-resolution precip target
    ap.add_argument("--scenes", default="")   # Sentinel-2 scenes manifest (JSON)
    ap.add_argument("--n", type=int, default=256)        # synthetic size
    ap.add_argument("--fine", type=int, default=64)
    ap.add_argument("--factor", type=int, default=4)
    ap.add_argument("--sat_bands", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--base", type=int, default=32)
    ap.add_argument("--l_div", type=float, default=1.0)
    ap.add_argument("--l_nn", type=float, default=1.0)
    ap.add_argument("--l_mass", type=float, default=1.0)
    ap.add_argument("--no_physics", action="store_true")
    ap.add_argument("--out", default="checkpoints")
    ap.add_argument("--resume", action="store_true",
                    help="resume from <out>/pcrf_last.pt if present (for the 24h wall)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    ds = build_dataset(args)
    # num_workers=0: ERA5Dataset holds open netCDF/HDF5 handles, which are not
    # fork-safe; worker processes deadlock the DataLoader (especially on the
    # open_mfdataset/dask glob path). Single-process loading was fast enough for the
    # small staged dataset.
    dl = DataLoader(ds, batch_size=args.batch, shuffle=True, num_workers=0,
                    drop_last=True)

    sample = ds[0]
    in_ch = sample["fine"].shape[0]
    cond_ch = sample["cond"].shape[0]
    spec = ChannelSpec(precip=0, u=1, v=2) if in_ch >= 3 else ChannelSpec(precip=0)

    model = UNet(in_ch=in_ch, cond_ch=cond_ch, base=args.base).to(args.device)
    physics = None if args.no_physics else PhysicsLoss(
        spec, l_div=args.l_div, l_nn=args.l_nn, l_mass=args.l_mass)
    rf = RectifiedFlow(model, physics).to(args.device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"device={args.device}  params={n_params/1e6:.2f}M  "
          f"in_ch={in_ch} cond_ch={cond_ch}  physics={not args.no_physics}")

    ckpt_path = os.path.join(args.out, "pcrf_last.pt")
    start_epoch, step = 0, 0
    if args.resume and os.path.exists(ckpt_path):
        ck = torch.load(ckpt_path, map_location=args.device)
        model.load_state_dict(ck["model"])
        if "opt" in ck:
            opt.load_state_dict(ck["opt"])
        start_epoch = int(ck.get("epoch", -1)) + 1
        step = int(ck.get("step", 0))
        print(f"[resume] {ckpt_path}: continuing at epoch {start_epoch}")

    for epoch in range(start_epoch, args.epochs):
        model.train()
        t0 = time.time()
        for batch in dl:
            fine = batch["fine"].to(args.device)
            cond = batch["cond"].to(args.device)
            cmass = batch["coarse_mass"].to(args.device)
            loss, logs = rf.training_loss(fine, cond, cmass)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            step += 1
            if step % 20 == 0:
                msg = " ".join(f"{k}={v:.4f}" for k, v in logs.items())
                print(f"e{epoch} s{step} {msg}")
        print(f"epoch {epoch} done in {time.time()-t0:.1f}s")
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                    "args": vars(args), "in_ch": in_ch, "cond_ch": cond_ch,
                    "epoch": epoch, "step": step},
                   ckpt_path)

    with open(os.path.join(args.out, "train_config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)
    # DONE marker: the self-resubmitting sbatch checks for this to stop the chain.
    open(os.path.join(args.out, "DONE"), "w").close()
    print("saved checkpoint to", ckpt_path, "(DONE)")


if __name__ == "__main__":
    main()
