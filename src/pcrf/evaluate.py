"""
Evaluate a trained PC-RF checkpoint and print the metrics table.

    pcrf-evaluate --ckpt checkpoints/pcrf_last.pt --data synthetic \
        --fine 64 --batch 16 --steps 20 --ensemble 8

Reports PSNR, SSIM, CRPS (perceptual) and DE, NVR, MCE (physics).
For ablations and baselines, evaluate each checkpoint separately.
"""
from __future__ import annotations
import argparse, json, os, torch
from pathlib import Path
from torch.utils.data import DataLoader

from pcrf.data import SyntheticDownscaleDataset
from pcrf.model import UNet
from pcrf.flow import RectifiedFlow
from pcrf.physics import ChannelSpec
from pcrf.metrics import all_metrics, crps_ensemble


def build_eval_dataset(args):
    if args.data == "synthetic":
        return SyntheticDownscaleDataset(n=args.n, fine=args.fine,
                                         factor=args.factor,
                                         sat_bands=args.sat_bands, seed=999)
    if args.data == "era5":
        from pcrf.data import ERA5Dataset
        coarse = args.coarse or os.path.join(args.root, "coarse.nc")
        target = args.target or os.path.join(args.root, "target.nc")
        return ERA5Dataset(coarse, target, split="test", fine=args.fine,
                           sat_bands=args.sat_bands)
    if args.data == "sentinel2":
        from pcrf.data import Sentinel2Dataset
        with open(args.scenes) as f:
            scenes = json.load(f)
        return Sentinel2Dataset(scenes, tile=args.fine)
    raise ValueError(args.data)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data", default="synthetic",
                    choices=["synthetic", "era5", "sentinel2"])
    ap.add_argument("--root", default="")
    ap.add_argument("--coarse", default="")
    ap.add_argument("--target", default="")
    ap.add_argument("--scenes", default="")
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--fine", type=int, default=64)
    ap.add_argument("--factor", type=int, default=4)
    ap.add_argument("--sat_bands", type=int, default=4)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--steps", type=int, default=20)      # ODE steps
    ap.add_argument("--ensemble", type=int, default=8)    # CRPS members
    ap.add_argument("--no_project", action="store_true")
    ap.add_argument("--s2_nonneg", action="store_true",
                    help="Sentinel-2 PC-RF arm: clamp reflectance >=0 at inference "
                         "(the meaningful optical projection; the mass projection is "
                         "undefined for reflectance so it is skipped for --data sentinel2).")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--json_out", default="",
                    help="record metrics into this JSON for make_tables.py")
    ap.add_argument("--json_method", default="PC-RF",
                    help="method/row label under which to record the metrics")
    ap.add_argument("--json_dataset", default="",
                    help="table key (era5/s2/ablation); defaults from --data")
    args = ap.parse_args()

    ck = torch.load(args.ckpt, map_location=args.device)
    in_ch, cond_ch = ck["in_ch"], ck["cond_ch"]
    base = ck["args"].get("base", 32)
    spec = ChannelSpec(precip=0, u=1, v=2) if in_ch >= 3 else ChannelSpec(precip=0)

    model = UNet(in_ch=in_ch, cond_ch=cond_ch, base=base).to(args.device)
    model.load_state_dict(ck["model"])
    model.eval()
    rf = RectifiedFlow(model, None).to(args.device)

    ds = build_eval_dataset(args)
    dl = DataLoader(ds, batch_size=args.batch, shuffle=False)

    agg, nb = {}, 0
    for batch in dl:
        fine = batch["fine"].to(args.device)
        cond = batch["cond"].to(args.device)
        cmass = batch["coarse_mass"].to(args.device)
        shape = fine.shape

        # point prediction (mean over a small ensemble for stability). The mass
        # projection is only meaningful for precip fields, so it is off for S2;
        # the S2 optical constraint (reflectance >= 0) is applied via --s2_nonneg.
        phys_project = (not args.no_project) and args.data != "sentinel2"
        members = torch.stack([
            rf.sample(shape, cond, steps=args.steps, device=args.device,
                      spec=spec, coarse_mass=cmass, project=phys_project)
            for _ in range(args.ensemble)
        ], dim=0)
        if args.s2_nonneg:
            members = members.clamp(min=0.0)
        pred = members.mean(0)

        m = all_metrics(pred, fine, cmass, spec)
        m["CRPS"] = crps_ensemble(members, fine)
        for k, v in m.items():
            agg[k] = agg.get(k, 0.0) + v
        nb += 1

    print("\n==== PC-RF evaluation (%s, %d batches) ====" % (args.data, nb))
    for k in ["PSNR", "SSIM", "SAM", "CRPS", "DE", "NVR", "MCE"]:
        if k in agg:
            print(f"  {k:5s}: {agg[k]/nb:.4f}")
    print("=========================================\n")

    if args.json_out:
        means = {k: round(agg[k] / nb, 6) for k in agg}
        key = args.json_dataset or {"sentinel2": "s2"}.get(args.data, args.data)
        path = Path(args.json_out)
        blob = json.loads(path.read_text()) if path.exists() else {}
        blob.setdefault(key, {})[args.json_method] = means
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(blob, indent=2, sort_keys=True))
        print(f"[json] recorded {key}/{args.json_method} -> {path}")


if __name__ == "__main__":
    main()
