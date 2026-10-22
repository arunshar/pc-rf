"""
Baselines for the PC-RF ERA5 precipitation-downscaling comparison (paper Table 2 /
``results/era5_metrics.tex``). Each baseline maps the conditioning image (the
bicubically upsampled coarse precip + wind, plus zero satellite bands) to a fine
field, and is scored with the SAME held-out ERA5 test split and the SAME metric
functions (``pcrf.metrics.all_metrics`` / ``crps_ensemble``) as PC-RF, so the rows
are directly comparable.

Comparison choices:
  * Baselines are trained with their own standard losses (no PC-RF physics
    penalties) and evaluated WITHOUT the PC-RF inference projection. Their raw
    NVR/MCE therefore reflect the method itself; PC-RF's zeros come from its
    projection, a known technique (see KNOWN_ISSUES.md) that any baseline could use.
  * Deterministic baselines (bicubic, SRCNN, SRGAN) report a single-member
    "ensemble", for which the fair CRPS estimator degenerates to the mean
    absolute error, the correct treatment of a point forecast.
  * The diffusion baselines (DDPM, CorrDiff) reuse the SAME ``UNet`` as PC-RF's
    velocity network, so any quality gap reflects the objective (diffusion vs
    rectified flow + physics), not the backbone.

CLI:
  python -m pcrf.baselines --mode train --baseline ddpm --data era5 \
      --coarse coarse.nc --target target.nc --fine 128 --epochs 300 \
      --batch 32 --base 64 --out runs/baseline_ddpm --resume --device cuda
  python -m pcrf.baselines --mode eval  --baseline ddpm --data era5 \
      --coarse coarse.nc --target target.nc --fine 128 --batch 16 \
      --ensemble 8 --steps 100 --baseline_ckpt runs/baseline_ddpm/model.pt \
      --json_out results/era5_baseline_ddpm.json --json_dataset era5 --device cuda
"""
from __future__ import annotations
import argparse
import json
import os
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from pcrf.model import UNet
from pcrf.physics import ChannelSpec
from pcrf.metrics import all_metrics, crps_ensemble

METHODS = ["bicubic", "srcnn", "srgan", "ddpm", "corrdiff"]
STOCHASTIC = {"ddpm", "corrdiff"}
# Canonical row labels expected by apps/training/make_tables.py (Table 2).
LABEL = {"bicubic": "Bicubic", "srcnn": "SRCNN", "srgan": "SRGAN",
         "ddpm": "DDPM", "corrdiff": "CorrDiff"}


# --------------------------------------------------------------------- SRCNN --
class SRCNN(nn.Module):
    """SRCNN (Dong et al., 2015), adapted to consume the conditioning image.

    The conditioning image is already at fine resolution (bicubically upsampled
    coarse), which matches SRCNN's original design of operating on a bicubic
    upsampling of the low-resolution input; the network refines it.
    """

    def __init__(self, cond_ch: int, out_ch: int = 3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(cond_ch, 64, 9, padding=4), nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, 1, padding=0), nn.ReLU(inplace=True),
            nn.Conv2d(32, out_ch, 5, padding=2),
        )

    def forward(self, cond):
        return self.net(cond)


# ----------------------------------------------------- SRGAN (SRResNet) ------
class _ResBlock(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.c1 = nn.Conv2d(ch, ch, 3, padding=1)
        self.b1 = nn.BatchNorm2d(ch)
        self.c2 = nn.Conv2d(ch, ch, 3, padding=1)
        self.b2 = nn.BatchNorm2d(ch)
        self.act = nn.PReLU()

    def forward(self, x):
        h = self.act(self.b1(self.c1(x)))
        h = self.b2(self.c2(h))
        return x + h


class SRResNet(nn.Module):
    """SRGAN generator. The conditioning image is already fine-resolution, so no
    pixel-shuffle upsampling stage is needed (a refinement generator)."""

    def __init__(self, cond_ch: int, out_ch: int = 3, ch: int = 64,
                 n_blocks: int = 8):
        super().__init__()
        self.head = nn.Sequential(nn.Conv2d(cond_ch, ch, 9, padding=4), nn.PReLU())
        self.body = nn.Sequential(*[_ResBlock(ch) for _ in range(n_blocks)])
        self.body_tail = nn.Sequential(nn.Conv2d(ch, ch, 3, padding=1),
                                       nn.BatchNorm2d(ch))
        self.tail = nn.Conv2d(ch, out_ch, 9, padding=4)

    def forward(self, cond):
        h = self.head(cond)
        h = h + self.body_tail(self.body(h))
        return self.tail(h)


class Discriminator(nn.Module):
    """PatchGAN discriminator on the fine field (returns per-patch logits)."""

    def __init__(self, in_ch: int = 3, ch: int = 64):
        super().__init__()

        def blk(i, o, s):
            return nn.Sequential(
                nn.Conv2d(i, o, 3, stride=s, padding=1),
                nn.BatchNorm2d(o), nn.LeakyReLU(0.2, inplace=True))

        self.net = nn.Sequential(
            nn.Conv2d(in_ch, ch, 3, padding=1), nn.LeakyReLU(0.2, inplace=True),
            blk(ch, ch, 2), blk(ch, ch * 2, 1), blk(ch * 2, ch * 2, 2),
            blk(ch * 2, ch * 4, 1), blk(ch * 4, ch * 4, 2),
            nn.Conv2d(ch * 4, 1, 3, padding=1),
        )

    def forward(self, x):
        return self.net(x)


# ---------------------------------------------------- Gaussian diffusion -----
class GaussianDiffusion:
    """DDPM forward process + DDIM sampler, shared by the DDPM and CorrDiff
    baselines. The denoiser is a UNet predicting epsilon(x_t, t/T, cond).

    Not an nn.Module: its schedule tensors are plain buffers moved by ``.to``.
    """

    def __init__(self, T: int = 1000, device="cpu"):
        self.T = int(T)
        betas = torch.linspace(1e-4, 2e-2, self.T, device=device)
        self.betas = betas
        self.alphas = 1.0 - betas
        self.acp = torch.cumprod(self.alphas, dim=0)      # alpha-bar

    def to(self, device):
        self.betas = self.betas.to(device)
        self.alphas = self.alphas.to(device)
        self.acp = self.acp.to(device)
        return self

    def q_sample(self, x0, t, noise):
        a = self.acp[t].view(-1, 1, 1, 1)
        return a.sqrt() * x0 + (1 - a).sqrt() * noise

    def p_losses(self, denoiser, x0, cond):
        b = x0.shape[0]
        t = torch.randint(0, self.T, (b,), device=x0.device)
        noise = torch.randn_like(x0)
        x_t = self.q_sample(x0, t, noise)
        pred = denoiser(x_t, t.float() / self.T, cond)
        return F.mse_loss(pred, noise)

    @torch.no_grad()
    def ddim_sample(self, denoiser, cond, shape, steps=100, eta=1.0, device="cpu"):
        """DDIM sampling over a strided timestep grid. eta=1 is stochastic
        (an ancestral-like sampler that gives ensemble spread for CRPS);
        eta=0 is deterministic."""
        ts = torch.linspace(self.T - 1, 0, steps, device=device).round().long()
        x = torch.randn(shape, device=device)
        for i in range(steps):
            t = int(ts[i])
            t_b = torch.full((shape[0],), t, device=device, dtype=torch.long)
            eps = denoiser(x, t_b.float() / self.T, cond)
            a_t = self.acp[t]
            x0 = (x - (1 - a_t).sqrt() * eps) / a_t.sqrt().clamp(min=1e-8)
            if i < steps - 1:
                a_prev = self.acp[int(ts[i + 1])]
                sigma = eta * ((1 - a_prev) / (1 - a_t)).clamp(min=0).sqrt() \
                    * (1 - a_t / a_prev).clamp(min=0).sqrt()
                dir_coef = (1 - a_prev - sigma ** 2).clamp(min=0).sqrt()
                x = a_prev.sqrt() * x0 + dir_coef * eps + sigma * torch.randn_like(x)
            else:
                x = x0
        return x


class DDPMBaseline(nn.Module):
    def __init__(self, cond_ch: int, out_ch: int = 3, base: int = 64, T: int = 1000):
        super().__init__()
        self.denoiser = UNet(in_ch=out_ch, cond_ch=cond_ch, base=base)
        self.diff = GaussianDiffusion(T=T)
        self.out_ch = out_ch

    def to(self, *a, **k):
        super().to(*a, **k)
        dev = a[0] if a else k.get("device", "cpu")
        self.diff.to(dev)
        return self

    def training_loss(self, x0, cond, coarse_mass=None):
        return self.diff.p_losses(self.denoiser, x0, cond)

    @torch.no_grad()
    def sample(self, cond, n, steps=100, eta=1.0, device="cpu"):
        b, _, H, W = cond.shape
        shape = (b, self.out_ch, H, W)
        return torch.stack(
            [self.diff.ddim_sample(self.denoiser, cond, shape, steps, eta, device)
             for _ in range(n)], dim=0)


class CorrDiffBaseline(nn.Module):
    """Two-stage CorrDiff: a deterministic regression mean plus a diffusion
    residual, mirroring CorrDiff's regression + generative-residual structure."""

    def __init__(self, cond_ch: int, out_ch: int = 3, base: int = 64, T: int = 1000):
        super().__init__()
        self.reg = UNet(in_ch=out_ch, cond_ch=cond_ch, base=base)
        self.denoiser = UNet(in_ch=out_ch, cond_ch=cond_ch, base=base)
        self.diff = GaussianDiffusion(T=T)
        self.out_ch = out_ch

    def to(self, *a, **k):
        super().to(*a, **k)
        dev = a[0] if a else k.get("device", "cpu")
        self.diff.to(dev)
        return self

    def _mean(self, cond):
        b, _, H, W = cond.shape
        zeros = torch.zeros(b, self.out_ch, H, W, device=cond.device)
        t0 = torch.zeros(b, device=cond.device)
        return self.reg(zeros, t0, cond)

    def training_loss(self, x0, cond, coarse_mass=None):
        mean = self._mean(cond)
        reg_loss = F.mse_loss(mean, x0)
        resid = x0 - mean.detach()
        diff_loss = self.diff.p_losses(self.denoiser, resid, cond)
        return reg_loss + diff_loss

    @torch.no_grad()
    def sample(self, cond, n, steps=100, eta=1.0, device="cpu"):
        b, _, H, W = cond.shape
        mean = self._mean(cond)
        shape = (b, self.out_ch, H, W)
        members = [mean + self.diff.ddim_sample(self.denoiser, cond, shape, steps,
                                                eta, device) for _ in range(n)]
        return torch.stack(members, dim=0)


# ---------------------------------------------------------------- factory ----
def build_baseline(name: str, cond_ch: int, out_ch: int = 3, base: int = 64,
                   T: int = 1000):
    if name == "srcnn":
        return SRCNN(cond_ch, out_ch)
    if name == "srgan":
        return SRResNet(cond_ch, out_ch)
    if name == "ddpm":
        return DDPMBaseline(cond_ch, out_ch, base, T)
    if name == "corrdiff":
        return CorrDiffBaseline(cond_ch, out_ch, base, T)
    if name == "bicubic":
        return None
    raise ValueError(f"unknown baseline: {name}")


@torch.no_grad()
def predict_members(name, model, batch, n, steps, eta, device, out_ch=3):
    """Return ensemble members [M, B, C, H, W] for a batch."""
    cond = batch["cond"].to(device)
    if name == "bicubic":
        return cond[:, :out_ch].unsqueeze(0)
    if name in ("srcnn", "srgan"):
        return model(cond).unsqueeze(0)
    if name in ("ddpm", "corrdiff"):
        return model.sample(cond, n=n, steps=steps, eta=eta, device=device)
    raise ValueError(name)


# ----------------------------------------------------------- data helpers ----
def _build_train_dataset(args):
    if args.data == "sentinel2":
        from pcrf.data import Sentinel2Dataset
        with open(args.scenes) as f:
            scenes = json.load(f)
        return Sentinel2Dataset(scenes, tile=args.fine)
    from pcrf.data import ERA5Dataset
    return ERA5Dataset(args.coarse, args.target, split="train", fine=args.fine,
                       sat_bands=args.sat_bands)


# --------------------------------------------------------------- training ----
def train_baseline(name, args):
    device = args.device
    out = args.out or f"runs/baseline_{name}"
    os.makedirs(out, exist_ok=True)
    ckpt_path = os.path.join(out, "model.pt")

    ds = _build_train_dataset(args)
    sample = ds[0]
    cond_ch = int(sample["cond"].shape[0])
    out_ch = int(sample["fine"].shape[0])

    if name == "bicubic":
        torch.save({"name": name, "cond_ch": cond_ch, "out_ch": out_ch}, ckpt_path)
        Path(os.path.join(out, "DONE")).touch()
        print("[bicubic] no training required")
        return

    dl = DataLoader(ds, batch_size=args.batch, shuffle=True, num_workers=0,
                    drop_last=True)
    model = build_baseline(name, cond_ch, out_ch, args.base, args.ddpm_T).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    disc = opt_d = None
    if name == "srgan":
        disc = Discriminator(out_ch).to(device)
        opt_d = torch.optim.AdamW(disc.parameters(), lr=args.lr)
    bce = nn.BCEWithLogitsLoss()

    start_epoch = 0
    if args.resume and os.path.exists(ckpt_path):
        ck = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ck["model"])
        if "opt" in ck:
            opt.load_state_dict(ck["opt"])
        if name == "srgan" and "disc" in ck:
            disc.load_state_dict(ck["disc"])
            opt_d.load_state_dict(ck["opt_d"])
        start_epoch = int(ck.get("epoch", -1)) + 1
        print(f"[resume] {ckpt_path}: continuing at epoch {start_epoch}", flush=True)

    print(f"[{name}] train {out}  cond_ch={cond_ch} out_ch={out_ch} "
          f"base={args.base} epochs={args.epochs} batch={args.batch}", flush=True)

    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()
        model.train()
        last = 0.0
        for batch in dl:
            fine = batch["fine"].to(device)
            cond = batch["cond"].to(device)
            if name == "srcnn":
                loss = F.mse_loss(model(cond), fine)
                opt.zero_grad(); loss.backward(); opt.step()
            elif name == "srgan":
                fake = model(cond)
                d_real, d_fake = disc(fine), disc(fake.detach())
                loss_d = bce(d_real, torch.ones_like(d_real)) + \
                    bce(d_fake, torch.zeros_like(d_fake))
                opt_d.zero_grad(); loss_d.backward(); opt_d.step()
                adv = bce(disc(fake), torch.ones_like(d_fake))
                loss = F.mse_loss(fake, fine) + args.adv_weight * adv
                opt.zero_grad(); loss.backward(); opt.step()
            else:  # ddpm / corrdiff
                loss = model.training_loss(fine, cond)
                opt.zero_grad(); loss.backward(); opt.step()
            last = float(loss.detach())
        ck = {"model": model.state_dict(), "opt": opt.state_dict(), "name": name,
              "cond_ch": cond_ch, "out_ch": out_ch, "base": args.base,
              "ddpm_T": args.ddpm_T, "batch": args.batch, "epoch": epoch}
        if name == "srgan":
            ck["disc"] = disc.state_dict()
            ck["opt_d"] = opt_d.state_dict()
        torch.save(ck, ckpt_path)
        print(f"[{name}] epoch {epoch} loss {last:.4f} {time.time() - t0:.1f}s",
              flush=True)

    Path(os.path.join(out, "DONE")).touch()
    print(f"[{name}] training complete -> {ckpt_path} (DONE)")


# -------------------------------------------------------------- evaluation ---
def evaluate_baseline(args):
    from pcrf.evaluate import build_eval_dataset      # identical test split
    device = args.device
    ds = build_eval_dataset(args)
    sample = ds[0]
    cond_ch = int(sample["cond"].shape[0])
    out_ch = int(sample["fine"].shape[0])
    spec = ChannelSpec(precip=0, u=1, v=2) if out_ch >= 3 else ChannelSpec(precip=0)
    dl = DataLoader(ds, batch_size=args.batch, shuffle=False)

    model = None
    if args.baseline != "bicubic":
        model = build_baseline(args.baseline, cond_ch, out_ch, args.base,
                               args.ddpm_T).to(device)
        ckpt = args.baseline_ckpt or os.path.join(
            args.out or f"runs/baseline_{args.baseline}", "model.pt")
        state = torch.load(ckpt, map_location=device)
        model.load_state_dict(state["model"])
        model.eval()

    n = args.ensemble if args.baseline in STOCHASTIC else 1
    agg, nb = {}, 0
    for batch in dl:
        fine = batch["fine"].to(device)
        cmass = batch["coarse_mass"].to(device)
        members = predict_members(args.baseline, model, batch, n, args.steps,
                                  args.eta, device, out_ch)
        pred = members.mean(0)
        m = all_metrics(pred, fine, cmass, spec)       # no PC-RF projection
        m["CRPS"] = crps_ensemble(members, fine)
        for k, v in m.items():
            agg[k] = agg.get(k, 0.0) + v
        nb += 1

    means = {k: round(agg[k] / nb, 6) for k in agg}
    label = LABEL[args.baseline]
    print(f"\n==== baseline {label} ({args.data}, {nb} batches, members={n}) ====")
    for k in ["PSNR", "SSIM", "SAM", "CRPS", "DE", "NVR", "MCE"]:
        if k in means:
            print(f"  {k:5s}: {means[k]:.4f}")
    print("==================================================\n")

    if args.json_out:
        key = args.json_dataset or {"sentinel2": "s2"}.get(args.data, args.data)
        path = Path(args.json_out)
        blob = json.loads(path.read_text()) if path.exists() else {}
        blob.setdefault(key, {})[label] = means
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(blob, indent=2, sort_keys=True))
        print(f"[json] recorded {key}/{label} -> {path}")
    return means


# --------------------------------------------------------------------- CLI ---
def build_argparser():
    ap = argparse.ArgumentParser(description="PC-RF Table-2 baselines")
    ap.add_argument("--mode", choices=["train", "eval"], default="train")
    ap.add_argument("--baseline", required=True, choices=METHODS)
    ap.add_argument("--data", default="era5",
                    choices=["synthetic", "era5", "sentinel2"])
    ap.add_argument("--root", default="")
    ap.add_argument("--coarse", default="")
    ap.add_argument("--target", default="")
    ap.add_argument("--scenes", default="")
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--fine", type=int, default=128)
    ap.add_argument("--factor", type=int, default=4)
    ap.add_argument("--sat_bands", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--base", type=int, default=64)
    ap.add_argument("--ddpm_T", type=int, default=1000)
    ap.add_argument("--steps", type=int, default=100, help="DDIM sampling steps")
    ap.add_argument("--eta", type=float, default=1.0, help="DDIM stochasticity")
    ap.add_argument("--ensemble", type=int, default=8, help="CRPS members (stochastic)")
    ap.add_argument("--adv_weight", type=float, default=1e-3)
    ap.add_argument("--out", default="")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--baseline_ckpt", default="")
    ap.add_argument("--json_out", default="")
    ap.add_argument("--json_method", default="")   # accepted, label is canonical
    ap.add_argument("--json_dataset", default="")
    ap.add_argument("--device",
                    default="cuda" if torch.cuda.is_available() else "cpu")
    return ap


def main(argv=None):
    args = build_argparser().parse_args(argv)
    if args.mode == "train":
        train_baseline(args.baseline, args)
    else:
        evaluate_baseline(args)


if __name__ == "__main__":
    main()
