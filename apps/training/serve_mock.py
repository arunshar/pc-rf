"""
FastAPI inference backend that runs the PyTorch sampler.

It runs the PyTorch RectifiedFlow sampler directly. Use it for development and as
a reference for the request and response contract in serve_schema.py. The Rust
inference service from the talk is not part of this repository.

  uvicorn serve_mock:app --app-dir apps/training --port 50080

PCRF_CKPT, if set, must point to a bare state_dict for the small network built
below (base=32, two levels, no attention). A checkpoint written by pcrf-train does
not fit it.
"""
from __future__ import annotations
import os
import time
from pathlib import Path

import torch
from fastapi import FastAPI
from pydantic import BaseModel

from pcrf import data as d
from pcrf import flow as f
from pcrf import metrics as me
from pcrf import model as m
from pcrf import physics as p

SPEC = p.ChannelSpec(precip=0, u=1, v=2)
_probe = d.SyntheticDownscaleDataset(n=1, fine=64, factor=4, sat_bands=4)[0]
IN_CH = _probe["fine"].shape[0]
COND_CH = _probe["cond"].shape[0]

_net = m.UNet(in_ch=IN_CH, cond_ch=COND_CH, base=32, ch_mults=(1, 2), attn_at=())
_rf = f.RectifiedFlow(_net, p.PhysicsLoss(SPEC))
_ckpt = os.environ.get("PCRF_CKPT")
if _ckpt and Path(_ckpt).exists():
    _net.load_state_dict(torch.load(_ckpt, map_location="cpu"))
    print(f"[serve_mock] loaded checkpoint {_ckpt}")
_net.eval()

app = FastAPI(title="PC-RF mock sampler", version="0.1.0")


class Req(BaseModel):
    fine: int = 64
    base: int = 16
    steps: int = 20
    ensemble: int = 8
    project: bool = True
    seed: int = 0


@app.get("/health")
def health():
    return {"ready": True, "model": "pcrf-synthetic", "backend": "torch-cpu"}


@app.post("/downscale")
def downscale(r: Req):
    factor = max(r.fine // max(r.base, 1), 1)
    ds = d.SyntheticDownscaleDataset(n=1, fine=r.fine, factor=factor,
                                     sat_bands=4, seed=r.seed)
    s = ds[0]
    cond_b = s["cond"][None].expand(r.ensemble, -1, -1, -1)
    cmass_b = s["coarse_mass"][None].expand(r.ensemble)
    shape = (r.ensemble, IN_CH, r.fine, r.fine)

    t0 = time.time()
    with torch.no_grad():
        x = _rf.sample(shape, cond_b, steps=r.steps, device="cpu",
                       spec=SPEC, coarse_mass=cmass_b, project=r.project)
    mean = x.mean(0, keepdim=True)
    mts = me.all_metrics(mean, s["fine"][None], s["coarse_mass"][None], SPEC)
    return {
        "channels": IN_CH, "height": r.fine, "width": r.fine,
        "metrics": {
            "divergence_error": float(mts.get("DE", 0.0)),
            "nonneg_violation": float(mts.get("NVR", 0.0)),
            "mass_cons_error": float(mts.get("MCE", 0.0)),
        },
        "latency_ms": (time.time() - t0) * 1e3,
        "backend": "torch-cpu",
    }
