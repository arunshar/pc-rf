# Experiments

## `synthetic_sanity_check.py`

This script is a standalone check that needs only NumPy and Matplotlib, with no torch, no GPU and no downloads. It builds a toy divergence-free wind field and a Gaussian-blob precipitation field, corrupts them, and applies the physics penalties and the projection.

```bash
cd experiments && python3 synthetic_sanity_check.py
```

It writes `results/synthetic_metrics.md` and `results/synthetic_physics_demo.png` in the directory where you run it. Install Matplotlib with `pip install -e ".[train]"`.

## What this check shows

The check shows that the penalty gradients and the projection written in this script behave as intended on one fixed field. The script does not import `pcrf`, so it does not test `src/pcrf/physics.py`. After the projection, NVR is zero and MCE is zero to within floating-point rounding.

The "PC-RF" row is not a trained network. It is the corrupted field refined by gradient descent on the penalties and then projected. The improvement in RMSE, correlation and SSIM therefore cannot be split between the penalties and the projection. These numbers come from toy fields, so they say nothing about ERA5 or about trained models.
