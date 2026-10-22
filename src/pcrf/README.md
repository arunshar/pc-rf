# pcrf package

The model, training, evaluation and serving math for PC-RF. See the [repository README](../../README.md) for how to run it and [KNOWN_ISSUES.md](../../KNOWN_ISSUES.md) for what is wrong with it.

| File | Purpose |
|---|---|
| `physics.py` | The three penalties, the finite-difference operators and the inference projection. |
| `model.py` | The conditional U-Net velocity field. |
| `flow.py` | The rectified-flow loss and Euler sampling. |
| `data.py` | The synthetic, ERA5 and Sentinel-2 datasets. |
| `metrics.py` | PSNR, SSIM, CRPS, DE, NVR and MCE. |
| `train.py` | Training entry point (`pcrf-train`). |
| `evaluate.py` | Sampling and metrics (`pcrf-evaluate`). |
| `baselines.py` | The reimplemented baselines. |
| `download_data.py` | ERA5 and ERA5-Land download through the CDS API. |
| `s2_ingest.py` | Sentinel-2 tile ingest. No ERA5 run used it. |
| `serving/math.py` | The sampler and projection for exported models. |
| `smoke_test.py` | An end-to-end check on synthetic data (`pcrf-smoke`). |

Some docstrings in these files refer to a paper, a Sentinel-2 encoder and FiLM fusion of land-surface features. They describe a design that the training code never implemented, and KNOWN_ISSUES.md explains what the trained network does.
