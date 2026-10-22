# apps/training

This directory holds export, serving and scale-out scripts that build on the `pcrf` package in `src/pcrf`. Read [../../KNOWN_ISSUES.md](../../KNOWN_ISSUES.md) before using any output.

| File | Role |
|---|---|
| `export_onnx.py` | Trains its own small two-level model on synthetic data, exports the velocity field to ONNX, and checks the ONNX output against PyTorch. |
| `export_aoti.py` | Exports with `torch.export` and compiles an AOTInductor `.pt2` that runs without ONNX. |
| `serve_mock.py` | A FastAPI sampler that runs the PyTorch model. |
| `serve_ray.py`, `serve_schema.py`, `serve_ingress.py` | A Ray Serve downscaler with optional dynamic batching and fractional-GPU replicas. |
| `bench_serving.py` | Latency and throughput across eager, `torch.compile`, AOTInductor and ONNX Runtime, plus a parity check. |
| `train_ray.py` | Data-parallel training with Ray Train, with checkpoint resume and per-epoch throughput. |
| `tune_pcrf.py` | A Ray Tune search over the three penalty weights and the learning rate. |
| `collect_scaling.py`, `make_scaling_plots.py` | Collect throughput from scaling runs and plot them. |
| `make_tables.py`, `make_figure3.py`, `visualize.py` | Build result tables and figures from saved metrics and checkpoints. |
| `merge_baseline_metrics.py` | Merge per-baseline metric files. It has a known overwrite bug. |
| `e2e_check.py` | A standard-library parser that checks the training loss fell during a short Ray Train run. The Slurm job that launches that run is not included. |
| `download_era5.sh` | Download ERA5 and ERA5-Land for Western Europe. |
| `Dockerfile`, `Dockerfile.serve` | Images for the FastAPI sampler and the Ray Serve app. |

The exported ONNX graph holds only the velocity field. The Euler loop and the physics projection run on the host in `pcrf.serving.math`.

`--attn` adds self-attention at the coarsest level of the export network. The attention block in the middle of the network is always exported. The flag is off by default to keep the demo graph small. The default export is a two-level network, and a checkpoint written by `pcrf-train` does not fit it.
