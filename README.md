# PC-RF: training, export and serving code

PC-RF (Physics-Constrained Rectified Flow) is a conditional rectified-flow model for climate downscaling. It maps coarse ERA5 precipitation and 10 m wind to a finer precipitation field. Training adds three soft physics penalties. Sampling ends with a projection step that clamps precipitation at zero and rescales it to the coarse domain mean.

This repository holds the code behind the talk [From PyTorch to Production: Serving a Physics-Constrained Generative Model with ONNX, Ray, and vLLM](https://www.youtube.com/watch?v=k1FH2Pu953w) at PyTorch Conference North America 2026. It covers the model, training on one GPU and with Ray Train, evaluation, ONNX and AOTInductor export, a Ray Serve downscaler, and a latency benchmark.

## Read this first

The code is published as it was presented. It has known problems that affect how far you can trust its outputs. [KNOWN_ISSUES.md](KNOWN_ISSUES.md) lists each one and says where it lives. The five that matter most are these:

1. **The divergence penalty and the DE metric are wrong on ERA5.** ERA5 files store latitude from north to south, and the code never sorts latitude. The operator therefore computes du/dx minus dv/dy, which is not the divergence.
2. **The projection can fail silently.** If sampling leaves no positive precipitation, or only values so small that the clamped mean is below 1e-8, the field does not reach the coarse mean.
3. **The ERA5-Land target is interpolated ERA5 precipitation.** It has no spatial detail finer than the ERA5 forcing it came from, and it covers a different accumulation window than the input. Scores against it say little about downscaling skill.
4. **Several metrics pool three channels, and two of them are copies of the input wind.** SSIM and PSNR are also scored on the ensemble mean.
5. **There is no seed control, and no trained checkpoints are included.** The quality metrics shown in the talk cannot be reproduced from this repository.

## What is here

```
src/pcrf/            The model package
  physics.py           Penalties, finite differences and the inference projection
  model.py             Conditional U-Net velocity field
  flow.py              Rectified-flow loss and Euler sampling
  data.py              Synthetic, ERA5 and Sentinel-2 datasets
  metrics.py           PSNR, SSIM, CRPS, DE, NVR, MCE
  train.py             Single-device training with checkpoint resume
  evaluate.py          Sampling and metrics
  baselines.py         Bicubic, SRCNN, SRGAN, DDPM and CorrDiff-style baselines
  download_data.py     ERA5 and ERA5-Land download through the CDS API
  s2_ingest.py         Sentinel-2 tile ingest (not used by any ERA5 run)
  serving/math.py      Sampler and projection for exported models
apps/training/       Export, Ray Train, Ray Serve and benchmark scripts
slurm/               Template Slurm scripts for single-GPU and multi-GPU runs
experiments/         A NumPy-only synthetic mechanism check
tests/               The test suite
```

## Quick start on a CPU

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,train,data]"

pcrf-smoke                                    # end-to-end check on synthetic data
pcrf-train --data synthetic --epochs 2 --batch 8 --fine 64 --base 32 --out runs/demo
pcrf-evaluate --ckpt runs/demo/pcrf_last.pt --data synthetic --fine 64 --steps 20 --ensemble 8
python -m pytest -q
```

The synthetic dataset exists only to exercise the code. Its numbers say nothing about real data.

## Train on ERA5

1. Install the data extras with `pip install -e ".[data]"`. Create a free CDS account, put the key in `~/.cdsapirc`, and accept the ERA5 and ERA5-Land licences on the CDS website.
2. Read the data warnings in [KNOWN_ISSUES.md](KNOWN_ISSUES.md), then download with `bash apps/training/download_era5.sh`. It writes `data/era5_<year>/coarse.nc` and `target.nc` for Western Europe.
3. Train and evaluate:

```bash
pcrf-train --data era5 --coarse data/era5_2020/coarse.nc --target data/era5_2020/target.nc \
    --fine 128 --epochs 300 --batch 32 --base 64 \
    --l_div 1.0 --l_nn 1.0 --l_mass 1.0 --device cuda --out runs/pcrf_full --resume
pcrf-evaluate --ckpt runs/pcrf_full/pcrf_last.pt --data era5 \
    --coarse data/era5_2020/coarse.nc --target data/era5_2020/target.nc \
    --fine 128 --steps 20 --ensemble 8
```

This command assumes one NVIDIA GPU. On a machine without one, replace `--device cuda` with `--device cpu` and lower `--epochs` a lot, because training will be very slow.

For an ablation, pass `--l_div 0`, `--l_nn 0` or `--l_mass 0`. Pass `--no_physics` for the unconstrained flow.

`slurm/train_single_gpu.sbatch` runs the same command under Slurm. Training saves a checkpoint every epoch, so after a wall-time stop you submit the script again and `--resume` continues from the last checkpoint. `slurm/train_ray_multi_gpu.sbatch` runs data-parallel training with Ray Train on several GPUs of one node. Both are templates, so edit the account, partition and paths first.

## Export and serve

```bash
pip install -e ".[export,serve]"
python apps/training/export_onnx.py --epochs 2 --fine 64 --base 32 --out artifacts/pcrf.onnx
python apps/training/export_aoti.py --help     # PyTorch-native AOTInductor export
```

`export_onnx.py` and `export_aoti.py` train their own small network on synthetic data and export that network. It has two levels and no attention unless you pass `--attn`. These scripts cannot export a checkpoint written by `pcrf-train`, which has three levels and attention at the coarsest level. The ONNX graph holds only the velocity field. The Euler loop and the projection run on the host in `pcrf.serving.math`.

```bash
# Ray Serve (needs the ray extra). It runs in the foreground, so send requests from a second terminal.
pip install -e ".[ray]"
PCRF_ONNX=artifacts/pcrf.onnx serve run apps.training.serve_ray:app

# Latency across eager, torch.compile, AOTInductor and ONNX Runtime.
# --tiny benchmarks a small two-level network. Omit it to benchmark the trained architecture.
python apps/training/bench_serving.py --tiny --out artifacts/bench.json
```

The latency numbers shown in the talk used the trained architecture with random weights. That is valid for latency and says nothing about output quality.

## Tests

The suite runs on a CPU and uses small synthetic arrays. It skips tests whose optional packages are missing (`onnx`, `onnxruntime`, `ray`, `xarray`, `netCDF4`, `rasterio`, `pandas`, `matplotlib`), so installing every extra runs the most tests.

```bash
pip install -e ".[dev,data,train,export,ray,serve]"
python -m pytest -q
```

The tests check that the code runs and that the projection behaves on random fields. They do not cover the all-clamped case described in KNOWN_ISSUES.md, and they do not validate the science.

## What is not here

This repository leaves out the trained checkpoints, the ERA5 data, the result files, the paper, the talk slides, the deployment and Slurm run scripts, and the Rust inference service, Go gateway and vLLM agent that the talk also showed. Some comments and test docstrings still name those files. The scripts `make_tables.py` and `merge_baseline_metrics.py` read `results/*.json` files that are not included, so they have no input until you produce your own.

## Credits and data

The model uses rectified flow (Liu, Gong and Liu, ICLR 2023). The baselines in `baselines.py` are reimplementations of SRCNN (Dong et al., 2015), SRGAN (Ledig et al., CVPR 2017), a DDPM (Ho et al., NeurIPS 2020) sampled with DDIM (Song et al., ICLR 2021), and a CorrDiff-style residual diffusion model (Mardani et al., 2025). The projection follows Hess et al. (Nature Machine Intelligence, 2022) and Harder et al. (JMLR, 2023), as KNOWN_ISSUES.md explains. The data come from ERA5 (Hersbach et al., 2020) and ERA5-Land (Muñoz-Sabater et al., 2021), through the Copernicus Climate Data Store. Check the CDS licence terms before you redistribute any data.

## License

MIT. See [LICENSE](LICENSE).
