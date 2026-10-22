# Known issues

This code is published as it was presented in the talk. The talk already disclosed some of the problems below, including the latitude sign error in the divergence operator. The rest were found afterward. Each item says where it lives so you can check it yourself. Treat any number this code produces on real ERA5 data as unvalidated until you have addressed the first four items.

## The divergence penalty and DE are wrong on ERA5

`ddx`, `ddy` and `divergence` in `src/pcrf/physics.py` (lines 22 to 36) take central differences along array rows and columns, and they assume row 0 is the southern edge. The CDS returns ERA5 with latitude decreasing along the rows. A one-month request for 40 to 50 degrees north returned latitude 50.0 in the first row and 40.0 in the last. The data loader in `src/pcrf/data.py` never sorts latitude. On real ERA5 data the penalty `l_div` and the metric `divergence_error` (`src/pcrf/metrics.py`, line 69) therefore compute du/dx minus dv/dy, which is a stretching-deformation term and not the divergence.

Two more things make DE hard to interpret even with the right sign. The two wind components are standardized separately (`data.py`, lines 230 and 231), and the derivatives use grid-index units with no metric terms. DE has no physical units. Every arm was trained once, so this repository cannot show whether a difference in DE between two runs is larger than run-to-run variation.

The penalty also acts on the total divergence of the generated wind. It does not act on the divergence the generator adds beyond the coarse input.

To fix it, sort latitude ascending at load time and use the physical grid spacing.

## The projection can fail silently

`project_physics` in `src/pcrf/physics.py` (line 102) clamps negative precipitation to zero and multiplies the field by `coarse_mass / mean`, with the mean floored at 1e-8 (line 114). It fails in four cases:

- **Every cell is clamped.** The field stays all zero, and MCE for that sample is 1.
- **The clamped mean is below 1e-8.** The floor then replaces the true mean, so the rescale is wrong. With one cell of 1e-6 in a 128 by 128 field, the output mean was far from the target.
- **The coarse mass is negative.** The rescale then makes the output negative, so non-negativity fails.
- **The input contains a NaN.** One NaN in the precipitation channel turns the whole channel into NaN.

In the normal case, with a non-negative coarse mass and a clamped mean of at least 1e-8, the output is non-negative and its domain mean matches the target.

The docstring in older versions called the clamp "softplus-equivalent". The code is a plain clamp at zero. One test feeds an all-negative field to the serving projection and checks only that the output is finite and non-negative. No test checks the domain mean in that case, and no test covers `project_physics` with an all-clamped field.

The serving version, `project_precip` in `src/pcrf/serving/math.py`, differs. With a coarse mass of zero or less it applies only the clamp, and the training-side function rescales to zero. Its MCE function returns 0.0 whenever the coarse mass is below 1e-8, which hides any error in that case.

## The ERA5-Land target is interpolated ERA5

ECMWF documents ERA5-Land total precipitation as a forcing field that is interpolated from ERA5 onto the finer grid, and not as a land-model output. The fine target therefore has no spatial detail finer than the ERA5 forcing it came from. Scores against it say little about whether a model downscales well.

Related data problems:

- **Accumulation windows differ.** An ERA5 value covers the hour ending at its timestamp. An ERA5-Land value accumulates from 00 UTC, so it covers 6, 12 or 18 hours at 06, 12 and 18 UTC and the previous 24 hours at 00 UTC. `download_data.py` and `data.py` never reconcile the two.
- **The ocean is zero-filled.** ERA5-Land has no ocean values, and the loader replaces missing values with zero (`data.py`, line 218 for the target and lines 215 to 217 for the coarse fields). In the author's download the ocean covered roughly 45 percent of the default box. Check the NaN fraction of your own `target.nc`.
- **The coarse mass covers land and ocean.** `coarse_mass` is the mean over the whole box (`data.py`, line 220), so the constraint and the target disagree wherever it rains over the sea.
- **The grids are resampled to a square 128 by 128 grid.** The target is resized bilinearly and the coarse fields bicubically (`data.py`, lines 235 to 244). The native grids are not integer multiples of each other, so the code cannot apply a per-coarse-cell constraint.
- **There is no fine wind target.** The wind channels of the target are the upsampled coarse wind (`data.py`, line 246).

## The metrics pool channels and score the ensemble mean

SSIM, PSNR and CRPS average over all three channels, and two of the three are the copied input wind. A model is rewarded or penalized on them for reproducing bicubic interpolation. No skill metric is reported for precipitation alone. PSNR takes its data range from the whole target across channels (`src/pcrf/metrics.py`, lines 22 and 40). NVR and MCE do look at the precipitation channel by itself, and the projection sets both to zero in the normal case.

`src/pcrf/evaluate.py` (line 102) scores SSIM, PSNR, DE, NVR and MCE on the ensemble mean. CRPS is the fair estimator over the members (`metrics.py`, line 56). NVR counts the share of predicted cells below zero and does not use the target.

## Seeds and checkpoints

`train.py`, `evaluate.py` and `baselines.py` have no seed argument and never seed their random generators. Every reported ERA5 run was a single training run, and sampling noise differed from call to call. Differences of a few thousandths in SSIM cannot be separated from that noise. The trained checkpoints and the ERA5 files are not included, and the quality metrics in the talk cannot be regenerated bit for bit.

## Baselines

`src/pcrf/baselines.py` holds reimplementations of SRCNN, SRGAN, a DDPM and a CorrDiff-style model. They are not the authors' released models, they were not tuned, and they are scored without the projection. SRCNN and the SRResNet generator have no input-to-output skip connection, so they must learn the identity for the two wind channels that bicubic copies exactly. Do not read their scores as evidence about those methods.

## Docstrings that describe a design the code never implemented

Several docstrings and file headers mention a Sentinel-2 encoder, FiLM fusion of land-surface features, and sections of a paper. The trained network does not do that. `model.py` concatenates the conditioning with the noisy field. It applies one FiLM layer, driven by the time embedding, right after the input convolution, and the residual blocks add a projected time embedding. Self-attention runs at the coarsest U-Net level and in the middle block. `SatelliteEncoder` is imported in `train.py` and never used. The four Sentinel-2 channels that the default `--sat_bands 4` adds to the ERA5 conditioning are filled with zeros, so no satellite input reaches the model. The header of `data.py` has been corrected, but other docstrings still refer to the paper.

## Smaller items

- **The projection is prior art.** It is a global rescale of the kind Hess et al. (2022, Nature Machine Intelligence) used on GAN precipitation, with a rectifier in front. It is also the generalized multiplicative constraint layer of Harder et al. (JMLR 2023) with g equal to ReLU and one window over the whole domain. This repository does not claim it as new.
- **A merge script overwrites rows.** `apps/training/merge_baseline_metrics.py` (line 20) globs `results/era5_baseline_*.json`, which also matches the three-year files, for example `results/era5_baseline_srcnn_3yr.json`. Those files are not in this repository. Both kinds of file use the dataset key `era5` and the same method labels, and the three-year file sorts second, so running the script overwrites the one-year rows.
- **The synthetic check is not a trained model.** `experiments/synthetic_sanity_check.py` does not import `pcrf`. It re-implements the penalties and the projection in NumPy, so it does not test `src/pcrf/physics.py`. Its "PC-RF" row refines one fixed field by gradient descent on the penalties and then projects it, so the improvement it shows cannot be split between the penalties and the projection.
- **The export scripts do not export a trained checkpoint.** `export_onnx.py` and `export_aoti.py` build their own small network. See the README.

## What a fix should do

Sort latitude at load time. Compute divergence with physical spacing and winds in physical units. Add a fallback to the projection that sets the precipitation field to the uniform coarse mean when every cell is clamped, guard against a tiny clamped mean, a negative target and NaN, and test those cases. Score precipitation alone, per member, on a land mask. Reconcile the accumulation windows, or use a target that is not interpolated from the input. Train several seeds and report intervals.
