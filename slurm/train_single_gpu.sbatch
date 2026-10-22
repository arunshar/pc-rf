#!/bin/bash -l
# ============================================================================
#  PC-RF training on one GPU under Slurm, resumable across the wall-time limit.
#  This file is a template. Edit the #SBATCH lines, DATA_ROOT, OUT and the
#  environment activation line below for your cluster.
#
#    mkdir -p logs && sbatch slurm/train_single_gpu.sbatch
#
#  Slurm opens the --output and --error files before this script starts, so
#  the logs/ directory must exist at submit time.
#
#  Training writes a checkpoint every epoch and a DONE file when it finishes.
#  If the wall time ends first, submit the script again. The --resume flag
#  continues from the last checkpoint.
# ============================================================================
#SBATCH --job-name=pcrf-era5
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --partition=YOUR_GPU_PARTITION
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64gb
#SBATCH --time=24:00:00
#SBATCH --output=logs/pcrf-%j.out
#SBATCH --error=logs/pcrf-%j.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

DATA_ROOT=${DATA_ROOT:-$SLURM_SUBMIT_DIR/data}
OUT=${OUT:-$SLURM_SUBMIT_DIR/runs/pcrf_full}
mkdir -p "$OUT"

# One year: point at the files directly. Several years: use the glob
# "$DATA_ROOT/era5_*/coarse.nc" (the loader opens it with open_mfdataset).
COARSE="$DATA_ROOT/era5_2020/coarse.nc"
TARGET="$DATA_ROOT/era5_2020/target.nc"

echo "host: $(hostname)  job: ${SLURM_JOB_ID:-NA}"
nvidia-smi || true

# Activate the environment where `pip install -e ".[data,train]"` was run.
# source /path/to/venv/bin/activate

srun python -m pcrf.train --data era5 \
     --coarse "$COARSE" --target "$TARGET" \
     --fine 128 --epochs 300 --batch 32 --base 64 \
     --l_div 1.0 --l_nn 1.0 --l_mass 1.0 \
     --device cuda --out "$OUT" --resume

if [ -f "$OUT/DONE" ]; then
  echo "training complete -> $OUT/pcrf_last.pt"
else
  echo "training did not finish. Submit the script again to resume from the last checkpoint."
fi
