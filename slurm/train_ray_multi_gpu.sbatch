#!/bin/bash -l
# ============================================================================
#  PC-RF data-parallel training with Ray Train on several GPUs of ONE node.
#  This file is a template. Edit the #SBATCH lines, DATA_ROOT and the
#  environment activation line below for your cluster.
#
#    mkdir -p logs && sbatch slurm/train_ray_multi_gpu.sbatch
#
#  Slurm opens the --output and --error files before this script starts, so
#  the logs/ directory must exist at submit time.
#
#  train_ray.py starts a local Ray cluster inside the job. Use --workers equal
#  to the number of GPUs you request. See the module docstring in
#  apps/training/train_ray.py for --ingest, --amp and --grad-accum.
# ============================================================================
#SBATCH --job-name=pcrf-ray
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --partition=YOUR_GPU_PARTITION
#SBATCH --gres=gpu:4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128gb
#SBATCH --time=12:00:00
#SBATCH --output=logs/pcrf-ray-%j.out
#SBATCH --error=logs/pcrf-ray-%j.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

WORKERS=${WORKERS:-4}
DATA_ROOT=${DATA_ROOT:-$SLURM_SUBMIT_DIR/data}

# source /path/to/venv/bin/activate     # environment with `pip install -e ".[data,ray]"`

python apps/training/train_ray.py --data era5 --gpu --workers "$WORKERS" \
    --coarse "$DATA_ROOT/era5_*/coarse.nc" --target "$DATA_ROOT/era5_*/target.nc" \
    --epochs 100 --storage "$SLURM_SUBMIT_DIR/ray_results" --name pcrf-ray
