"""
Merge the per-method baseline metrics JSONs (results/era5_baseline_<method>.json,
written by pcrf.baselines --mode eval on parallel array tasks) into the shared
results/metrics.json, without disturbing the existing PC-RF / RF-base / ablation
entries. Stdlib only, so it runs on a login node (no container needed).

    python3 apps/training/merge_baseline_metrics.py
    python3 apps/training/make_tables.py --metrics results/metrics.json --out-dir results
"""
from __future__ import annotations
import glob
import json
from pathlib import Path

METRICS = Path("results/metrics.json")


def main():
    blob = json.loads(METRICS.read_text()) if METRICS.exists() else {}
    frags = sorted(glob.glob("results/era5_baseline_*.json"))
    if not frags:
        print("no results/era5_baseline_*.json fragments found")
    for f in frags:
        d = json.loads(Path(f).read_text())
        for dataset, methods in d.items():
            blob.setdefault(dataset, {}).update(methods)
            print(f"merged {f}: {dataset} <- {list(methods)}")
    METRICS.parent.mkdir(parents=True, exist_ok=True)
    METRICS.write_text(json.dumps(blob, indent=2, sort_keys=True))
    print(f"wrote {METRICS}")


if __name__ == "__main__":
    main()
