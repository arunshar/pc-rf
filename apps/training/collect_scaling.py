#!/usr/bin/env python
"""Collect a PC-RF scaling sweep into the two CSVs make_scaling_plots.py expects.

train_ray.py reports per-epoch metrics through Ray Train, which writes them as
newline-delimited JSON to  <runs>/pcrf-w<W>/TorchTrainer_*/result.json  (one JSON
object per report() call). This walks those files for each worker count W and
emits:

  scaling_results.csv : workers,samples_per_sec,epoch_time_s,gpu_cost_per_hr,val_L_total
  scaling_curves.csv  : workers,epoch,L_total

Throughput / epoch-time per W are the MEDIAN over the post-warmup epochs (epoch 0
dropped), which is the honest steady-state number, not a lucky best epoch.
val_L_total is the final reported L_total (for synthetic there is no held-out
split, so it is the final training L_total; label it as such in the talk).

Pure stdlib so it runs in any env that has Python; no pandas needed here.

  python collect_scaling.py --runs scale_runs \
         --workers 1 2 4 8 --gpu-cost-per-hr 3.67 \
         --out-results scaling_results.csv --out-curves scaling_curves.csv

Optional comparison tags (opt-in, both default to empty so the CSV schema is
unchanged when you do not pass them). These let make_scaling_plots.py draw the
new overlay figures (bf16-vs-fp32 and intra-node-vs-inter-node):

  --precision bf16      tag every collected row with a precision label
  --topology  intra     tag every collected row with a topology/platform label

A typical comparison workflow runs this twice into two CSVs (one per precision
or topology) and then passes both to make_scaling_plots.py, OR tags each run and
concatenates the two CSVs into one before plotting. Either layout is supported
by the overlay figures downstream.

WARNING: a tagged single CSV (one carrying a 'precision' or 'topology' column),
or two such CSVs concatenated into one, is a MULTI-CONDITION sweep. Feed it to
make_scaling_plots.py via --scaling-fp32 or the overlay options, NOT as the
primary --scaling: the core throughput/efficiency/cost figures assume one row
per worker count and would zig-zag across the interleaved conditions. The
plotter prints a loud WARN if it detects this, but the clean pattern is to keep
the untagged single-condition CSV for --scaling and reserve tagged CSVs for the
overlays.
"""
from __future__ import annotations
import argparse
import csv
import glob
import json
import os
import statistics
import sys

LOSS_KEY = "L_total"          # emitted by pcrf.flow training_loss


def _latest_result_json(runs_root: str, w: int) -> str | None:
    """Newest result.json under <runs>/pcrf-w<W>/ (the Ray Train trial dir)."""
    pats = [
        os.path.join(runs_root, f"pcrf-w{w}", "**", "result.json"),
        os.path.join(runs_root, f"pcrf-w{w}", "result.json"),
    ]
    hits: list[str] = []
    for p in pats:
        hits.extend(glob.glob(p, recursive=True))
    if not hits:
        return None
    return max(hits, key=os.path.getmtime)


def _read_epochs(result_json: str) -> list[dict]:
    """Parse the JSONL result file into one dict per reported epoch."""
    recs: list[dict] = []
    with open(result_json) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                recs.append(json.loads(line))
            except json.JSONDecodeError:
                pass  # tolerate a half-written trailing line
    return recs


def _epoch_idx(rec: dict, fallback: int) -> int:
    if "epoch" in rec:
        return int(rec["epoch"])
    if "training_iteration" in rec:
        return int(rec["training_iteration"]) - 1
    return fallback


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True, help="root holding pcrf-w<W> run dirs")
    ap.add_argument("--workers", type=int, nargs="+", required=True)
    ap.add_argument("--gpu-cost-per-hr", type=float, default=0.0,
                    help="per-GPU reference price for the cost-per-epoch curve")
    ap.add_argument("--out-results", required=True)
    ap.add_argument("--out-curves", required=True)
    # --- optional comparison tags (opt-in; empty = column omitted) -----------
    ap.add_argument("--precision", default="",
                    help="precision label for these runs (e.g. bf16, fp32). "
                         "Adds a 'precision' column so make_scaling_plots.py can "
                         "draw the bf16-vs-fp32 overlay. Empty = column omitted. "
                         "A tagged CSV is for the overlay, not for --scaling.")
    ap.add_argument("--topology", default="",
                    help="topology/platform label for these runs (e.g. intra, "
                         "inter, nvlink-single-node, multi-node-efa). Adds a 'topology' "
                         "column for the intra-vs-inter efficiency figure. "
                         "Empty = column omitted. A tagged CSV is for the "
                         "overlay, not for --scaling.")
    args = ap.parse_args()

    # Only widen the schema when a tag is actually provided, so the default
    # invocation produces the exact same CSV columns as before.
    extra_cols: list[str] = []
    if args.precision:
        extra_cols.append("precision")
    if args.topology:
        extra_cols.append("topology")

    result_rows: list[dict] = []
    curve_rows: list[dict] = []
    missing: list[int] = []

    for w in sorted(args.workers):
        rj = _latest_result_json(args.runs, w)
        if not rj:
            print(f"[collect] WARN: no result.json for W={w} under {args.runs}",
                  file=sys.stderr)
            missing.append(w)
            continue
        recs = _read_epochs(rj)
        if not recs:
            print(f"[collect] WARN: empty result.json for W={w}: {rj}",
                  file=sys.stderr)
            missing.append(w)
            continue

        sps, ep_t, losses = [], [], []
        for i, rec in enumerate(recs):
            e = _epoch_idx(rec, i)
            if LOSS_KEY in rec:
                curve_rows.append({"workers": w, "epoch": e,
                                   "L_total": float(rec[LOSS_KEY])})
                losses.append((e, float(rec[LOSS_KEY])))
            if "samples_per_sec" in rec:
                sps.append((e, float(rec["samples_per_sec"])))
            if "epoch_time_s" in rec:
                ep_t.append((e, float(rec["epoch_time_s"])))

        # steady state = drop the warmup epoch 0 when there is more than one epoch
        def steady(pairs):
            vals = [v for e, v in pairs if e > 0] or [v for _, v in pairs]
            return statistics.median(vals) if vals else float("nan")

        val_loss = losses[-1][1] if losses else float("nan")
        row = {
            "workers": w,
            "samples_per_sec": round(steady(sps), 3),
            "epoch_time_s": round(steady(ep_t), 4),
            "gpu_cost_per_hr": args.gpu_cost_per_hr,
            "val_L_total": round(val_loss, 6),
        }
        if args.precision:
            row["precision"] = args.precision
        if args.topology:
            row["topology"] = args.topology
        result_rows.append(row)
        print(f"[collect] W={w}: samples/s={result_rows[-1]['samples_per_sec']} "
              f"epoch_s={result_rows[-1]['epoch_time_s']} "
              f"final L_total={result_rows[-1]['val_L_total']}  ({len(recs)} epochs)")

    if not result_rows:
        print("[collect] ERROR: no runs collected; nothing written.", file=sys.stderr)
        return 1

    os.makedirs(os.path.dirname(os.path.abspath(args.out_results)), exist_ok=True)
    with open(args.out_results, "w", newline="") as fh:
        wri = csv.DictWriter(fh, fieldnames=["workers", "samples_per_sec",
                                             "epoch_time_s", "gpu_cost_per_hr",
                                             "val_L_total"] + extra_cols)
        wri.writeheader()
        wri.writerows(result_rows)
    with open(args.out_curves, "w", newline="") as fh:
        wri = csv.DictWriter(fh, fieldnames=["workers", "epoch", "L_total"])
        wri.writeheader()
        wri.writerows(curve_rows)

    print(f"[collect] wrote {args.out_results} ({len(result_rows)} rows) and "
          f"{args.out_curves} ({len(curve_rows)} rows)")
    if extra_cols:
        print(f"[collect] tagged columns: {extra_cols} "
              f"(precision={args.precision!r}, topology={args.topology!r}). "
              "NOTE: a tagged CSV is a multi-condition sweep; pass it to the "
              "overlay options, not to --scaling.")
    if missing:
        print(f"[collect] NOTE: missing/empty W = {missing} (not in CSVs)",
              file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
