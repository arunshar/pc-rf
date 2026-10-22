"""
Turn the PC-RF scaling sweep into scaling figures: throughput, scaling
efficiency, cost-per-epoch, and convergence parity. Reads the CSVs produced by
collect_scaling.py and writes PNGs + a markdown summary.

  python apps/training/make_scaling_plots.py \
      --scaling scaling_results.csv \
      --curves  scaling_curves.csv \
      --out-dir artifacts/scaling

scaling_results.csv (one row per worker count):
  workers,samples_per_sec,epoch_time_s,gpu_cost_per_hr,val_L_total
scaling_curves.csv (optional, long; one row per worker count per epoch):
  workers,epoch,L_total

IMPORTANT: --scaling must be a SINGLE-condition sweep (one row per worker count,
strictly increasing workers). The core throughput / efficiency / cost figures
assume a single baseline at the smallest worker count. Do NOT pass a multi-
condition CSV (one that carries a 'topology', 'platform', or 'precision' column
with several groups, or that repeats worker counts) as --scaling: load() would
sort the interleaved rows by worker count and the core curves would zig-zag
between conditions. Multi-condition CSVs belong on the dedicated overlays below
(--scaling-fp32 / the 'precision' or 'topology' columns), which compute a fair
WITHIN-group baseline per condition. A loud WARN is printed if a multi-condition
CSV is detected on --scaling.

OPTIONAL comparison figures (all opt-in; the four core figures above are always
drawn and unchanged). A new figure is only produced when the column or CSV it
needs is present, so the existing pipeline is unaffected:

  (1) bf16-vs-fp32 throughput + efficiency overlay. Triggered by EITHER a
      'precision' column in --scaling with >=2 distinct values, OR a second CSV
      passed via --scaling-fp32 (the primary --scaling is then treated as bf16).
      If --scaling-fp32 is given but the path is missing, a WARN is printed and
      the overlay falls back to any in-file 'precision' column (or is skipped).

  (2) intra-node (NVLink) vs inter-node (multi-node) efficiency. Triggered
      by a 'topology' (or 'platform') column in --scaling with >=2 distinct
      values. Efficiency is computed WITHIN each group, relative to that group's
      own smallest worker count, so the comparison is apples-to-apples.

  (3) Ray Tune sweep figures: a tuning curve (best-so-far val metric vs trial)
      and a parallel-coordinates plot of {l_div,l_nn,l_mass,lr} vs the val
      metric. Triggered by --tune pointing at a Tune results CSV. Expected
      columns: trial index plus any of l_div,l_nn,l_mass,lr and a metric column
      (auto-detected: prefers val_L_total, then L_total, then --tune-metric).

  python apps/training/make_scaling_plots.py \
      --scaling scaling_bf16.csv --scaling-fp32 scaling_fp32.csv \
      --tune tune_results.csv --out-dir artifacts/scaling

Stdlib + matplotlib + pandas only (no seaborn).
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

# Grouping columns that, if present with >=2 distinct values, mark a CSV as a
# multi-condition sweep that should NOT be used as the primary --scaling input.
CONDITION_COLS = ("topology", "platform", "precision")


def detect_multicondition(df):
    """Return a human-readable reason if `df` looks like a multi-condition sweep.

    A multi-condition CSV (several topology / platform / precision groups, or
    repeated worker counts) must not feed the core single-baseline figures: it
    produces a zig-zag because load() sorts the interleaved rows by worker count.
    Returns None when the frame is a clean single-condition sweep.

    Pure function (no I/O) so it is unit-testable on a synthetic DataFrame.
    """
    for col in CONDITION_COLS:
        if col in df.columns and df[col].astype(str).str.len().gt(0).any():
            n = df[col].astype(str).nunique()
            if n >= 2:
                return f"{col!r} column has {n} distinct values"
    if "workers" in df.columns and df["workers"].duplicated().any():
        dups = sorted(set(df["workers"][df["workers"].duplicated()].tolist()))
        return f"worker count(s) {dups} repeat (rows per worker > 1)"
    return None


def load(scaling_csv):
    df = pd.read_csv(scaling_csv).sort_values("workers").reset_index(drop=True)
    reason = detect_multicondition(df)
    if reason is not None:
        print(
            "[scaling] WARN: --scaling CSV looks like a MULTI-CONDITION sweep "
            f"({reason}). The core throughput/efficiency/cost figures assume a "
            "single-condition sweep with one row per worker count; using a "
            "multi-condition CSV here makes them zig-zag between conditions. "
            "Pass the conditions through the overlay options instead "
            "(--scaling-fp32 or the 'precision'/'topology' columns).",
            file=sys.stderr,
        )
    base = df.loc[df["workers"].idxmin()]
    sps1, w1 = float(base["samples_per_sec"]), float(base["workers"])
    # scaling efficiency relative to the smallest worker count
    df["ideal_sps"] = sps1 * df["workers"] / w1
    df["efficiency"] = df["samples_per_sec"] / df["ideal_sps"]
    if "gpu_cost_per_hr" in df and "epoch_time_s" in df:
        df["cost_per_epoch_usd"] = (
            df["workers"] * df["gpu_cost_per_hr"] * df["epoch_time_s"] / 3600.0)
    return df


def _add_efficiency_within(g):
    """Per-group ideal_sps + efficiency relative to that group's smallest W.

    Used by the bf16-vs-fp32 and intra-vs-inter overlays so each series is
    measured against its OWN single-worker baseline (a fair within-condition
    efficiency), independent of the global load() baseline. The reported number
    is T_W / (W * T_1) expressed via throughput: samples_per_sec(W) divided by
    the per-group ideal (sps_1 * W / W_1).
    """
    g = g.sort_values("workers").reset_index(drop=True)
    base = g.loc[g["workers"].idxmin()]
    sps1, w1 = float(base["samples_per_sec"]), float(base["workers"])
    g = g.copy()
    g["ideal_sps"] = sps1 * g["workers"] / w1
    g["efficiency"] = g["samples_per_sec"] / g["ideal_sps"]
    return g


def plot_throughput(df, out):
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(df["workers"], df["ideal_sps"], "--", color="gray", label="ideal linear")
    ax.plot(df["workers"], df["samples_per_sec"], "o-", color="#1f77b4", label="PC-RF (measured)")
    ax.set_xlabel("GPU workers"); ax.set_ylabel("throughput (samples/s)")
    ax.set_title("Ray Train throughput scaling"); ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)


def plot_efficiency(df, out):
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.axhline(1.0, ls="--", color="gray", label="ideal (1.0)")
    ax.plot(df["workers"], df["efficiency"], "o-", color="#2ca02c")
    ax.set_xlabel("GPU workers"); ax.set_ylabel("scaling efficiency  T_W / (W*T_1)")
    # Dynamic upper bound so a marginally super-linear point (efficiency slightly
    # > 1, e.g. fixed per-epoch checkpoint I/O amortizing across workers on
    # intra-node NVLink) is not clipped at the ideal line.
    ax.set_ylim(0, max(1.1, float(df["efficiency"].max()) * 1.05))
    ax.set_title("Scaling efficiency"); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)


def plot_cost(df, out):
    if "cost_per_epoch_usd" not in df:
        return False
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(df["workers"], df["cost_per_epoch_usd"], "o-", color="#d62728")
    ax.set_xlabel("GPU workers"); ax.set_ylabel("cost per epoch (USD)")
    ax.set_title("Cost per epoch (you pay for the efficiency loss)"); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)
    return True


def plot_parity(curves_csv, out):
    if not curves_csv or not Path(curves_csv).exists():
        return False
    c = pd.read_csv(curves_csv)
    fig, ax = plt.subplots(figsize=(5, 4))
    for w, g in c.groupby("workers"):
        ax.plot(g["epoch"], g["L_total"], label=f"{int(w)} GPU")
    ax.set_xlabel("epoch"); ax.set_ylabel("L_total")
    ax.set_title("Convergence parity (DDP does not change the answer)")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)
    return True


# --------------------------------------------------------------------------- #
# OPT-IN comparison figures. Each returns True only if it actually drew a plot.
# --------------------------------------------------------------------------- #
def _build_precision_frame(scaling_csv, fp32_csv):
    """Return a long DataFrame with a 'precision' column, or None if no compare.

    Two supported layouts:
      (a) one --scaling CSV that already carries a 'precision' column;
      (b) --scaling (treated as bf16) plus a separate --scaling-fp32 CSV.

    If `fp32_csv` is given but the path does not exist, a WARN is printed (rather
    than silently no-op'ing) and the function falls back to layout (a).
    """
    primary = pd.read_csv(scaling_csv)
    if fp32_csv and Path(fp32_csv).exists():
        # Layout (b): label the two files and stack them.
        a = primary.copy()
        if "precision" not in a:
            a["precision"] = "bf16"
        b = pd.read_csv(fp32_csv)
        if "precision" not in b:
            b["precision"] = "fp32"
        combined = pd.concat([a, b], ignore_index=True)
    else:
        if fp32_csv:
            # Path was requested but is not on disk: do not silently drop it.
            print(
                f"[scaling] WARN: --scaling-fp32 path {fp32_csv!r} does not "
                "exist; the bf16-vs-fp32 overlay will fall back to any in-file "
                "'precision' column (and is skipped if there is none).",
                file=sys.stderr,
            )
        # Layout (a): rely on an in-file precision column.
        if "precision" not in primary:
            return None
        combined = primary
    combined = combined[combined["precision"].astype(str).str.len() > 0]
    if combined["precision"].nunique() < 2:
        return None
    return combined


def plot_precision_overlay(scaling_csv, fp32_csv, out_thr, out_eff):
    """bf16-vs-fp32 throughput + efficiency overlay (figure 1)."""
    combined = _build_precision_frame(scaling_csv, fp32_csv)
    if combined is None:
        return False

    colors = {"bf16": "#1f77b4", "fp32": "#ff7f0e"}
    # throughput overlay
    fig, ax = plt.subplots(figsize=(5, 4))
    for prec, g in combined.groupby("precision"):
        g = g.sort_values("workers")
        ax.plot(g["workers"], g["samples_per_sec"], "o-",
                color=colors.get(str(prec)), label=f"{prec}")
    ax.set_xlabel("GPU workers"); ax.set_ylabel("throughput (samples/s)")
    ax.set_title("Throughput: bf16 vs fp32"); ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(out_thr, dpi=150); plt.close(fig)

    # efficiency overlay (each precision against its own 1-worker baseline)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.axhline(1.0, ls="--", color="gray", label="ideal (1.0)")
    for prec, g in combined.groupby("precision"):
        g = _add_efficiency_within(g)
        ax.plot(g["workers"], g["efficiency"], "o-",
                color=colors.get(str(prec)), label=f"{prec}")
    ax.set_xlabel("GPU workers"); ax.set_ylabel("scaling efficiency  T_W / (W*T_1)")
    ax.set_ylim(0, 1.05); ax.set_title("Efficiency: bf16 vs fp32")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(out_eff, dpi=150); plt.close(fig)
    return True


def topology_efficiency_table(scaling_csv):
    """Return a within-group efficiency DataFrame for the topology overlay.

    Reads the CSV, finds the grouping column ('topology' or 'platform'), and for
    each group computes T_W / (W * T_1) via _add_efficiency_within (each group
    against its OWN smallest worker count). Returns (col, frame) or (None, None)
    when there is no grouping column with >=2 distinct values. Split out from the
    plotting so the grouping math is unit-testable without a figure.
    """
    df = pd.read_csv(scaling_csv)
    col = "topology" if "topology" in df else ("platform" if "platform" in df else None)
    if col is None:
        return None, None
    df = df[df[col].astype(str).str.len() > 0]
    if df[col].nunique() < 2:
        return None, None
    parts = []
    for topo, g in df.groupby(col):
        gg = _add_efficiency_within(g)
        gg[col] = topo
        parts.append(gg)
    return col, pd.concat(parts, ignore_index=True)


def plot_topology_efficiency(scaling_csv, out):
    """intra-node (NVLink) vs inter-node (EFA) efficiency overlay (figure 2)."""
    col, frame = topology_efficiency_table(scaling_csv)
    if col is None:
        return False

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.axhline(1.0, ls="--", color="gray", label="ideal (1.0)")
    palette = ["#9467bd", "#8c564b", "#17becf", "#e377c2", "#7f7f7f"]
    for i, (topo, g) in enumerate(frame.groupby(col)):
        g = g.sort_values("workers")
        ax.plot(g["workers"], g["efficiency"], "o-",
                color=palette[i % len(palette)], label=f"{topo}")
    ax.set_xlabel("GPU workers"); ax.set_ylabel("scaling efficiency  T_W / (W*T_1)")
    ax.set_ylim(0, 1.05)
    ax.set_title("Efficiency: intra-node (NVLink) vs inter-node (EFA)")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)
    return True


# Hyperparameters the Tune sweep varies (paper's lambda_1..3 + learning rate).
TUNE_PARAMS = ["l_div", "l_nn", "l_mass", "lr"]
# Preferred validation-metric column names, in priority order.
TUNE_METRIC_CANDIDATES = ["val_L_total", "L_total", "val_loss", "loss"]
# Colormap for the parallel-coordinates lines / colorbar. viridis runs DARK
# (low) -> LIGHT (high). The metric is a loss (lower = better), and the color
# fraction is (metric - min) / (max - min), so the BEST (lowest-loss) trial gets
# frac=0 -> the DARK end. Hence "darker = better" is literally true, and the
# polyline color and the colorbar share this exact cmap so they cannot diverge.
PARALLEL_CMAP = "viridis"


def _resolve_tune_metric(df, requested):
    if requested and requested in df:
        return requested
    for c in TUNE_METRIC_CANDIDATES:
        if c in df:
            return c
    return None


def _resolve_trial_col(df):
    for c in ["trial", "trial_id", "trial_index", "iteration", "training_iteration"]:
        if c in df:
            return c
    return None


def plot_tune(tune_csv, metric, out_curve, out_parallel):
    """Ray Tune tuning curve + parallel-coordinates of params vs metric (fig 3)."""
    if not tune_csv or not Path(tune_csv).exists():
        return False
    df = pd.read_csv(tune_csv)
    metric = _resolve_tune_metric(df, metric)
    params = [p for p in TUNE_PARAMS if p in df]
    if metric is None or not params:
        # Nothing tunable to show; do not draw a misleading figure.
        return False

    # Lower metric is better (these are losses); best-so-far is a running min.
    trial_col = _resolve_trial_col(df)
    if trial_col is not None:
        order = df.sort_values(trial_col).reset_index(drop=True)
        x = order[trial_col]
    else:
        order = df.reset_index(drop=True)
        x = pd.Series(range(1, len(order) + 1))
    best_so_far = order[metric].cummin()

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(x, order[metric], "o", color="#bbbbbb", label=f"trial {metric}")
    ax.plot(x, best_so_far, "-", color="#d62728", label="best so far")
    ax.set_xlabel(trial_col if trial_col else "trial")
    ax.set_ylabel(metric)
    ax.set_title("Ray Tune tuning curve"); ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(out_curve, dpi=150); plt.close(fig)

    # ---- parallel-coordinates: one polyline per trial, axes = params+metric --
    axes_cols = params + [metric]
    sub = order[axes_cols].dropna()
    if sub.empty:
        # curve drew fine; skip the parallel plot if no complete rows
        return True

    # min-max normalize each axis to [0,1] so axes with different scales align
    norm = sub.copy()
    bounds = {}
    for c in axes_cols:
        lo, hi = float(sub[c].min()), float(sub[c].max())
        bounds[c] = (lo, hi)
        norm[c] = 0.5 if hi == lo else (sub[c] - lo) / (hi - lo)

    n_axes = len(axes_cols)
    xs = list(range(n_axes))
    mvals = sub[metric]
    mlo, mhi = float(mvals.min()), float(mvals.max())
    # Same cmap object drives both the polylines and the colorbar so the legend
    # cannot invert relative to the lines. viridis: dark = low loss = better.
    cmap = plt.get_cmap(PARALLEL_CMAP)

    fig, ax = plt.subplots(figsize=(7, 4))
    for idx in range(len(sub)):
        ys = [norm.iloc[idx][c] for c in axes_cols]
        frac = 0.5 if mhi == mlo else (mvals.iloc[idx] - mlo) / (mhi - mlo)
        ax.plot(xs, ys, "-", color=cmap(frac), alpha=0.7, linewidth=1.2)
    for xv in xs:
        ax.axvline(xv, color="#cccccc", linewidth=0.8, zorder=0)
    ax.set_xticks(xs)
    ax.set_xticklabels(axes_cols, rotation=0)
    ax.set_yticks([0.0, 0.5, 1.0])
    # annotate each axis with its real min/max so normalized axes stay readable
    for xv, c in zip(xs, axes_cols):
        lo, hi = bounds[c]
        ax.annotate(f"{hi:.3g}", (xv, 1.0), textcoords="offset points",
                    xytext=(0, 4), ha="center", fontsize=7, color="#444444")
        ax.annotate(f"{lo:.3g}", (xv, 0.0), textcoords="offset points",
                    xytext=(0, -12), ha="center", fontsize=7, color="#444444")
    ax.set_ylabel("normalized value")
    ax.set_title(f"Ray Tune parallel coordinates (color = {metric}, darker = better)")
    sm = plt.cm.ScalarMappable(cmap=cmap,
                               norm=plt.Normalize(vmin=mlo, vmax=mhi))
    sm.set_array([])
    fig.colorbar(sm, ax=ax, label=metric)
    fig.tight_layout(); fig.savefig(out_parallel, dpi=150); plt.close(fig)
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scaling", required=True)
    ap.add_argument("--curves", default="")
    ap.add_argument("--out-dir", default="artifacts/scaling")
    # opt-in comparison inputs (absent => those figures are simply skipped)
    ap.add_argument("--scaling-fp32", default="",
                    help="optional second scaling CSV; --scaling is then bf16 "
                         "and this is fp32, for the precision overlay")
    ap.add_argument("--tune", default="",
                    help="optional Ray Tune results CSV for the tuning-curve and "
                         "parallel-coordinates figures")
    ap.add_argument("--tune-metric", default="",
                    help="metric column in --tune to optimize (default: "
                         "auto-detect val_L_total / L_total)")
    args = ap.parse_args()

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    df = load(args.scaling)

    # --- core figures (unchanged) --------------------------------------------
    plot_throughput(df, out / "throughput.png")
    plot_efficiency(df, out / "efficiency.png")
    has_cost = plot_cost(df, out / "cost_per_epoch.png")
    has_parity = plot_parity(args.curves, out / "convergence_parity.png")

    # --- new opt-in figures --------------------------------------------------
    has_prec = plot_precision_overlay(
        args.scaling, args.scaling_fp32,
        out / "precision_throughput.png", out / "precision_efficiency.png")
    has_topo = plot_topology_efficiency(
        args.scaling, out / "topology_efficiency.png")
    has_tune = plot_tune(
        args.tune, args.tune_metric,
        out / "tune_curve.png", out / "tune_parallel.png")

    cols = ["workers", "samples_per_sec", "efficiency"]
    if has_cost:
        cols.append("cost_per_epoch_usd")
    if "val_L_total" in df:
        cols.append("val_L_total")
    summary = df[cols].to_markdown(index=False, floatfmt=".3f")
    (out / "scaling_summary.md").write_text("# PC-RF Ray Train scaling\n\n" + summary + "\n")

    extra = ""
    if has_prec:
        extra += ", precision_throughput.png, precision_efficiency.png"
    if has_topo:
        extra += ", topology_efficiency.png"
    if has_tune:
        extra += ", tune_curve.png, tune_parallel.png"
    print(f"[scaling] wrote {out}/throughput.png, efficiency.png" +
          (", cost_per_epoch.png" if has_cost else "") +
          (", convergence_parity.png" if has_parity else "") +
          extra +
          ", scaling_summary.md")
    print(summary)


if __name__ == "__main__":
    main()
