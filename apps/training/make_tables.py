"""
Render result tables as LaTeX \\input fragments from a single results file.

Source of truth: results/metrics.json, written by evaluate.py --json-out for each
(dataset, method) it scores on real data. Schema:

    {
      "era5":     {"PC-RF": {"SSIM": .., "PSNR": .., "CRPS": .., "NVR": .., "MCE": ..}, ...},
      "s2":       {"PC-RF": {"SSIM": .., "PSNR": .., "DE": ..}, ...},
      "ablation": {"full":  {"SSIM": .., "DE": .., "NVR": .., "MCE": ..}, ...}
    }

Missing cells render as "--".
The synthetic table uses values copied by hand from experiments/synthetic_sanity_check.py.

    python apps/training/make_tables.py --metrics results/metrics.json --out-dir results
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

# Per-table column order and the method rows we report. Method labels are the
# LaTeX row labels; the best (PC-RF / full) row is bolded.
TABLES = {
    "era5": {
        "caption_metrics": ["SSIM", "PSNR", "CRPS", "NVR", "MCE"],
        "arrows": {"SSIM": r"\uparrow", "PSNR": r"\uparrow", "CRPS": r"\downarrow",
                   "NVR": r"\downarrow", "MCE": r"\downarrow"},
        "methods": ["Bicubic", "SRCNN", "SRGAN", "DDPM", "RF-base", "CorrDiff", "PC-RF"],
        "bold": "__best__",
    },
    "s2": {
        "caption_metrics": ["SSIM", "PSNR", "SAM"],
        "arrows": {"SSIM": r"\uparrow", "PSNR": r"\uparrow", "SAM": r"\downarrow"},
        "methods": ["Bicubic", "SRCNN", "RF-base", "PC-RF"],
        "bold": "__best__",
    },
    "ablation": {
        "caption_metrics": ["SSIM", "DE", "NVR", "MCE"],
        "arrows": {"SSIM": r"\uparrow", "DE": r"\downarrow", "NVR": r"\downarrow",
                   "MCE": r"\downarrow"},
        "methods": ["full", "no_div", "no_nn", "no_mass", "no_s2", "rf_base"],
        "bold": "full",
    },
}

# These rows are copied by hand from the output of experiments/synthetic_sanity_check.py
# (numpy, fixed seed). They come from toy fields, so they say nothing about real data.
SYNTHETIC = {
    "metrics": ["DE", "NVR", "MCE", "RMSE", "Corr", "SSIM"],
    "arrows": {"DE": r"\downarrow", "NVR": r"\downarrow", "MCE": r"\downarrow",
               "RMSE": r"\downarrow", "Corr": r"\uparrow", "SSIM": r"\uparrow"},
    "rows": [
        ("Ground truth",         [0.0000, 0.000, 0.0000, 0.0000, 1.000, 1.000], False),
        ("RF-base (no physics)", [0.0081, 0.236, 0.1838, 0.1397, 0.956, 0.518], False),
        (r"\pcrf (physics)",     [0.0057, 0.000, 0.0000, 0.1116, 0.966, 0.682], True),
    ],
}

_FMT = {"SSIM": "{:.3f}", "PSNR": "{:.2f}", "CRPS": "{:.3f}", "DE": "{:.4f}",
        "NVR": "{:.3f}", "MCE": "{:.4f}", "RMSE": "{:.4f}", "Corr": "{:.3f}",
        "SAM": "{:.3f}"}


def _cell(metric, value, bold):
    if value is None:
        s = "--"
    else:
        s = _FMT.get(metric, "{:.3f}").format(value)
    return rf"\textbf{{{s}}}" if bold else s


def _header(metrics, arrows):
    cols = " & ".join(rf"{m}${arrows[m]}$" for m in metrics)
    return rf"\textbf{{Method}} & {cols} \\"


def render_table(metrics, arrows, rows):
    """rows: list of (label, [values aligned to metrics], is_bold)."""
    ncol = "l" + "c" * len(metrics)
    lines = [rf"\begin{{tabular}}{{{ncol}}}", r"\toprule", _header(metrics, arrows),
             r"\midrule"]
    for label, values, bold in rows:
        if isinstance(bold, (list, tuple)):          # per-cell bold mask
            cells = " & ".join(_cell(m, v, b)
                               for m, v, b in zip(metrics, values, bold))
            lab = label
        else:                                        # whole-row bold (label + cells)
            cells = " & ".join(_cell(m, v, bold) for m, v in zip(metrics, values))
            lab = rf"\textbf{{{label}}}" if bold else label
        lines.append(rf"{lab} & {cells} \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


def emit_dataset(name, spec, data, out_dir):
    metrics = spec["caption_metrics"]
    rows = []
    table_data = data.get(name, {})
    values_by_method = [[table_data.get(method, {}).get(m) for m in metrics]
                        for method in spec["methods"]]
    if spec.get("bold") == "__best__":
        # Bold the true best value in each column (max for up-metrics, min for
        # down-metrics), honoring ties. Comparison uses the raw (unrounded) values.
        best_idx = {}
        for j, m in enumerate(metrics):
            col = [(i, vs[j]) for i, vs in enumerate(values_by_method)
                   if vs[j] is not None]
            if not col:
                best_idx[j] = set()
                continue
            up = spec["arrows"][m] == r"\uparrow"
            target = max(v for _, v in col) if up else min(v for _, v in col)
            best_idx[j] = {i for i, v in col if abs(v - target) < 1e-9}
        for i, method in enumerate(spec["methods"]):
            mask = [i in best_idx[j] for j in range(len(metrics))]
            rows.append((method, values_by_method[i], mask))
    else:
        for i, method in enumerate(spec["methods"]):
            rows.append((method, values_by_method[i], method == spec["bold"]))
    tex = render_table(metrics, spec["arrows"], rows)
    out = Path(out_dir) / f"{name}_metrics.tex"
    out.write_text(tex)
    filled = sum(1 for _, vs, _ in rows for v in vs if v is not None)
    total = len(rows) * len(metrics)
    print(f"[tables] wrote {out}  ({filled}/{total} cells filled)")


def emit_synthetic(out_dir):
    tex = render_table(SYNTHETIC["metrics"], SYNTHETIC["arrows"], SYNTHETIC["rows"])
    out = Path(out_dir) / "synthetic_metrics.tex"
    out.write_text(tex)
    print(f"[tables] wrote {out}  (synthetic mechanism check)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics", default="results/metrics.json",
                    help="JSON written by evaluate.py --json-out (may not exist yet)")
    ap.add_argument("--out-dir", default="results")
    args = ap.parse_args()

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    data = {}
    p = Path(args.metrics)
    if p.exists():
        data = json.loads(p.read_text())
    else:
        print(f"[tables] {p} not found; emitting empty (--) real-data tables")

    emit_synthetic(args.out_dir)
    for name, spec in TABLES.items():
        emit_dataset(name, spec, data, args.out_dir)


if __name__ == "__main__":
    main()
