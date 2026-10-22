"""Unit tests for apps/training/make_scaling_plots.py (the PC-RF scaling figs).

CPU-only, matplotlib Agg, stdlib + matplotlib + pandas (pandas is used only
because the module under test consumes DataFrames; the test inputs are written
with the stdlib csv module). Each test asserts real behavior:

  * the precision-frame builder reads both supported layouts (in-file
    'precision' column, and the two-CSV --scaling / --scaling-fp32 split);
  * a missing --scaling-fp32 path WARNs instead of silently no-op'ing;
  * the topology-efficiency grouping computes within-group efficiency
    T_W / (W * T_1) per group, against each group's own smallest worker count;
  * the multi-condition guard fires for a topology CSV (must not be --scaling);
  * the parallel-coords color direction matches "darker = better" (the best /
    lowest-loss trial maps to the dark end of the colormap);
  * the plot functions write PNG files to a tmp dir without error.

conftest.py puts apps/training on sys.path, so we import the script directly.
"""
import csv
import os

import pytest

pytest.importorskip("matplotlib")
pytest.importorskip("pandas")

import matplotlib
matplotlib.use("Agg")

import make_scaling_plots as msp


# --------------------------------------------------------------------------- #
# helpers (stdlib csv writers for synthetic inputs)
# --------------------------------------------------------------------------- #
def _write_csv(path, header, rows):
    with open(path, "w", newline="") as fh:
        wri = csv.writer(fh)
        wri.writerow(header)
        wri.writerows(rows)
    return str(path)


def _write_single_scaling(path, precision=None, topology=None):
    """A clean single-condition sweep (one row per worker count)."""
    header = ["workers", "samples_per_sec", "epoch_time_s",
              "gpu_cost_per_hr", "val_L_total"]
    rows = [
        [1, 100.0, 40.0, 1.20, 0.851],
        [2, 190.0, 21.0, 1.20, 0.850],
        [4, 360.0, 11.1, 1.20, 0.857],
        [8, 640.0, 6.25, 1.20, 0.860],
    ]
    if precision is not None:
        header = header + ["precision"]
        rows = [r + [precision] for r in rows]
    if topology is not None:
        header = header + ["topology"]
        rows = [r + [topology] for r in rows]
    return _write_csv(path, header, rows)


def _write_topology_scaling(path):
    """A two-group topology sweep (intra-nvlink + inter-efa)."""
    header = ["workers", "samples_per_sec", "epoch_time_s",
              "gpu_cost_per_hr", "val_L_total", "topology"]
    rows = [
        [1, 120.0, 40.0, 1.20, 0.851, "intra-nvlink"],
        [2, 224.0, 21.4, 1.20, 0.850, "intra-nvlink"],
        [4, 420.0, 11.4, 1.20, 0.857, "intra-nvlink"],
        [1, 120.0, 40.0, 3.67, 0.851, "inter-efa"],
        [2, 208.0, 23.1, 3.67, 0.850, "inter-efa"],
        [4, 360.0, 13.3, 3.67, 0.857, "inter-efa"],
    ]
    return _write_csv(path, header, rows)


def _write_tune(path):
    header = ["trial", "l_div", "l_nn", "l_mass", "lr", "val_L_total"]
    rows = [
        [0, 1.0, 1.0, 1.0, 2.0e-4, 0.872],   # worst (highest loss)
        [1, 0.5, 1.0, 2.0, 1.0e-4, 0.864],
        [2, 0.8, 1.2, 0.8, 1.5e-4, 0.849],   # best (lowest loss)
        [3, 1.0, 2.0, 0.5, 2.0e-4, 0.858],
    ]
    return _write_csv(path, header, rows)


def _relative_luminance(rgba):
    """Rec.709-ish luma in [0,1]; lower = darker."""
    r, g, b = rgba[0], rgba[1], rgba[2]
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


# --------------------------------------------------------------------------- #
# _build_precision_frame: both layouts + the missing-fp32 WARN
# --------------------------------------------------------------------------- #
def test_precision_frame_reads_in_file_precision_column(tmp_path):
    """Layout (a): a single CSV that already has a 'precision' column."""
    bf16 = _write_single_scaling(tmp_path / "bf16.csv", precision="bf16")
    fp32 = _write_single_scaling(tmp_path / "fp32.csv", precision="fp32")
    # stack the two into one file so the in-file column carries both values
    import pandas as pd
    combined_path = tmp_path / "combined.csv"
    pd.concat([pd.read_csv(bf16), pd.read_csv(fp32)],
              ignore_index=True).to_csv(combined_path, index=False)

    frame = msp._build_precision_frame(str(combined_path), "")
    assert frame is not None
    assert set(frame["precision"].unique()) == {"bf16", "fp32"}
    assert frame["precision"].nunique() == 2


def test_precision_frame_two_csv_layout_labels_bf16_and_fp32(tmp_path):
    """Layout (b): --scaling (-> bf16) plus a separate --scaling-fp32 CSV."""
    primary = _write_single_scaling(tmp_path / "primary.csv")          # no label
    fp32 = _write_single_scaling(tmp_path / "fp32.csv")                # no label
    frame = msp._build_precision_frame(primary, fp32)
    assert frame is not None
    assert set(frame["precision"].unique()) == {"bf16", "fp32"}
    # both files had 4 worker rows -> 8 stacked rows
    assert len(frame) == 8


def test_precision_frame_missing_fp32_path_warns(tmp_path, capsys):
    """A requested-but-missing --scaling-fp32 path must WARN, not silently drop."""
    primary = _write_single_scaling(tmp_path / "primary.csv")  # no precision col
    missing = str(tmp_path / "does_not_exist.csv")
    frame = msp._build_precision_frame(primary, missing)
    err = capsys.readouterr().err
    assert "WARN" in err and "scaling-fp32" in err
    # no in-file precision column to fall back to => no overlay
    assert frame is None


def test_precision_frame_no_compare_returns_none(tmp_path):
    """A plain single-condition CSV with no precision info => None (skip fig)."""
    primary = _write_single_scaling(tmp_path / "primary.csv")
    assert msp._build_precision_frame(primary, "") is None


# --------------------------------------------------------------------------- #
# topology grouping: within-group efficiency T_W / (W * T_1)
# --------------------------------------------------------------------------- #
def test_topology_efficiency_grouping_is_within_group(tmp_path):
    """Each group's efficiency uses that group's OWN 1-worker baseline."""
    path = _write_topology_scaling(tmp_path / "topo.csv")
    col, frame = msp.topology_efficiency_table(path)
    assert col == "topology"
    assert set(frame["topology"].unique()) == {"intra-nvlink", "inter-efa"}

    # every group's smallest worker count is the baseline => efficiency 1.0 there
    for _, g in frame.groupby("topology"):
        base = g.loc[g["workers"].idxmin()]
        assert abs(float(base["efficiency"]) - 1.0) < 1e-9

    # spot-check one off-baseline point: inter-efa W=4, sps=360, sps_1=120, W_1=1
    # ideal = 120 * 4 / 1 = 480 ; efficiency = 360 / 480 = 0.75
    inter = frame[frame["topology"] == "inter-efa"]
    w4 = inter[inter["workers"] == 4].iloc[0]
    assert abs(float(w4["efficiency"]) - 0.75) < 1e-9
    assert abs(float(w4["ideal_sps"]) - 480.0) < 1e-9

    # intra-nvlink W=2: ideal = 120*2 = 240, sps=224 -> 224/240
    intra = frame[frame["topology"] == "intra-nvlink"]
    w2 = intra[intra["workers"] == 2].iloc[0]
    assert abs(float(w2["efficiency"]) - (224.0 / 240.0)) < 1e-9


def test_topology_table_none_when_single_condition(tmp_path):
    """No grouping column with >=2 values => no topology overlay."""
    path = _write_single_scaling(tmp_path / "single.csv")
    col, frame = msp.topology_efficiency_table(path)
    assert col is None and frame is None


def test_add_efficiency_within_against_own_baseline(tmp_path):
    """_add_efficiency_within is relative to the group's smallest worker count."""
    import pandas as pd
    g = pd.DataFrame({
        "workers": [2, 4, 8],            # baseline here is W=2, not W=1
        "samples_per_sec": [200.0, 360.0, 640.0],
    })
    out = msp._add_efficiency_within(g)
    base = out.loc[out["workers"].idxmin()]
    assert abs(float(base["efficiency"]) - 1.0) < 1e-9
    # W=4: ideal = 200 * 4 / 2 = 400, eff = 360/400 = 0.9
    w4 = out[out["workers"] == 4].iloc[0]
    assert abs(float(w4["efficiency"]) - 0.9) < 1e-9


# --------------------------------------------------------------------------- #
# multi-condition guard (must-fix: a topology CSV must not double as --scaling)
# --------------------------------------------------------------------------- #
def test_detect_multicondition_flags_topology_csv(tmp_path):
    import pandas as pd
    df = pd.read_csv(_write_topology_scaling(tmp_path / "topo.csv"))
    reason = msp.detect_multicondition(df)
    assert reason is not None
    # either the topology grouping or the repeated worker counts trips it
    assert "topology" in reason or "repeat" in reason


def test_detect_multicondition_none_for_clean_sweep(tmp_path):
    import pandas as pd
    df = pd.read_csv(_write_single_scaling(tmp_path / "single.csv"))
    assert msp.detect_multicondition(df) is None


def test_load_warns_on_multicondition_scaling(tmp_path, capsys):
    """load() on a topology CSV still runs but prints a loud zig-zag WARN."""
    path = _write_topology_scaling(tmp_path / "topo.csv")
    df = msp.load(path)
    err = capsys.readouterr().err
    assert "WARN" in err and "MULTI-CONDITION" in err
    # load still produced the efficiency columns (it does not abort)
    assert "efficiency" in df.columns


# --------------------------------------------------------------------------- #
# parallel-coords color direction: darker = better (best/lowest loss -> dark)
# --------------------------------------------------------------------------- #
def test_parallel_cmap_dark_is_low_loss():
    """The documented luminance end: cmap(0.0) (best/lowest loss) is darker."""
    import matplotlib.pyplot as plt
    cmap = plt.get_cmap(msp.PARALLEL_CMAP)
    lum_best = _relative_luminance(cmap(0.0))   # frac=0 == lowest loss == best
    lum_worst = _relative_luminance(cmap(1.0))  # frac=1 == highest loss == worst
    assert lum_best < lum_worst, (
        "best (lowest-loss) trial must map to the DARK end so the "
        "'darker = better' label is correct")


def test_parallel_color_direction_matches_best_trial(tmp_path):
    """Replicate the module's frac math on the example tune data: the best
    (lowest-loss) trial gets the darkest color, the worst the lightest."""
    import matplotlib.pyplot as plt
    cmap = plt.get_cmap(msp.PARALLEL_CMAP)
    losses = [0.872, 0.864, 0.849, 0.858]      # matches _write_tune
    mlo, mhi = min(losses), max(losses)
    fracs = [(m - mlo) / (mhi - mlo) for m in losses]
    lums = [_relative_luminance(cmap(f)) for f in fracs]
    best_i = losses.index(min(losses))         # 0.849 -> should be darkest
    worst_i = losses.index(max(losses))        # 0.872 -> should be lightest
    assert lums[best_i] == min(lums)
    assert lums[worst_i] == max(lums)
    # the best trial maps to frac 0 (the dark end)
    assert fracs[best_i] == 0.0


# --------------------------------------------------------------------------- #
# end to end: every plot writes a PNG without error
# --------------------------------------------------------------------------- #
def test_core_plots_write_pngs(tmp_path):
    path = _write_single_scaling(tmp_path / "single.csv")
    df = msp.load(path)
    thr = tmp_path / "throughput.png"
    eff = tmp_path / "efficiency.png"
    cost = tmp_path / "cost.png"
    msp.plot_throughput(df, thr)
    msp.plot_efficiency(df, eff)
    assert msp.plot_cost(df, cost) is True
    for p in (thr, eff, cost):
        assert p.exists() and os.path.getsize(p) > 0


def test_precision_overlay_writes_pngs(tmp_path):
    primary = _write_single_scaling(tmp_path / "primary.csv")
    fp32 = _write_single_scaling(tmp_path / "fp32.csv")
    out_thr = tmp_path / "prec_thr.png"
    out_eff = tmp_path / "prec_eff.png"
    assert msp.plot_precision_overlay(primary, fp32, out_thr, out_eff) is True
    assert out_thr.exists() and os.path.getsize(out_thr) > 0
    assert out_eff.exists() and os.path.getsize(out_eff) > 0


def test_topology_overlay_writes_png(tmp_path):
    path = _write_topology_scaling(tmp_path / "topo.csv")
    out = tmp_path / "topo.png"
    assert msp.plot_topology_efficiency(path, out) is True
    assert out.exists() and os.path.getsize(out) > 0


def test_tune_plots_write_pngs(tmp_path):
    path = _write_tune(tmp_path / "tune.csv")
    out_curve = tmp_path / "tune_curve.png"
    out_par = tmp_path / "tune_parallel.png"
    assert msp.plot_tune(path, "", out_curve, out_par) is True
    assert out_curve.exists() and os.path.getsize(out_curve) > 0
    assert out_par.exists() and os.path.getsize(out_par) > 0
