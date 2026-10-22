"""Pure-python pass/fail parser for the PC-RF end-to-end smoke run.

The end-to-end Slurm job (not in this repository) launches a short Ray Train run on SYNTHETIC
data and Ray writes per-epoch metrics as newline-delimited JSON (JSONL) to
  <runs>/pcrf-w<W>/TorchTrainer_*/result.json
one JSON object per reported epoch, each carrying "epoch" and "L_total"
(L_total is emitted by pcrf.flow.RectifiedFlow.training_loss).

This module is deliberately dependency-free (stdlib only: json, glob, os). It
imports NO torch / ray / numpy, so it byte-compiles and unit-tests on a login
node. The sbatch invokes it INSIDE the container after training; the unit test
exercises the same pure functions on hand-written JSONL fixtures.

The headline assertion is: L_total at the LAST reported epoch is strictly less
than L_total at the FIRST reported epoch (the loss decreased over training).
"""
from __future__ import annotations

import glob
import json
import os

LOSS_KEY = "L_total"          # emitted by pcrf.flow.RectifiedFlow.training_loss


def find_result_json(runs_root: str, workers: int) -> str | None:
    """Return the newest Ray Train result.json for a pcrf-w<W> run, or None.

    Mirrors collect_scaling.py: Ray nests the trial dir as
    <runs_root>/pcrf-w<W>/TorchTrainer_*/result.json, but tolerate a flat
    <runs_root>/pcrf-w<W>/result.json too. Newest by mtime wins so a re-run in
    the same dir picks the latest trial.
    """
    pats = [
        os.path.join(runs_root, f"pcrf-w{workers}", "**", "result.json"),
        os.path.join(runs_root, f"pcrf-w{workers}", "result.json"),
    ]
    hits: list[str] = []
    for pat in pats:
        hits.extend(glob.glob(pat, recursive=True))
    if not hits:
        return None
    return max(hits, key=os.path.getmtime)


def find_v2_snapshot(runs_root: str, workers: int) -> str | None:
    """Return the Ray Train **v2** checkpoint_manager_snapshot.json, or None.

    Ray Train v2 (Ray >= ~2.50, this stack runs 2.55) no longer writes a Tune
    `result.json` JSONL for a plain TorchTrainer.fit(); the per-run state lives
    in <runs_root>/pcrf-w<W>/checkpoint_manager_snapshot.json as a single JSON
    object whose `checkpoint_results[*].metrics` carry the same per-epoch dicts
    (epoch, L_total, ...) the v1 result.json used to. This is the decoupled,
    layout-stable source the e2e parses when no result.json exists.
    """
    pats = [
        os.path.join(runs_root, f"pcrf-w{workers}", "**",
                     "checkpoint_manager_snapshot.json"),
        os.path.join(runs_root, f"pcrf-w{workers}",
                     "checkpoint_manager_snapshot.json"),
    ]
    hits: list[str] = []
    for pat in pats:
        hits.extend(glob.glob(pat, recursive=True))
    if not hits:
        return None
    return max(hits, key=os.path.getmtime)


def read_epoch_records(result_json_path: str) -> list[dict]:
    """Parse a Ray Train metrics file into one dict per reported epoch.

    Handles BOTH layouts:
      - v1 / Tune  result.json: newline-delimited JSON, one object per epoch.
      - v2 checkpoint_manager_snapshot.json: a single JSON object with a
        `checkpoint_results` list, each entry carrying a `metrics` dict.
    Records are returned sorted by "epoch" so first/last loss comparison is over
    training order regardless of on-disk ordering. Tolerates blank lines and a
    half-written trailing line (a crash mid-write must not turn a real PASS into
    a parse crash), matching collect_scaling.py.
    """
    # v2 snapshot: one JSON object with checkpoint_results[].metrics.
    if os.path.basename(result_json_path) == "checkpoint_manager_snapshot.json":
        try:
            with open(result_json_path) as fh:
                obj = json.load(fh)
        except (json.JSONDecodeError, OSError):
            return []
        recs = [cr["metrics"] for cr in obj.get("checkpoint_results", [])
                if isinstance(cr, dict) and isinstance(cr.get("metrics"), dict)]
        return sorted(recs, key=lambda r: r.get("epoch", 0))

    # v1 / Tune: JSONL, one object per line.
    recs: list[dict] = []
    with open(result_json_path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                recs.append(json.loads(line))
            except json.JSONDecodeError:
                # tolerate a half-written trailing line
                pass
    return sorted(recs, key=lambda r: r.get("epoch", 0))


def extract_losses(records: list[dict], loss_key: str = LOSS_KEY) -> list[float]:
    """Pull the loss value out of each epoch record that carries loss_key.

    Records missing the key (e.g. a non-rank-0 report that omitted logs) are
    skipped rather than treated as zero, so the first/last comparison is over
    real loss samples only.
    """
    out: list[float] = []
    for rec in records:
        if loss_key in rec and rec[loss_key] is not None:
            out.append(float(rec[loss_key]))
    return out


def loss_decreased(losses: list[float], min_epochs: int = 2) -> bool:
    """True iff there are >= min_epochs loss samples and last < first.

    Strict inequality: a flat or rising loss is a FAIL. Requires at least
    min_epochs samples so a single-epoch (or empty) run cannot vacuously PASS.
    """
    if len(losses) < min_epochs:
        return False
    return losses[-1] < losses[0]


def evaluate_result_file(result_json_path: str, min_epochs: int = 2,
                         loss_key: str = LOSS_KEY) -> dict:
    """Read a result.json and return a structured verdict.

    Returns a dict with: passed (bool), reason (str), first_loss, last_loss,
    n_epochs. Pure: no printing, no file writes, so it is unit-testable. The
    sbatch driver renders this dict into the PASS/FAIL summary line.
    """
    records = read_epoch_records(result_json_path)
    losses = extract_losses(records, loss_key=loss_key)
    n = len(losses)
    if n < min_epochs:
        return {
            "passed": False,
            "reason": f"need >= {min_epochs} epochs with {loss_key}; got {n}",
            "first_loss": (losses[0] if n else None),
            "last_loss": (losses[-1] if n else None),
            "n_epochs": n,
        }
    passed = loss_decreased(losses, min_epochs=min_epochs)
    first, last = losses[0], losses[-1]
    if passed:
        reason = f"{loss_key} decreased: first={first:.6g} last={last:.6g}"
    else:
        reason = f"{loss_key} did NOT decrease: first={first:.6g} last={last:.6g}"
    return {
        "passed": passed,
        "reason": reason,
        "first_loss": first,
        "last_loss": last,
        "n_epochs": n,
    }


def main(argv: list[str] | None = None) -> int:
    """CLI: print a PASS/FAIL line and exit 0 (PASS) / 1 (FAIL).

    The sbatch calls this inside the container. It does NOT write the summary /
    DONE marker itself; the sbatch owns those (so the file layout stays in one
    place). Exit code lets the sbatch branch on PASS vs FAIL.
    """
    import argparse

    ap = argparse.ArgumentParser(description="PC-RF e2e loss-decreased check")
    ap.add_argument("--runs", required=True,
                    help="Ray runs root holding the pcrf-w<W> trial dir")
    ap.add_argument("--workers", type=int, required=True,
                    help="worker count W used in the run name pcrf-w<W>")
    ap.add_argument("--min-epochs", type=int, default=2,
                    help="minimum reported epochs required to render a verdict")
    args = ap.parse_args(argv)

    # Prefer a v1/Tune result.json; fall back to the v2 checkpoint-manager
    # snapshot (Ray 2.55 TorchTrainer.fit() writes the latter, not result.json).
    rj = find_result_json(args.runs, args.workers)
    if rj is None:
        rj = find_v2_snapshot(args.runs, args.workers)
    if rj is None:
        print(f"FAIL no result.json or checkpoint_manager_snapshot.json under "
              f"{args.runs}/pcrf-w{args.workers}")
        return 1
    verdict = evaluate_result_file(rj, min_epochs=args.min_epochs)
    tag = "PASS" if verdict["passed"] else "FAIL"
    print(f"{tag} {verdict['reason']} (epochs={verdict['n_epochs']}) [{rj}]")
    return 0 if verdict["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
