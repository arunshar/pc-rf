"""Pure-python unit tests for the PC-RF e2e loss-decreased parser.

These exercise apps/training/e2e_check.py on hand-written Ray Train JSONL
fixtures. NO torch / ray / GPU: this is a fast smoke test for the parser and
verdict logic that an end-to-end Slurm job (not in this repository) relies on. conftest.py adds
apps/training to sys.path so `import e2e_check` resolves.
"""
import json
import os

import e2e_check as ec


# ----------------------------------------------------------------- fixtures
def _write_jsonl(path, records):
    """Write one JSON object per line (Ray Train's result.json format)."""
    with open(path, "w") as fh:
        for rec in records:
            fh.write(json.dumps(rec) + "\n")
    return str(path)


def _write_v2_snapshot(path, records):
    """Write a Ray Train v2 checkpoint_manager_snapshot.json (one JSON object
    with a checkpoint_results list whose entries carry a metrics dict)."""
    obj = {"ray_version": "2.55.1",
           "checkpoint_results": [
               {"version": 0,
                "checkpoint_dir_name": f"checkpoint_{i}",
                "metrics": rec}
               for i, rec in enumerate(records)]}
    with open(path, "w") as fh:
        json.dump(obj, fh)
    return str(path)


# ----------------------------------------------------------------- extract_losses
def test_extract_losses_pulls_l_total_in_order():
    recs = [{"epoch": 0, "L_total": 3.0}, {"epoch": 1, "L_total": 2.0},
            {"epoch": 2, "L_total": 1.0}]
    assert ec.extract_losses(recs) == [3.0, 2.0, 1.0]


def test_extract_losses_skips_records_missing_key():
    # A non-rank-0 report may omit logs; those rows must be skipped, not zeroed.
    recs = [{"epoch": 0, "L_total": 3.0}, {"epoch": 0, "samples_per_sec": 10.0},
            {"epoch": 1, "L_total": 1.5}]
    assert ec.extract_losses(recs) == [3.0, 1.5]


def test_extract_losses_skips_none_value():
    recs = [{"L_total": None}, {"L_total": 2.0}]
    assert ec.extract_losses(recs) == [2.0]


# ----------------------------------------------------------------- loss_decreased
def test_loss_decreased_true_when_last_below_first():
    assert ec.loss_decreased([5.0, 4.0, 2.0]) is True


def test_loss_decreased_false_when_flat():
    # Strict inequality: a flat loss is a FAIL.
    assert ec.loss_decreased([2.0, 2.0]) is False


def test_loss_decreased_false_when_rising():
    assert ec.loss_decreased([1.0, 3.0]) is False


def test_loss_decreased_false_with_too_few_epochs():
    # A single-epoch run must not vacuously pass.
    assert ec.loss_decreased([1.0]) is False
    assert ec.loss_decreased([]) is False


def test_loss_decreased_uses_first_and_last_not_monotonic():
    # A transient bump in the middle is fine; only first vs last matters.
    assert ec.loss_decreased([5.0, 9.0, 1.0]) is True


# ----------------------------------------------------------------- read_epoch_records
def test_read_epoch_records_tolerates_blank_and_truncated_lines(tmp_path):
    p = tmp_path / "result.json"
    with open(p, "w") as fh:
        fh.write(json.dumps({"epoch": 0, "L_total": 4.0}) + "\n")
        fh.write("\n")                                  # blank line
        fh.write(json.dumps({"epoch": 1, "L_total": 2.0}) + "\n")
        fh.write('{"epoch": 2, "L_total": 1.0')          # truncated, no close
    recs = ec.read_epoch_records(str(p))
    assert ec.extract_losses(recs) == [4.0, 2.0]


# ----------------------------------------------------------------- evaluate_result_file
def test_evaluate_result_file_pass(tmp_path):
    p = _write_jsonl(tmp_path / "result.json",
                     [{"epoch": 0, "L_total": 3.0}, {"epoch": 1, "L_total": 2.5},
                      {"epoch": 2, "L_total": 1.2}])
    v = ec.evaluate_result_file(p)
    assert v["passed"] is True
    assert v["n_epochs"] == 3
    assert v["first_loss"] == 3.0 and v["last_loss"] == 1.2
    assert "decreased" in v["reason"]


def test_evaluate_result_file_fail_not_decreasing(tmp_path):
    p = _write_jsonl(tmp_path / "result.json",
                     [{"epoch": 0, "L_total": 1.0}, {"epoch": 1, "L_total": 2.0}])
    v = ec.evaluate_result_file(p)
    assert v["passed"] is False
    assert "did NOT decrease" in v["reason"]


def test_evaluate_result_file_fail_too_few_epochs(tmp_path):
    p = _write_jsonl(tmp_path / "result.json", [{"epoch": 0, "L_total": 1.0}])
    v = ec.evaluate_result_file(p)
    assert v["passed"] is False
    assert v["n_epochs"] == 1


# ----------------------------------------------------------------- find_result_json
def test_find_result_json_nested_trial_dir(tmp_path):
    # Ray nests: <runs>/pcrf-w<W>/TorchTrainer_*/result.json
    trial = tmp_path / "pcrf-w2" / "TorchTrainer_abc123"
    os.makedirs(trial)
    p = _write_jsonl(trial / "result.json", [{"epoch": 0, "L_total": 1.0}])
    found = ec.find_result_json(str(tmp_path), 2)
    assert found == p


def test_find_result_json_returns_none_when_absent(tmp_path):
    assert ec.find_result_json(str(tmp_path), 4) is None


# ----------------------------------------------------------------- CLI main
def test_main_returns_zero_on_pass(tmp_path):
    trial = tmp_path / "pcrf-w2" / "TorchTrainer_x"
    os.makedirs(trial)
    _write_jsonl(trial / "result.json",
                 [{"epoch": 0, "L_total": 5.0}, {"epoch": 1, "L_total": 1.0}])
    rc = ec.main(["--runs", str(tmp_path), "--workers", "2", "--min-epochs", "2"])
    assert rc == 0


def test_main_returns_one_on_missing_result(tmp_path):
    rc = ec.main(["--runs", str(tmp_path), "--workers", "2"])
    assert rc == 1


# ------------------------------------------------- Ray Train v2 snapshot path
def test_read_epoch_records_reads_v2_snapshot(tmp_path):
    # The v2 snapshot is one JSON object; read_epoch_records must pull the
    # per-epoch metrics dicts out of checkpoint_results[].metrics, sorted by
    # epoch (here written out of order to prove the sort).
    p = _write_v2_snapshot(tmp_path / "checkpoint_manager_snapshot.json",
                           [{"epoch": 2, "L_total": 3.0},
                            {"epoch": 0, "L_total": 5.0},
                            {"epoch": 1, "L_total": 4.0}])
    recs = ec.read_epoch_records(p)
    assert [r["epoch"] for r in recs] == [0, 1, 2]
    assert ec.extract_losses(recs) == [5.0, 4.0, 3.0]


def test_find_v2_snapshot_nested_and_flat(tmp_path):
    # nested under the run name dir (where TorchTrainer.fit() writes it)
    run = tmp_path / "pcrf-w2"
    os.makedirs(run)
    p = _write_v2_snapshot(run / "checkpoint_manager_snapshot.json",
                           [{"epoch": 0, "L_total": 1.0}])
    assert ec.find_v2_snapshot(str(tmp_path), 2) == p
    assert ec.find_v2_snapshot(str(tmp_path), 4) is None


def test_main_falls_back_to_v2_snapshot_pass(tmp_path):
    # No result.json anywhere; only the v2 snapshot exists. main() must fall
    # back to it and PASS on a decreasing loss (this is the real Ray 2.55 case
    # that made the e2e wrongly FAIL before the fix).
    run = tmp_path / "pcrf-w2"
    os.makedirs(run)
    _write_v2_snapshot(run / "checkpoint_manager_snapshot.json",
                       [{"epoch": 0, "L_total": 4.25},
                        {"epoch": 1, "L_total": 4.16},
                        {"epoch": 2, "L_total": 3.15},
                        {"epoch": 3, "L_total": 3.35},
                        {"epoch": 4, "L_total": 3.20}])
    rc = ec.main(["--runs", str(tmp_path), "--workers", "2", "--min-epochs", "2"])
    assert rc == 0


def test_main_v2_snapshot_fail_on_rising_loss(tmp_path):
    # Snapshot present but loss rose -> FAIL (the strict-inequality guard still
    # applies on the v2 path).
    run = tmp_path / "pcrf-w2"
    os.makedirs(run)
    _write_v2_snapshot(run / "checkpoint_manager_snapshot.json",
                       [{"epoch": 0, "L_total": 1.0},
                        {"epoch": 1, "L_total": 2.0}])
    rc = ec.main(["--runs", str(tmp_path), "--workers", "2", "--min-epochs", "2"])
    assert rc == 1
