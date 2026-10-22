"""Serving benchmark tests: run bench_serving.py on a tiny CPU network and check
the report schema, export parity, and the post-projection guarantee.

These need the export stack (onnx + onnxruntime), like tests/test_export.py.
AOTInductor and torch.compile are left out here because they need a C++
toolchain and are slow on CPU; the cluster run covers them.
"""
import json

import pytest

pytest.importorskip("torch")
pytest.importorskip("onnx")
pytest.importorskip("onnxruntime")
import bench_serving as bs
import torch

TINY = ["--tiny", "--fine", "32", "--base", "16", "--device", "cpu",
        "--batches", "1,2", "--parity-batches", "1,3",
        "--steps", "2", "--warmup", "1", "--iters", "3", "--sample-iters", "2"]


def _run(tmp_path, *extra):
    out = tmp_path / "bench.json"
    bs.main(TINY + ["--out", str(out), "--backends", "eager,ort", *extra])
    return json.loads(out.read_text())


def test_report_schema(tmp_path):
    rep = _run(tmp_path)
    assert rep["schema_version"] == bs.SCHEMA_VERSION
    assert rep["config"]["params"] > 0
    assert rep["weights"] == {"weights": "random", "ckpt_sha256": None}
    for name in ("eager", "ort"):
        e = rep["backends"][name]
        assert e["status"] == "ok"
        assert set(e["parity_max_abs_err"]) == {"1", "3"}
        for part in ("velocity", "sample"):
            for b in ("1", "2"):
                m = e[part][b]
                assert m["p50_ms"] > 0 and m["p99_ms"] >= m["p50_ms"]
                assert m["samples_per_s"] > 0


def test_ort_parity_at_unseen_batch(tmp_path):
    rep = _run(tmp_path)
    errs = rep["backends"]["ort"]["parity_max_abs_err"]
    # batch 3 differs from the batch-1 export example, so a baked-in static
    # batch dimension would fail here
    assert max(errs.values()) < 1e-4, errs
    assert rep["backends"]["eager"]["parity_max_abs_err"] == {"1": 0.0, "3": 0.0}


def test_projection_guarantee_holds_for_every_backend(tmp_path):
    rep = _run(tmp_path)
    for name in ("eager", "ort"):
        pp = rep["backends"][name]["post_projection"]
        assert pp["nvr"] == 0.0
        assert pp["mce_max"] < 1e-5


def test_trained_weights_are_hashed_not_pathed(tmp_path):
    m, _f, _p, _d = bs.ex.load_pcrf_modules()
    net = bs.build_net(m, 3, 7, 16, tiny=True)
    ck = tmp_path / "ck.pt"
    torch.save({"model": net.state_dict(), "in_ch": 3, "cond_ch": 7}, ck)
    rep = _run(tmp_path, "--ckpt", str(ck))
    assert rep["weights"]["weights"] == "trained"
    assert len(rep["weights"]["ckpt_sha256"]) == 64
    assert str(tmp_path) not in json.dumps(rep)


def test_failed_backend_is_recorded_not_raised(tmp_path, monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("no toolchain")
    monkeypatch.setattr(bs, "make_aoti", boom)
    out = tmp_path / "bench.json"
    bs.main(TINY + ["--out", str(out), "--backends", "eager,aoti"])
    rep = json.loads(out.read_text())
    assert rep["backends"]["aoti"]["status"] == "failed"
    assert rep["backends"]["aoti"]["error"] == "RuntimeError"
    assert rep["backends"]["eager"]["status"] == "ok"


def test_unknown_backend_rejected():
    with pytest.raises(SystemExit):
        bs.parse_args(["--out", "x.json", "--backends", "tensorrt"])


def test_measurement_failure_is_recorded_and_earlier_results_saved(tmp_path, monkeypatch):
    real_make_ort = bs.make_ort

    def broken_ort(*a, **k):
        _run, used = real_make_ort(*a, **k)

        def boom(x, t, c):
            raise RuntimeError("provider died mid-run")
        return boom, used
    monkeypatch.setattr(bs, "make_ort", broken_ort)
    out = tmp_path / "bench.json"
    bs.main(TINY + ["--out", str(out), "--backends", "eager,ort"])
    rep = json.loads(out.read_text())
    assert rep["backends"]["eager"]["status"] == "ok"
    assert rep["backends"]["ort"]["status"] == "failed"
    assert rep["backends"]["ort"]["stage"] == "measure"
    assert not (tmp_path / "bench.json.tmp").exists()


def test_report_is_saved_after_each_backend(tmp_path):
    saved = []
    args = bs.parse_args(TINY + ["--out", str(tmp_path / "x.json"), "--backends", "eager,ort"])
    bs.run_bench(args, save=lambda r: saved.append(sorted(r["backends"])))
    assert saved[0] == ["eager"]
    assert saved[-1] == ["eager", "ort"]


def test_ort_note_names_the_provider(tmp_path):
    rep = _run(tmp_path)
    assert "CPUExecutionProvider" in rep["backends"]["ort"]["note"]


def test_parity_only_skips_timing_and_records_tf32(tmp_path):
    out = tmp_path / "bench.json"
    bs.main(TINY + ["--out", str(out), "--backends", "eager,ort", "--parity-only",
                    "--tf32", "off"])
    rep = json.loads(out.read_text())
    assert rep["config"]["parity_only"] is True
    assert rep["config"]["tf32"] == "off"
    assert rep["config"]["allow_tf32_matmul"] is False
    assert rep["config"]["allow_tf32_cudnn"] is False
    for name in ("eager", "ort"):
        e = rep["backends"][name]
        assert e["velocity"] == {} and e["sample"] == {}
        assert set(e["parity_max_abs_err"]) == {"1", "3"}
        assert e["post_projection"]["nvr"] == 0.0
