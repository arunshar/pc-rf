"""Unit tests for the PC-RF serve deploy helpers (apps/training/serve_schema.py).

These are pure, dependency-free checks (no ray / onnxruntime / torch / network),
so they run anywhere. They pin the request/response wire contract that a
deploy smoke test and the optional FastAPI ingress (serve_ingress.py) rely on.

conftest.py puts apps/training on sys.path, so `import serve_schema` resolves.
"""
import pytest

import serve_schema as ss


# --------------------------------------------------------------------------- #
# normalize_request: defaults, coercion, range validation
# --------------------------------------------------------------------------- #
def test_normalize_empty_body_uses_defaults():
    out = ss.normalize_request({})
    assert out == ss.DEFAULT_REQUEST
    # returns a copy, not the module-level default object
    assert out is not ss.DEFAULT_REQUEST


def test_normalize_none_body_uses_defaults():
    assert ss.normalize_request(None) == ss.DEFAULT_REQUEST


def test_normalize_coerces_types():
    out = ss.normalize_request({"ensemble": "4", "steps": "10",
                                "coarse_mass": "0.2", "project": 0,
                                "return_field": 1})
    assert out["ensemble"] == 4 and isinstance(out["ensemble"], int)
    assert out["steps"] == 10
    assert out["coarse_mass"] == pytest.approx(0.2)
    assert out["project"] is False
    assert out["return_field"] is True


def test_normalize_partial_body_keeps_other_defaults():
    out = ss.normalize_request({"ensemble": 16})
    assert out["ensemble"] == 16
    assert out["steps"] == ss.DEFAULT_REQUEST["steps"]
    assert out["coarse_mass"] == ss.DEFAULT_REQUEST["coarse_mass"]


@pytest.mark.parametrize("bad", [
    {"ensemble": 0},
    {"steps": -1},
    {"ensemble": "abc"},
    {"coarse_mass": -0.5},
    {"coarse_mass": "nope"},
])
def test_normalize_rejects_bad_values(bad):
    with pytest.raises(ValueError):
        ss.normalize_request(bad)


def test_normalize_rejects_non_object_body():
    with pytest.raises(ValueError):
        ss.normalize_request([1, 2, 3])


# --------------------------------------------------------------------------- #
# validate_response: shape / metrics / backend / optional field
# --------------------------------------------------------------------------- #
def _good_response(field=None):
    resp = {
        "shape": [3, 8, 8],
        "metrics": {"DE": 0.01, "NVR": 0.0, "MCE": 1e-6},
        "backend": "onnxruntime-cpu",
    }
    if field is not None:
        resp["field"] = field
    return resp


def test_validate_response_accepts_good():
    resp = _good_response()
    assert ss.validate_response(resp) is resp


def test_validate_response_accepts_field_when_expected():
    resp = _good_response(field=[0.0] * (3 * 8 * 8))
    assert ss.validate_response(resp, expect_field=True) is resp


def test_validate_response_rejects_wrong_field_length():
    resp = _good_response(field=[0.0] * 5)
    with pytest.raises(ValueError):
        ss.validate_response(resp, expect_field=True)


def test_validate_response_rejects_missing_field_when_expected():
    with pytest.raises(ValueError):
        ss.validate_response(_good_response(), expect_field=True)


@pytest.mark.parametrize("mutate", [
    lambda r: r.pop("shape"),
    lambda r: r.__setitem__("shape", [3, 8]),          # wrong length
    lambda r: r.__setitem__("shape", [3, 0, 8]),       # non-positive dim
    lambda r: r["metrics"].pop("NVR"),                 # missing metric
    lambda r: r.__setitem__("metrics", "notdict"),
    lambda r: r.__setitem__("backend", "torch-cpu"),   # wrong backend prefix
    lambda r: r.__setitem__("backend", 42),
])
def test_validate_response_rejects_bad(mutate):
    resp = _good_response()
    mutate(resp)
    with pytest.raises(ValueError):
        ss.validate_response(resp)


def test_validate_response_rejects_non_object():
    with pytest.raises(ValueError):
        ss.validate_response("nope")


# --------------------------------------------------------------------------- #
# smoke_metrics_ok: tolerance gate
# --------------------------------------------------------------------------- #
def test_smoke_metrics_pass():
    assert ss.smoke_metrics_ok({"DE": 0.5, "NVR": 0.0, "MCE": 1e-6}) is True


def test_smoke_metrics_fail_on_nvr():
    assert ss.smoke_metrics_ok({"DE": 0.5, "NVR": 0.01, "MCE": 1e-6}) is False


def test_smoke_metrics_fail_on_mce():
    assert ss.smoke_metrics_ok({"DE": 0.5, "NVR": 0.0, "MCE": 1.0}) is False


def test_smoke_metrics_fail_on_nonfinite_de():
    assert ss.smoke_metrics_ok({"DE": float("inf"), "NVR": 0.0, "MCE": 1e-6}) is False


def test_smoke_metrics_custom_tolerance():
    # a looser NVR tolerance lets a small positive NVR pass
    m = {"DE": 0.5, "NVR": 0.005, "MCE": 1e-6}
    assert ss.smoke_metrics_ok(m, tolerance={"NVR": 0.01, "MCE": 1e-4}) is True
