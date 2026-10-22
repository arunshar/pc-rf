"""
Pure request/response schema helpers for the PC-RF Ray Serve downscaler.

These functions are dependency-free (stdlib only) so they can be unit-tested on
the login node without ray / onnxruntime / torch, and reused by a deploy smoke
script to validate that a live /downscale response is well formed (HTTP 200 plus
the three physics metrics within tolerance).

Contract (matches apps/training/serve_ray.py PCRFDownscaler._run):

  request  JSON object, all keys optional, defaults shown:
    {
      "ensemble":     8,      # int  >= 1   ensemble members
      "steps":        20,     # int  >= 1   Euler steps
      "coarse_mass":  0.1,    # float >= 0  target coarse precip mass
      "project":      true,   # bool        apply physics projection
      "return_field": false   # bool        also return the raw mean field
    }

  response JSON object:
    {
      "shape":   [C, H, W],            # list[int], length 3
      "metrics": {"DE": .., "NVR": .., "MCE": ..},   # all floats
      "backend": "onnxruntime-cpu",    # str, "onnxruntime-<cpu|cuda>"
      "field":   [..]                  # optional list[float], only if return_field
    }

This module intentionally does NOT import the deployment; it just encodes the
wire contract so a smoke test can assert against it.
"""
from __future__ import annotations

from typing import Any, Dict, Tuple

# Default request body. A POST with an empty JSON object {} is valid and means
# "use all defaults", matching the body.get(..., default) reads in serve_ray._run.
DEFAULT_REQUEST: Dict[str, Any] = {
    "ensemble": 8,
    "steps": 20,
    "coarse_mass": 0.1,
    "project": True,
    "return_field": False,
}

# Metric -> tolerance used by the deploy smoke. The serving path drives NVR to 0
# and MCE below 1e-4 (same physics as the Rust runtime); DE is finite but not
# bounded a priori, so it is only checked for finiteness.
METRIC_KEYS: Tuple[str, ...] = ("DE", "NVR", "MCE")
SMOKE_TOLERANCE: Dict[str, float] = {"NVR": 0.0, "MCE": 1e-4}


def normalize_request(body: Dict[str, Any] | None) -> Dict[str, Any]:
    """Fill a (possibly partial) request body with defaults and coerce types.

    Pure: returns a new dict, does not mutate the input. Raises ValueError on a
    value that cannot be coerced or that is out of range, so a gateway can reject
    a bad request with a 4xx instead of letting it reach the sampler.
    """
    if body is None:
        body = {}
    if not isinstance(body, dict):
        raise ValueError(f"request body must be a JSON object, got {type(body).__name__}")

    out = dict(DEFAULT_REQUEST)
    for key in ("ensemble", "steps"):
        if key in body:
            try:
                out[key] = int(body[key])
            except (TypeError, ValueError):
                raise ValueError(f"{key} must be an int, got {body[key]!r}")
            if out[key] < 1:
                raise ValueError(f"{key} must be >= 1, got {out[key]}")
    if "coarse_mass" in body:
        try:
            out["coarse_mass"] = float(body["coarse_mass"])
        except (TypeError, ValueError):
            raise ValueError(f"coarse_mass must be a float, got {body['coarse_mass']!r}")
        if out["coarse_mass"] < 0.0:
            raise ValueError(f"coarse_mass must be >= 0, got {out['coarse_mass']}")
    for key in ("project", "return_field"):
        if key in body:
            out[key] = bool(body[key])
    return out


def validate_response(resp: Dict[str, Any], expect_field: bool = False) -> Dict[str, Any]:
    """Validate a /downscale response dict against the wire contract.

    Returns the response unchanged on success; raises ValueError otherwise. Used
    by the deploy smoke after a live POST so a malformed 200 still fails loudly.
    """
    if not isinstance(resp, dict):
        raise ValueError(f"response must be a JSON object, got {type(resp).__name__}")

    shape = resp.get("shape")
    if not isinstance(shape, (list, tuple)) or len(shape) != 3:
        raise ValueError(f"shape must be a length-3 list [C,H,W], got {shape!r}")
    if not all(isinstance(d, int) and d > 0 for d in shape):
        raise ValueError(f"shape entries must be positive ints, got {shape!r}")

    metrics = resp.get("metrics")
    if not isinstance(metrics, dict):
        raise ValueError(f"metrics must be a JSON object, got {metrics!r}")
    for k in METRIC_KEYS:
        if k not in metrics:
            raise ValueError(f"metrics missing key {k!r}; got {sorted(metrics)}")
        if not isinstance(metrics[k], (int, float)):
            raise ValueError(f"metric {k} must be a number, got {metrics[k]!r}")

    backend = resp.get("backend")
    if not isinstance(backend, str) or not backend.startswith("onnxruntime-"):
        raise ValueError(f"backend must be 'onnxruntime-*', got {backend!r}")

    if expect_field:
        field = resp.get("field")
        if not isinstance(field, (list, tuple)):
            raise ValueError("return_field was requested but 'field' is missing")
        expected = shape[0] * shape[1] * shape[2]
        if len(field) != expected:
            raise ValueError(
                f"field length {len(field)} != C*H*W ({expected})"
            )
    return resp


def smoke_metrics_ok(metrics: Dict[str, Any], tolerance: Dict[str, float] | None = None) -> bool:
    """True if the physics metrics meet the smoke tolerance (NVR==0, MCE<tol).

    DE is only required to be finite. Pure boolean check so the smoke script can
    branch on it and the unit test can assert both pass and fail cases.
    """
    import math

    tol = SMOKE_TOLERANCE if tolerance is None else tolerance
    de = metrics.get("DE")
    if de is None or not math.isfinite(float(de)):
        return False
    nvr = float(metrics.get("NVR", 1.0))
    if nvr > tol.get("NVR", 0.0):
        return False
    mce = float(metrics.get("MCE", 1.0))
    if mce >= tol.get("MCE", 1e-4):
        return False
    return True
