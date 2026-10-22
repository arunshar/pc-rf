"""
Ray Serve deployment of the PC-RF downscaler. This is a serving
demo: it runs the SAME host-side sampler + physics projection as the Rust
inference service (both call pcrf.serving.math.sample_onnx), so the two runtimes
produce metrics that agree within tolerance (NVR==0, MCE<1e-4) ("same physics,
two runtimes"). Ray Serve adds autoscaling and Python-native batching on
managed Ray services.

Local:
  PCRF_ONNX=artifacts/pcrf.onnx serve run apps.training.serve_ray:app
  curl -s localhost:8000/ -H 'content-type: application/json' \
       -d '{"ensemble":8,"steps":20,"coarse_mass":0.1}' | jq '.metrics'

Managed Ray services: point the service at the `app` object in this module.

Throughput / latency / cost knobs (all opt-in; defaults reproduce the prior
single-replica CPU behavior exactly):

  PCRF_SERVE_BATCH=1            enable @serve.batch dynamic batching so concurrent
                               downscale requests coalesce into one actor call.
  PCRF_SERVE_MAX_BATCH=8       max requests coalesced per batch (default 8).
  PCRF_SERVE_BATCH_WAIT_S=0.01 batch_wait_timeout_s, the coalesce window (s).
  PCRF_SERVE_GPUS=0.25         fractional GPU per replica so several tiny (7.6M)
                               replicas pack onto one physical GPU. "0" => CPU.
  PCRF_SERVE_MIN_REPLICAS=1    autoscaling lower bound.
  PCRF_SERVE_MAX_REPLICAS=8    autoscaling upper bound (the 1..8 ramp slide).
  PCRF_SERVE_TARGET_REQUESTS=4 target_ongoing_requests per replica.
"""
from __future__ import annotations
import json
import os
from pathlib import Path

from ray import serve
from starlette.requests import Request

from pcrf.serving import math as sm


def _env_float(name: str, default: str) -> float:
    return float(os.environ.get(name, default))


def _env_int(name: str, default: str) -> int:
    return int(os.environ.get(name, default))


def _gpus_enabled(val: float) -> bool:
    return val > 0.0


# Read serving knobs once at import so the decorator and the @serve.batch
# wrapper are configured before the deployment class is built. All have
# behavior-preserving defaults.
_SERVE_GPUS = _env_float("PCRF_SERVE_GPUS", "0")
_BATCH_ENABLED = os.environ.get("PCRF_SERVE_BATCH", "0") not in ("0", "", "false", "False")
_MAX_BATCH = _env_int("PCRF_SERVE_MAX_BATCH", "8")
_BATCH_WAIT_S = _env_float("PCRF_SERVE_BATCH_WAIT_S", "0.01")
_MIN_REPLICAS = _env_int("PCRF_SERVE_MIN_REPLICAS", "1")
_MAX_REPLICAS = _env_int("PCRF_SERVE_MAX_REPLICAS", "8")
_TARGET_REQUESTS = _env_int("PCRF_SERVE_TARGET_REQUESTS", "4")


@serve.deployment(
    autoscaling_config={
        "min_replicas": _MIN_REPLICAS,
        "max_replicas": _MAX_REPLICAS,
        # Ray 2.30+ (deployed image 2.55.1) serve autoscaling key. With
        # fractional GPUs the load-ramp demo scales replicas 1..max while
        # several replicas share one GPU, so the cost-per-request slide
        # reflects GPU packing, not whole GPUs.
        "target_ongoing_requests": _TARGET_REQUESTS,
    },
    # Fractional-GPU packing: the velocity net is tiny (7.6M params), so
    # num_gpus=0.25 lets up to 4 replicas share one physical GPU. Gated by
    # PCRF_SERVE_GPUS; "0" keeps the CPU-only default working unchanged.
    ray_actor_options={"num_gpus": _SERVE_GPUS},
)
class PCRFDownscaler:
    def __init__(self, onnx_path: str):
        import onnxruntime as ort

        providers = ["CPUExecutionProvider"]
        if _gpus_enabled(_SERVE_GPUS):
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        # Single shared session per replica. Each coalesced request is run
        # sequentially through this session inside the batch handler; the
        # session is reused, not driven by concurrent Run() calls.
        self.sess = ort.InferenceSession(onnx_path, providers=providers)
        meta = json.loads(Path(onnx_path).with_suffix(".json").read_text())
        self.tile = int(meta["tile"])
        self.in_ch = int(meta["in_ch"])
        self.cond_ch = int(meta["cond_ch"])
        self.backend = "onnxruntime-" + ("cuda" if "CUDA" in providers[0] else "cpu")
        self._batch_enabled = _BATCH_ENABLED
        if self._batch_enabled:
            # Configure the @serve.batch wrapper from env at construction time.
            # Ray 2.30+ (deployed image 2.55.1) exposes set_max_batch_size /
            # set_batch_wait_timeout_s on the wrapped method so the window is
            # tunable without re-decorating.
            self._batched_run.set_max_batch_size(_MAX_BATCH)
            self._batched_run.set_batch_wait_timeout_s(_BATCH_WAIT_S)

    def _run(self, body: dict) -> dict:
        """Single-request sample + projection. Unchanged numeric path: each
        request keeps its own ensemble size, steps, coarse_mass and seed, so
        results match the prior implementation and the Rust runtime to within
        tolerance (NVR==0, MCE<1e-4) regardless of how requests are coalesced."""
        mean, metrics = sm.sample_onnx(
            self.sess,
            B=int(body.get("ensemble", 8)),
            tile=self.tile,
            in_ch=self.in_ch,
            cond_ch=self.cond_ch,
            steps=int(body.get("steps", 20)),
            coarse_mass=float(body.get("coarse_mass", 0.1)),
            project=bool(body.get("project", True)),
        )
        out = {
            "shape": list(mean.shape),
            "metrics": {k: float(v) for k, v in metrics.items()},
            "backend": self.backend,
        }
        if body.get("return_field", False):
            out["field"] = mean.astype(float).ravel().tolist()
        return out

    # --- dynamic batching -----------------------------------------------------
    # @serve.batch coalesces concurrent calls into one list. Defaults here are
    # placeholders; the real max_batch_size / batch_wait_timeout_s are set from
    # env in __init__ so the window is tunable for the load-ramp demo.
    #
    # NOTE on numerics: each PC-RF request is an independent ensemble sample
    # with its own ensemble size, step count, coarse_mass and fixed seed.
    # Fusing requests into one stacked ONNX tensor would change the per-request
    # RNG draw and therefore the output, so we deliberately do NOT do that.
    # Instead the batch handler runs each coalesced request through the shared
    # session sequentially (one _run per body) and returns a list aligned to the
    # input order. The win is replica-level concurrency and fewer Serve/asyncio
    # round-trips, not intra-batch parallel Run() calls; the per-request numeric
    # path is unchanged.
    # TODO(perf): a fully fused fast-path that stacks same-shape, no-custom-cond,
    # same-steps requests into one (sum(B), C, tile, tile) tensor per Euler step
    # is possible and would cut kernel launches further, but only if we also
    # make the host-side RNG seed per-request-stable under stacking. Left out
    # here because it cannot be verified on the login node and would otherwise
    # silently change outputs.
    @serve.batch(max_batch_size=8, batch_wait_timeout_s=0.01)
    async def _batched_run(self, bodies: list) -> list:
        return [self._run(b) for b in bodies]

    async def __call__(self, request: Request):
        body = await request.json()
        if self._batch_enabled:
            # Returns this request's slice of the coalesced batch.
            return await self._batched_run(body)
        return self._run(body)


app = PCRFDownscaler.bind(os.environ.get("PCRF_ONNX", "artifacts/pcrf.onnx"))
