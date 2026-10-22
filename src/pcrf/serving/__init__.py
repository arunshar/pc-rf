"""Serving-time math: the host-side ONNX ODE sampler and physics projection.

`pcrf.serving.math` is the projection and sampler math used by the Python serving
paths (FastAPI, Ray Serve). A Rust service that is not part of this repository
implemented the same steps.
"""
