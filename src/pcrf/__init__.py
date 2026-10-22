"""PC-RF: Physics-Constrained Rectified Flow for climate downscaling and EO super-resolution.

The single source of truth for the model. Submodules:
  physics   three differentiable penalties (divergence, non-negativity, mass) + inference projection
  model     conditional U-Net velocity field, FiLM conditioning, Sentinel-2 encoder
  flow      rectified-flow training loss + ODE sampling
  data      SyntheticDownscaleDataset + ERA5Dataset (+ Sentinel2Dataset)
  metrics   PSNR, SSIM, CRPS, DE, NVR, MCE
  serving   host-side ONNX sampler + physics projection (mirrored by the Rust service)
"""

__version__ = "0.1.0"
