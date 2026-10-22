"""
Datasets for PC-RF.

SyntheticDownscaleDataset  -- runs anywhere, no downloads. Generates
    physically-structured (u, v, precip) fine fields + a coarse (block-mean)
    conditioning field + a fake "satellite" tensor. Use it to smoke-test the
    full training/sampling/metrics pipeline before touching real data.

ERA5Dataset, Sentinel2Dataset -- loaders for real data. KNOWN_ISSUES.md lists what
    the ERA5 loader gets wrong. No Sentinel-2 result was produced with this code.

Every dataset returns a dict:
    {
      "fine":        [C, Hf, Wf]   target fine-resolution field,
      "cond":        [Cc, Hf, Wf]  conditioning image at fine resolution
                                   (upsampled coarse  [+ satellite bands]),
      "coarse_mass": scalar tensor domain-mean precip of the coarse input,
    }
Channel order of "fine": [precip, u, v]  (see physics.ChannelSpec(0,1,2)).
"""
from __future__ import annotations
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F


def _smooth_noise(h, w, scale, gen):
    n = torch.randn(h, w, generator=gen)
    fx = torch.fft.fftfreq(w)[None, :]
    fy = torch.fft.fftfreq(h)[:, None]
    flt = torch.exp(-(fx ** 2 + fy ** 2) * (scale ** 2) * 20.0)
    out = torch.fft.ifft2(torch.fft.fft2(n) * flt).real
    return (out - out.mean()) / (out.std() + 1e-8)


class SyntheticDownscaleDataset(Dataset):
    """Self-contained synthetic downscaling task with physical structure."""

    def __init__(self, n: int = 256, fine: int = 64, factor: int = 4,
                 sat_bands: int = 4, seed: int = 0):
        self.n, self.H, self.W = n, fine, fine
        self.factor = factor
        self.sat_bands = sat_bands
        self.seed = seed

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        gen = torch.Generator().manual_seed(self.seed * 100003 + idx)
        H, W = self.H, self.W

        # divergence-free wind from a stream function
        psi = _smooth_noise(H, W, 7.0, gen)
        u = 0.5 * (torch.roll(psi, -1, 0) - torch.roll(psi, 1, 0))     # d/dy
        v = -0.5 * (torch.roll(psi, -1, 1) - torch.roll(psi, 1, 1))    # d/dx

        # non-negative precip from Gaussian blobs
        yy, xx = torch.meshgrid(torch.arange(H), torch.arange(W), indexing="ij")
        p = torch.zeros(H, W)
        k = 6
        for _ in range(k):
            cy = torch.randint(0, H, (1,), generator=gen).item()
            cx = torch.randint(0, W, (1,), generator=gen).item()
            r = torch.rand(1, generator=gen).item() * 10 + 5
            p += torch.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * r ** 2))

        fine = torch.stack([p, u, v], dim=0)                      # [3,H,W]

        # coarse conditioning = block-mean then upsample back to fine res
        cf = self.factor
        coarse = F.avg_pool2d(fine[None], cf)[0]                  # [3,H/cf,W/cf]
        cond_up = F.interpolate(coarse[None], size=(H, W),
                                mode="bicubic", align_corners=False)[0]
        coarse_mass = coarse[0].mean()                           # precip mass

        # fake satellite bands (stand-in for Sentinel-2 at fine res)
        sat = torch.stack([_smooth_noise(H, W, 3.0, gen)
                           for _ in range(self.sat_bands)], dim=0)
        cond = torch.cat([cond_up, sat], dim=0)                  # [3+sat, H, W]

        return {"fine": fine, "cond": cond, "coarse_mass": coarse_mass}


# ============================================================================
#  REAL-DATA LOADERS  (ERA5 + Sentinel-2)
#
#  Both return the same dict schema as SyntheticDownscaleDataset:
#    {"fine": [C, Hf, Wf], "cond": [Cc, Hf, Wf], "coarse_mass": scalar tensor}
#  Heavy I/O dependencies (xarray / rioxarray) are imported lazily so importing
#  this module (and the synthetic path + the test suite) needs only torch.
# ============================================================================
import numpy as np


def spectral_indices(b3, b4, b8, eps: float = 1e-6):
    """Sentinel-2 indices from green (B3), red (B4), NIR (B8).

    NDVI = (B8 - B4) / (B8 + B4)
    NDWI = (B3 - B8) / (B3 + B8)
    SAVI = 1.5 * (B8 - B4) / (B8 + B4 + 0.5)
    Accepts numpy arrays or torch tensors; returns the same type.
    """
    ndvi = (b8 - b4) / (b8 + b4 + eps)
    ndwi = (b3 - b8) / (b3 + b8 + eps)
    savi = 1.5 * (b8 - b4) / (b8 + b4 + 0.5)
    return ndvi, ndwi, savi


def _open_netcdf(path):
    """Open a single NetCDF (no dask) or a glob / list of them (needs dask)."""
    import xarray as xr
    if isinstance(path, (list, tuple)) or any(c in str(path) for c in "*?["):
        return xr.open_mfdataset(path, combine="by_coords")
    return xr.open_dataset(path)


def write_toy_era5(coarse_path, target_path, n_time=6, hc=16, wc=16,
                   hf=64, wf=64, seed=0):
    """Write tiny synthetic ERA5-like coarse + target NetCDFs (for tests/demos).

    Lets ERA5Dataset be exercised end to end without downloading real data.
    """
    import xarray as xr
    rng = np.random.default_rng(seed)
    time = np.arange(n_time)
    coarse = xr.Dataset(
        {
            "tp": (("time", "lat", "lon"), rng.random((n_time, hc, wc), dtype="f4")),
            "u10": (("time", "lat", "lon"),
                    rng.standard_normal((n_time, hc, wc)).astype("f4")),
            "v10": (("time", "lat", "lon"),
                    rng.standard_normal((n_time, hc, wc)).astype("f4")),
        },
        coords={"time": time, "lat": np.arange(hc), "lon": np.arange(wc)},
    )
    target = xr.Dataset(
        {"precip": (("time", "lat", "lon"),
                    rng.random((n_time, hf, wf), dtype="f4"))},
        coords={"time": time, "lat": np.arange(hf), "lon": np.arange(wf)},
    )
    coarse.to_netcdf(coarse_path)
    target.to_netcdf(target_path)
    return coarse_path, target_path


class ERA5Dataset(Dataset):
    """ERA5 coarse -> fine precipitation downscaling.

    Args:
      coarse_path: NetCDF (or glob) openable by xarray with dims (time, lat, lon)
                   and variables for precip + wind (defaults: tp, u10, v10).
      target_path: high-resolution precip target (IMERG / Stage IV / regional
                   reanalysis) with dims (time, lat, lon).
      split: "train" / "val" / "test" (chronological split by shared timestamps).
      fine: target grid side length (the target is resized to fine x fine).

    Returns the SyntheticDownscaleDataset schema. ERA5 has no fine-resolution wind
    target, so the wind channels of `fine` carry the upsampled coarse wind; the
    divergence penalty regularizes them at the fine grid. Normalization stats are
    computed once on the train split.
    """

    def __init__(self, coarse_path, target_path, split="train", fine=128,
                 var_precip="tp", var_u="u10", var_v="v10",
                 target_precip="precip", splits=(0.8, 0.1, 0.1),
                 normalize=True, sat_bands=0):
        self.fine = fine
        self.sat_bands = sat_bands
        self.var_precip, self.var_u, self.var_v = var_precip, var_u, var_v
        self.target_precip = target_precip

        self.coarse = _open_netcdf(coarse_path)
        self.target = _open_netcdf(target_path)
        times = np.intersect1d(np.asarray(self.coarse["time"].values),
                               np.asarray(self.target["time"].values))
        if times.size == 0:
            raise ValueError("no overlapping timestamps between coarse and target")

        n = times.size
        a, b = int(splits[0] * n), int(splits[0] * n) + int(splits[1] * n)
        sl = {"train": slice(0, a), "val": slice(a, b), "test": slice(b, n)}[split]
        self.times = times[sl]

        self._stats = None
        if normalize:
            tr = times[slice(0, max(a, 1))]
            self._stats = {
                "p": self._mean_std(self.coarse[var_precip].sel(time=tr)),
                "u": self._mean_std(self.coarse[var_u].sel(time=tr)),
                "v": self._mean_std(self.coarse[var_v].sel(time=tr)),
            }

    @staticmethod
    def _mean_std(da):
        v = np.asarray(da.values, dtype="float32")
        return float(np.nanmean(v)), float(np.nanstd(v) + 1e-6)

    def __len__(self):
        return int(self.times.size)

    def __getitem__(self, idx):
        t = self.times[idx]
        cz = self.coarse.sel(time=t)
        pc = np.asarray(cz[self.var_precip].values, dtype="float32")
        uc = np.asarray(cz[self.var_u].values, dtype="float32")
        vc = np.asarray(cz[self.var_v].values, dtype="float32")
        tgt = np.asarray(self.target[self.target_precip].sel(time=t).values,
                         dtype="float32")

        # ERA5-Land is a land-only product: ocean / out-of-mask cells are NaN.
        # Fill with 0 (no reported precip) BEFORE normalization or interpolation,
        # otherwise the NaN target poisons the regression field and L_RF -> NaN.
        # Guard the coarse vars too (currently clean, but cheap insurance).
        pc = np.nan_to_num(pc, nan=0.0, posinf=0.0, neginf=0.0)
        uc = np.nan_to_num(uc, nan=0.0, posinf=0.0, neginf=0.0)
        vc = np.nan_to_num(vc, nan=0.0, posinf=0.0, neginf=0.0)
        tgt = np.nan_to_num(tgt, nan=0.0, posinf=0.0, neginf=0.0)

        coarse_mass = float(np.mean(pc))   # physical (>= 0), pre-normalization
        if self._stats:
            _, sp = self._stats["p"]
            mu, su = self._stats["u"]
            mv, sv = self._stats["v"]
            # Precip: positive-scale (divide by std, NO mean subtraction) so
            # coarse_mass stays >= 0 and project_physics' multiplicative mass
            # rescale (p * tgt/cur) cannot flip the precip field negative.
            # Wind stays standardized.
            pc = pc / sp
            uc = (uc - mu) / su
            vc = (vc - mv) / sv
            tgt = tgt / sp
            coarse_mass = coarse_mass / sp

        hf = wf = self.fine

        def to_fine(a, mode):
            x = torch.from_numpy(np.ascontiguousarray(a))[None, None]
            return F.interpolate(x, size=(hf, wf), mode=mode,
                                 align_corners=False)[0, 0]

        p_up, u_up, v_up = (to_fine(pc, "bicubic"), to_fine(uc, "bicubic"),
                            to_fine(vc, "bicubic"))
        p_fine = to_fine(tgt, "bilinear")

        fine = torch.stack([p_fine, u_up, v_up], dim=0)
        cond_layers = [p_up, u_up, v_up]
        cond_layers += [torch.zeros(hf, wf) for _ in range(self.sat_bands)]
        cond = torch.stack(cond_layers, dim=0)
        return {"fine": fine, "cond": cond,
                "coarse_mass": torch.tensor(coarse_mass, dtype=torch.float32)}


class Sentinel2Dataset(Dataset):
    """Sentinel-2 20m -> 10m super-resolution with NDVI/NDWI/SAVI indices.

    `scenes` is a list of dicts with paths to single-band rasters (GeoTIFF / JP2),
    read via rasterio (each raster a single-band reflectance tile, e.g. produced by
    pcrf.s2_ingest, already scaled to [0, ~1]):
        {"b3": green20m, "b4": red20m, "b8": nir20m, "target": [10m band paths...]}

    Returns:
      fine        : [len(target), Hf, Wf]   the 10m super-resolution target bands
      cond        : [3 + 3, Hf, Wf]         upsampled 20m (B3,B4,B8) + NDVI/NDWI/SAVI
      coarse_mass : tensor(0.0)             unused for pure SR

    Fidelity metrics (SSIM, PSNR, SAM) are what apply here; the optical constraint
    is non-negativity of reflectance, enforced at inference for the PC-RF arm.

    normalize defaults to False: the ingest already emits reflectance in [0, 1], and
    per-band z-scoring would distort the spectral vectors that SAM and the
    non-negativity projection depend on.
    """

    def __init__(self, scenes, tile=256, normalize=False):
        self.scenes = list(scenes)
        self.tile = tile
        self.normalize = normalize

    def __len__(self):
        return len(self.scenes)

    def __getitem__(self, idx):
        import rasterio
        sc = self.scenes[idx]

        def read(path):
            with rasterio.open(path) as ds:
                a = ds.read(1).astype("float32")
            return np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)

        b3, b4, b8 = read(sc["b3"]), read(sc["b4"]), read(sc["b8"])
        ndvi, ndwi, savi = spectral_indices(b3, b4, b8)
        target = np.stack([read(p) for p in sc["target"]], axis=0).astype("float32")

        hf = wf = self.tile

        def to_fine(a):
            x = torch.from_numpy(np.ascontiguousarray(a))[None, None]
            return F.interpolate(x, size=(hf, wf), mode="bilinear",
                                 align_corners=False)[0, 0]

        if self.normalize:
            def norm(a):
                return (a - a.mean()) / (a.std() + 1e-6)
            b3, b4, b8 = norm(b3), norm(b4), norm(b8)
            target = np.stack([(t - t.mean()) / (t.std() + 1e-6) for t in target], 0)

        cond = torch.stack([to_fine(b3), to_fine(b4), to_fine(b8),
                            to_fine(ndvi), to_fine(ndwi), to_fine(savi)], dim=0)
        fine = torch.stack([to_fine(t) for t in target], dim=0)
        return {"fine": fine, "cond": cond,
                "coarse_mass": torch.tensor(0.0, dtype=torch.float32)}
