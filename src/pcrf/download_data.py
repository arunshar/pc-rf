"""
Download a real ERA5 coarse + ERA5-Land target pair from the Copernicus Climate
Data Store (CDS), ready for `pcrf-train --data era5`. One credential, two datasets:

  coarse  = ERA5 single levels, 0.25 deg  (tp, u10, v10)   -> coarse.nc
  target  = ERA5-Land,          0.1  deg  (precip)          -> target.nc

ERA5-Land (0.1 deg) lives on the same CDS as ERA5, so you need only one account.
Its total precipitation is interpolated from ERA5 and uses a different
accumulation window, so it is not an independent high-resolution target. Read
KNOWN_ISSUES.md before you use it. For an independent target you can swap in
IMERG (NASA Earthdata, via `earthaccess`) or Stage IV, and point `--target` at
that file.

One-time setup:
  1. Free CDS account: https://cds.climate.copernicus.eu/
  2. Put your key in ~/.cdsapirc :
        url: https://cds.climate.copernicus.eu/api
        key: <your-key>
  3. On the CDS website, accept the licenses for "ERA5 hourly data on single
     levels" and "ERA5-Land hourly data" once.
  4. pip install -e ".[data]"   (installs cdsapi, xarray and netCDF4)

Usage (a small JJA 2022 slice over Western Europe):
  python download_data.py --year 2022 --months 6 7 8 \
      --area 50 -10 40 5 --out-dir data/era5
  # --area is North West South East, in degrees

Then train (GPU recommended for the full run):
  pcrf-train --data era5 --coarse data/era5/coarse.nc \
      --target data/era5/target.nc --fine 128 --epochs 200 --base 64 --out runs/pcrf_full

NOTE: this script is written against the current CDS API but cannot be tested
without your credentials; if a variable name or option has changed, the CDS error
message names the fix.
"""
from __future__ import annotations
import argparse
from pathlib import Path


def _open_cds(path):
    """Open a CDS download that may be a raw NetCDF OR a .zip of NetCDFs.
    The new CADS wraps results in a zip: ERA5 single-levels splits the instant and
    accumulated streams into two .nc files; ERA5-Land returns data_0.nc. Extract,
    merge the members, and return an in-memory Dataset so the temp files can go."""
    import os, shutil, tempfile, zipfile
    import xarray as xr
    p = str(path)
    if not zipfile.is_zipfile(p):
        ds = xr.open_dataset(p)
    else:
        tmpd = tempfile.mkdtemp(dir=os.path.dirname(p) or ".")
        with zipfile.ZipFile(p) as z:
            members = [m for m in z.namelist() if m.endswith((".nc", ".netcdf"))]
            z.extractall(tmpd, members)
        ncs = sorted(os.path.join(tmpd, m) for m in members)
        if not ncs:
            raise RuntimeError(f"no NetCDF found inside CDS zip {p}")
        if len(ncs) == 1:
            ds = xr.open_dataset(ncs[0]).load()
        else:
            ds = xr.merge([xr.open_dataset(n) for n in ncs], compat="override").load()
        shutil.rmtree(tmpd, ignore_errors=True)
    # Canonicalize the new-CADS schema to what ERA5Dataset expects: it indexes by a
    # coord named `time` and does not want the scalar `number`/`expver` coords.
    if "valid_time" in ds.variables:
        ds = ds.rename({"valid_time": "time"})
    drop = [c for c in ("number", "expver") if c in ds.variables]
    if drop:
        ds = ds.drop_vars(drop)
    return ds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, default=2022)
    ap.add_argument("--months", type=int, nargs="+", default=[6, 7, 8])
    ap.add_argument("--days", type=int, nargs="+", default=list(range(1, 32)))
    ap.add_argument("--hours", nargs="+",
                    default=["00:00", "06:00", "12:00", "18:00"])
    ap.add_argument("--area", type=float, nargs=4, default=[50, -10, 40, 5],
                    metavar=("N", "W", "S", "E"))
    ap.add_argument("--out-dir", default="data/era5")
    args = ap.parse_args()

    import cdsapi
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    months = [f"{m:02d}" for m in args.months]
    days = [f"{d:02d}" for d in args.days]
    client = cdsapi.Client()

    coarse_nc = out / "coarse.nc"
    coarse_raw = out / "_coarse_raw"
    print(f"[era5 single-levels 0.25deg] -> {coarse_nc}")
    client.retrieve(
        "reanalysis-era5-single-levels",
        {
            "product_type": "reanalysis",
            "variable": ["total_precipitation",
                         "10m_u_component_of_wind",
                         "10m_v_component_of_wind"],
            "year": str(args.year), "month": months, "day": days,
            "time": args.hours, "area": args.area, "data_format": "netcdf",
        },
        str(coarse_raw),
    )
    # CADS returns a zip (instant + accum streams); merge to one NetCDF.
    cds = _open_cds(coarse_raw)
    cds.to_netcdf(coarse_nc)
    cds.close()
    coarse_raw.unlink(missing_ok=True)

    target_nc = out / "target.nc"
    raw = out / "_era5land_tp.nc"
    print(f"[era5-land 0.1deg] -> {target_nc}")
    client.retrieve(
        "reanalysis-era5-land",
        {
            "variable": ["total_precipitation"],
            "year": str(args.year), "month": months, "day": days,
            "time": args.hours, "area": args.area, "data_format": "netcdf",
        },
        str(raw),
    )

    # Rename tp -> precip so the file is drop-in for ERA5Dataset defaults.
    ds = _open_cds(raw)
    var = "tp" if "tp" in ds.data_vars else list(ds.data_vars)[0]
    ds.rename({var: "precip"}).to_netcdf(target_nc)
    ds.close()
    raw.unlink(missing_ok=True)

    print("\ndone. Train with:")
    print(f"  pcrf-train --data era5 --coarse {coarse_nc} "
          f"--target {target_nc} --fine 128 --epochs 200 --base 64 --out runs/pcrf_full")


if __name__ == "__main__":
    main()
