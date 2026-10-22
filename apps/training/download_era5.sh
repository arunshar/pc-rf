#!/bin/bash
# ============================================================================
#  Download the ERA5 and ERA5-Land corpus for PC-RF over Western Europe.
#  Each year of data takes about 1 to 2 GB. Run it on a machine with internet access.
#
#    bash apps/training/download_era5.sh
#
#  One-time prerequisites:
#    1) Create a free Copernicus CDS account at https://cds.climate.copernicus.eu.
#    2) Put your key in ~/.cdsapirc. The CDS profile page shows the exact two
#       lines, a `url:` and a `key:`. Never commit this file.
#    3) Accept the ERA5 and ERA5-Land licences on the CDS website, or the API
#       returns a silent 403.
#    4) pip install -e ".[data]"   (installs cdsapi, xarray, netCDF4)
#
#  Read KNOWN_ISSUES.md before using the result. ERA5-Land total precipitation
#  is interpolated from ERA5, and the two products use different accumulation
#  conventions that download_data.py does not reconcile.
# ============================================================================
set -euo pipefail

REPO=${REPO:-$(cd "$(dirname "$0")/../.." && pwd)}
DATA_ROOT=${DATA_ROOT:-$REPO/data}
YEARS=${YEARS:-2019 2020 2021}
AREA=${AREA:-50 -10 40 5}                    # North West South East (W. Europe)
HOURS=${HOURS:-00:00 06:00 12:00 18:00}

mkdir -p "$DATA_ROOT"
for Y in $YEARS; do
  OUT="$DATA_ROOT/era5_$Y"
  if [ -f "$OUT/coarse.nc" ] && [ -f "$OUT/target.nc" ]; then
    echo "[$Y] already present -> $OUT (skipping)"
    continue
  fi
  echo "[$Y] downloading -> $OUT"
  python "$REPO/src/pcrf/download_data.py" \
      --year "$Y" --months 1 2 3 4 5 6 7 8 9 10 11 12 \
      --hours $HOURS --area $AREA --out-dir "$OUT"
done

echo
echo "done. Train on the whole corpus with the globs:"
echo "  --coarse '$DATA_ROOT/era5_*/coarse.nc' --target '$DATA_ROOT/era5_*/target.nc'"
