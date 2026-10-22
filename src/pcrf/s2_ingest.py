"""
Sentinel-2 20m->10m super-resolution ingest for PC-RF (fills the s2 table).

Queries the Element84 Earth Search STAC (public, no auth) for low-cloud
`sentinel-2-l2a` scenes, then via rasterio windowed `/vsicurl/` reads (NO
full-scene downloads, NO cloud credentials) cuts cloud-free 10m tiles of
green/red/NIR, scales DN to reflectance, and builds a Wald degrade-and-recover
super-resolution pair per tile: the 10m bands are the target, and an x2
block-averaged 20m copy is the coarse conditioning (b3/b4/b8). Writes per-tile
single-band GeoTIFFs plus train/test scene-split manifest JSONs in the schema
`pcrf.data.Sentinel2Dataset` consumes.

Run on a machine with internet access (cluster compute nodes often have none).
Windowed reads are I/O-bound (network wait, not user-CPU), so they stay light on
a shared login node.

  python -m pcrf.s2_ingest --out data/s2 \
      --max-scenes 16 --tiles-per-scene 200 --tile 128 --cloud-thresh 0.9
"""
from __future__ import annotations
import argparse
import json
import urllib.request
from pathlib import Path

import numpy as np
import rasterio
from rasterio.windows import Window

STAC_URL = "https://earth-search.aws.element84.com/v1/search"
# SCL (Scene Classification Layer) classes treated as clear; drop nodata(0),
# cloud-shadow(3), cloud med/high(8,9), cirrus(10), saturated/dark(1,2).
SCL_CLEAR = {4, 5, 6, 7, 11}          # veg, bare, water, unclassified, snow/ice
BAND_ASSETS = {"b3": "green", "b4": "red", "b8": "nir"}   # all native 10m


def stac_search(bbox, max_scenes, max_cloud, datetime_str):
    body = {
        "collections": ["sentinel-2-l2a"],
        "bbox": list(bbox),
        "limit": max_scenes,
        "query": {"eo:cloud_cover": {"lt": max_cloud}},
        "sortby": [{"field": "properties.eo:cloud_cover", "direction": "asc"}],
    }
    if datetime_str:
        body["datetime"] = datetime_str
    req = urllib.request.Request(
        STAC_URL, data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.load(r).get("features", [])


def _vsi(href):
    return href if href.startswith("/vsicurl/") else "/vsicurl/" + href


def _href(assets, key):
    a = assets.get(key) or assets.get(key.upper()) or {}
    return a.get("href")


def block_avg2(a):
    """x2 block-average, 10m -> 20m coarse."""
    h, w = a.shape
    h2, w2 = h - h % 2, w - w % 2
    a = a[:h2, :w2]
    return a.reshape(h2 // 2, 2, w2 // 2, 2).mean(axis=(1, 3)).astype("float32")


def write_tif(path, arr):
    arr = np.asarray(arr, dtype="float32")[None]           # [1, H, W]
    with rasterio.open(path, "w", driver="GTiff", height=arr.shape[1],
                       width=arr.shape[2], count=1, dtype="float32",
                       compress="deflate") as dst:
        dst.write(arr)


def ingest_scene(feat, tiles_dir, tile, tiles_per_scene, cloud_thresh, rng, sid):
    assets = feat["assets"]
    hrefs = {b: _href(assets, a) for b, a in BAND_ASSETS.items()}
    scl_href = _href(assets, "scl")
    if not all(hrefs.values()) or not scl_href:
        print(f"[s2] scene {sid}: missing assets, skipping", flush=True)
        return []
    srcs = {b: rasterio.open(_vsi(h)) for b, h in hrefs.items()}
    scl = rasterio.open(_vsi(scl_href))
    tiles = []
    try:
        H = min(s.height for s in srcs.values())
        W = min(s.width for s in srcs.values())
        coords = [(i, j) for i in range(H // tile) for j in range(W // tile)]
        rng.shuffle(coords)
        for (i, j) in coords:
            if len(tiles) >= tiles_per_scene:
                break
            r0, c0 = i * tile, j * tile
            # SCL is 20m (half the 10m grid); read the co-registered half-window.
            scl_tile = scl.read(1, window=Window(c0 // 2, r0 // 2,
                                                 tile // 2, tile // 2))
            if np.isin(scl_tile, list(SCL_CLEAR)).mean() < cloud_thresh:
                continue
            win = Window(c0, r0, tile, tile)
            bands, ok = {}, True
            for b, s in srcs.items():
                a = s.read(1, window=win).astype("float32") / 10000.0   # reflectance
                a = np.clip(np.nan_to_num(a), 0.0, 1.5)
                if a.shape != (tile, tile) or a.std() < 1e-4:
                    ok = False
                    break
                bands[b] = a
            if not ok:
                continue
            k = len(tiles)
            rec = {"target": []}
            for b in ("b3", "b4", "b8"):
                tgt10 = bands[b]
                p_c = tiles_dir / f"s{sid:02d}_t{k:04d}_{b}_20m.tif"
                p_t = tiles_dir / f"s{sid:02d}_t{k:04d}_{b}_10m.tif"
                write_tif(p_c, block_avg2(tgt10))       # 20m coarse conditioning
                write_tif(p_t, tgt10)                   # 10m target band
                rec[b] = str(p_c)
                rec["target"].append(str(p_t))
            tiles.append(rec)
        return tiles
    finally:
        for s in srcs.values():
            s.close()
        scl.close()


def main(argv=None):
    ap = argparse.ArgumentParser(description="Sentinel-2 SR ingest for PC-RF")
    ap.add_argument("--out", required=True)
    ap.add_argument("--bbox", nargs=4, type=float, default=[-6.0, 43.0, 6.0, 51.0],
                    help="lon_min lat_min lon_max lat_max (default W. Europe)")
    ap.add_argument("--max-scenes", type=int, default=16, dest="max_scenes")
    ap.add_argument("--tiles-per-scene", type=int, default=200, dest="tiles_per_scene")
    ap.add_argument("--tile", type=int, default=128, help="10m target tile size")
    ap.add_argument("--cloud-thresh", type=float, default=0.9, dest="cloud_thresh",
                    help="min clear-pixel fraction (from SCL) to keep a tile")
    ap.add_argument("--max-cloud", type=float, default=10.0, dest="max_cloud",
                    help="scene-level eo:cloud_cover upper bound (percent)")
    ap.add_argument("--datetime",
                    default="2024-01-01T00:00:00Z/2026-07-01T00:00:00Z",
                    help="RFC3339 interval (Earth Search rejects date-only strings)")
    ap.add_argument("--test-frac", type=float, default=0.2, dest="test_frac")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    rng = np.random.default_rng(args.seed)
    out = Path(args.out)
    tiles_dir = out / "tiles"
    tiles_dir.mkdir(parents=True, exist_ok=True)

    feats = stac_search(args.bbox, args.max_scenes, args.max_cloud, args.datetime)
    print(f"[s2] STAC returned {len(feats)} scenes", flush=True)
    if not feats:
        raise SystemExit("[s2] no scenes returned; widen --bbox / --datetime / --max-cloud")
    n_test = max(1, int(round(len(feats) * args.test_frac)))

    # GDAL /vsicurl tuning: skip directory listing, cache blocks, multiplex.
    env = rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR",
                       CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif",
                       GDAL_HTTP_MULTIPLEX="YES", VSI_CACHE="TRUE")
    train_m, test_m = [], []
    with env:
        for sid, feat in enumerate(feats):
            cc = feat["properties"].get("eo:cloud_cover")
            try:
                tiles = ingest_scene(feat, tiles_dir, args.tile,
                                     args.tiles_per_scene, args.cloud_thresh, rng, sid)
            except Exception as e:                       # one bad scene must not sink the run
                print(f"[s2] scene {sid} ERROR {type(e).__name__}: {e}", flush=True)
                tiles = []
            (test_m if sid < n_test else train_m).extend(tiles)
            print(f"[s2] scene {sid} cloud={cc}% -> {len(tiles)} tiles "
                  f"({'test' if sid < n_test else 'train'})", flush=True)

    (out / "manifest_train.json").write_text(json.dumps(train_m))
    (out / "manifest_test.json").write_text(json.dumps(test_m))
    print(f"[s2] DONE train={len(train_m)} test={len(test_m)} tiles -> {out}")


if __name__ == "__main__":
    main()
