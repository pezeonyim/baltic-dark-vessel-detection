"""Daily Baltic SAR job: search new Sentinel-1 scenes, download, detect, match with AIS, save.

    maritime sar land                  # once: OSM land polygons clipped to the region
    maritime sar search --hours 48     # what is available
    maritime sar run --hours 36        # process everything new (what the nightly job runs)

Outputs: data/processed/<region>/sar/detections/date=YYYY-MM-DD/<product>.parquet and a manifest.jsonl line
per scene (timings, counts) so already-processed products are skipped.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

from maritime.paths import DATA, PROCESSED, REFERENCE, ROOT

LAND_URL = "https://osmdata.openstreetmap.de/download/land-polygons-split-4326.zip"
MODEL_DIR = DATA / "models"
DEFAULT_MODELS = ("detector_VV+VH+hn+manual@2.pt",)   # best single seed on held-out scenes


def scene_medians_db(img, step: int = 16) -> dict:
    """Median VV/VH of valid pixels (mostly sea) in dB, from the quantised image: a calibration sanity check
    against the xView3 training data (sea VV -26..-17.5 dB, VH ~ -28 dB)."""
    from maritime.sar.xview3 import DB_MAX, DB_MIN
    out = {}
    for b, k in ((0, "vv_db"), (1, "vh_db")):
        q = img[b, ::step, ::step]
        q = q[q > 0]
        out[k] = round(float((np.median(q) - 1) / 254 * (DB_MAX - DB_MIN) + DB_MIN), 1) if q.size else float("nan")
    return out


def sar_dir(region: str) -> Path:
    return PROCESSED / region / "sar"


def land_path(region: str) -> Path:
    return REFERENCE / f"{region}_land.parquet"


def fetch_land(region, log=print) -> Path:
    import requests

    from maritime.sar.scene_pipeline import prepare_land
    b = region.bbox
    raw = DATA / "raw" / "osm" / "land-polygons-split-4326.zip"
    raw.parent.mkdir(parents=True, exist_ok=True)
    if not raw.exists():
        log(f"downloading {LAND_URL} (~900 MB, once)")
        tmp = raw.with_suffix(".part")
        with requests.get(LAND_URL, stream=True, timeout=120) as r:
            r.raise_for_status()
            with open(tmp, "wb") as f:
                for chunk in r.iter_content(1 << 22):
                    f.write(chunk)
        tmp.replace(raw)
    out = land_path(region.name)
    n = prepare_land(raw, (b.min_lon, b.min_lat, b.max_lon, b.max_lat), out)
    log(f"{n:,} land polygons -> {out} ({out.stat().st_size / 1e6:.0f} MB)")
    raw.unlink()
    return out


def search_region(region, hours: float, end: pd.Timestamp | None = None) -> pd.DataFrame:
    from maritime.sar.sentinel1 import bbox_wkt, search
    b = region.bbox
    end = end or pd.Timestamp.now("UTC").tz_localize(None)
    start = end - pd.Timedelta(hours=hours)
    iso = lambda t: t.strftime("%Y-%m-%dT%H:%M:%S.000Z")   # noqa: E731
    return search(bbox_wkt(b.min_lon, b.min_lat, b.max_lon, b.max_lat), iso(start), iso(end))


MIN_SEA_KM2 = 2_000      # a full IW slice is ~250 x 170 km = ~42,000 km2; below this it is essentially land


def add_sea_area(products: pd.DataFrame, land, region) -> pd.DataFrame:
    """Add sea_km2 / area_km2: how much open sea of the region each product's footprint covers."""
    b = region.bbox
    bbox = (b.min_lon, b.min_lat, b.max_lon, b.max_lat)
    areas = [land.sea_km2(w, bbox) if isinstance(w, str) else (np.nan, np.nan) for w in products["footprint"]]
    out = products.copy()
    out["sea_km2"] = [round(a[0]) if a[0] == a[0] else np.nan for a in areas]
    out["area_km2"] = [round(a[1]) if a[1] == a[1] else np.nan for a in areas]
    return out


def processed_products(region: str) -> set[str]:
    man = sar_dir(region) / "manifest.jsonl"
    if not man.exists():
        return set()
    return {json.loads(x)["product"] for x in man.read_text().splitlines() if x.strip()}


def _token():
    from dotenv import load_dotenv

    from maritime.sar.sentinel1 import TokenSource
    load_dotenv(ROOT / ".env")
    user, pw = os.environ.get("CDSE_USERNAME", "").strip(), os.environ.get("CDSE_PASSWORD", "").strip()
    if not user or not pw:
        raise SystemExit("CDSE_USERNAME / CDSE_PASSWORD missing in .env (free account: dataspace.copernicus.eu)")
    return TokenSource(user, pw)


def bench(models: tuple = DEFAULT_MODELS, tiles: int = 32, batch: int = 8, log=print) -> float:
    """Detector speed on this machine (s per 512 px tile), on random input."""
    import torch

    from maritime.sar.scene_pipeline import load_models
    torch.set_num_threads(max(1, os.cpu_count() or 1))
    m = load_models([MODEL_DIR / models[0]])[0]
    x = torch.randn(batch, 2, 512, 512).to(memory_format=torch.channels_last)
    with torch.inference_mode():
        m(x)
        t = time.time()
        for _ in range(max(1, tiles // batch)):
            m(x)
    s = (time.time() - t) / (max(1, tiles // batch) * batch)
    log(f"{os.cpu_count()} CPUs, {torch.get_num_threads()} threads: {s:.3f} s per tile "
        f"(~{s * 2000 / 60:.0f} min for a 2,000-tile scene)")
    return s


def run(region, hours: float = 36, max_scenes: int | None = None, product: str | None = None,
        models: tuple = DEFAULT_MODELS, keep_zip: bool = False, force: bool = False, log=print) -> list[dict]:
    """Process every new product of the last `hours` (or one named `product`). Returns manifest rows."""
    import fcntl

    import torch

    from maritime.sar import scene_pipeline as P
    from maritime.sar.sentinel1 import download, read_grd

    # one SAR job at a time (timer and manual runs share this lock; released when the process exits)
    lock_path = DATA / "logs" / "sar.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock = open(lock_path, "w")  # noqa: SIM115 - held for the whole run
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        log("another SAR run is in progress; exiting")
        return []
    log(f"== run {pd.Timestamp.now('UTC'):%Y-%m-%d %H:%M} UTC")
    torch.set_num_threads(max(1, os.cpu_count() or 1))
    found = search_region(region, hours if not product else 24 * 30)
    if product:
        found = found[found["name"].str.startswith(product.removesuffix(".SAFE"))]
    done = set() if force else processed_products(region.name)
    todo = found[~found["name"].str.removesuffix(".SAFE").isin(done)]
    lp = land_path(region.name)
    land = None
    if lp.exists():
        from maritime.sar.scene_pipeline import Land
        land = Land.load(lp)
        todo = add_sea_area(todo, land, region)
        if not product:
            skip = todo["sea_km2"] < MIN_SEA_KM2
            log(f"skipping {int(skip.sum())} mostly-land slices (< {MIN_SEA_KM2:,} km2 of sea in the region)")
            todo = todo[~skip].sort_values("sea_km2", ascending=False)
    if max_scenes:
        todo = todo.head(max_scenes)
    log(f"{len(found)} products in the last {hours:g} h, {len(todo)} new to process")
    if todo.empty:
        return []
    paths = [MODEL_DIR / m for m in models]
    missing = [p.name for p in paths if not p.exists()]
    if missing:
        raise SystemExit(f"model weights missing in {MODEL_DIR}: {missing}")
    nets = P.load_models(paths)
    from maritime.sar.length import HEAD_FILE, load_head
    len_head = load_head(nets[0], MODEL_DIR / HEAD_FILE) if (MODEL_DIR / HEAD_FILE).exists() else None
    if len_head is None:
        log(f"note: no {HEAD_FILE}; small craft use the model-scale fallback rule")
    if land is None:
        log(f"warning: no land polygons ({lp}); run `maritime sar land` first. Continuing without a land mask.")
    token = _token()
    out_rows = []
    for p in todo.to_dict("records"):
        t0 = time.time()
        z = download(p, DATA / "raw" / "sentinel1", token, log=log)
        t1 = time.time()
        sc = read_grd(z, log=log)
        t2 = time.time()
        pad = P.AIS_WINDOW
        lat, lon = sc.geo.lat, sc.geo.lon
        ais = P.load_ais(PROCESSED / region.name / "ais", pd.Timestamp(sc.start) - pad, pd.Timestamp(sc.stop) + pad,
                         (lon.min() - 0.2, lat.min() - 0.2, lon.max() + 0.2, lat.max() + 0.2))
        cover = P.coverage_cells(PROCESSED / region.name / "ais", pd.Timestamp(sc.start))
        log(f"  AIS coverage: {len(cover)} grid cells heard >= {P.COVER_MIN_MMSI} ships in the last {P.COVER_DAYS} days")
        box = (lon.min() - 0.2, lat.min() - 0.2, lon.max() + 0.2, lat.max() + 0.2)
        reception = P.reception_cells(PROCESSED / region.name / "ais", pd.Timestamp(sc.start), box)
        low = sum(v < P.RECEPTION_MIN_RATIO for v in reception.values())
        log(f"  AIS reception: {low} of {len(reception)} judged cells hear < {P.RECEPTION_MIN_RATIO:.0%} of usual")
        det = P.process_scene(sc, nets, land, ais, cover=cover, reception=reception, log=log)
        t3 = time.time()
        day = pd.Timestamp(sc.start).date()
        out = sar_dir(region.name) / "detections" / f"date={day}" / f"{sc.name}.parquet"
        out.parent.mkdir(parents=True, exist_ok=True)
        det = det.reset_index(drop=True)
        det["chip"] = range(len(det))                     # index into the chips file of this scene
        chips = P.cut_chips(sc.image, det["row"], det["col"]) if len(det) else None
        if len(det) and len_head is not None:             # Baltic length head on the cut-outs
            from maritime.sar.length import estimate
            det["length_est_m"] = np.round(estimate(nets[0], len_head, chips), 1)
        det.to_parquet(out, index=False)
        if len(det):
            P.save_chips(sc.image, det["row"], det["col"], sar_dir(region.name) / "chips" / f"date={day}" /
                         f"{sc.name}.npz", chips=chips)
        med = scene_medians_db(sc.image)
        # VV swings with wind (calm -25, gale -8 dB); VH is steadier. Warn only on implausible levels.
        if not (-30 <= med["vv_db"] <= -5 and -35 <= med["vh_db"] <= -20):
            log(f"  WARNING: median VV {med['vv_db']} / VH {med['vh_db']} dB is implausible for sea: check calibration")
        elif med["vv_db"] > -14:
            log(f"  note: VV {med['vv_db']} dB = windy sea, brighter than most xView3 training scenes (-26..-17.5)")
        from maritime.sar.report import corners_wkt
        row = {"product": sc.name, "start": str(sc.start), "orbit": p.get("orbit_direction"), **med,
               "footprint": corners_wkt(sc.geo, *sc.image.shape[1:]),
               "platform": sc.platform, "shape": list(sc.image.shape[1:]),
               "valid_frac": round(float((sc.image[0] > 0).mean()), 3), "ais_rows": len(ais),
               "detections": len(det), "ais_matched": int((det["status"] == "ais").sum()),
               "dark": int((det["status"] == "dark").sum()), "dark_high": int((det["tier"] == "high").sum()),
               "no_coverage": int((det["status"] == "no_coverage").sum()), "cover_cells": len(cover),
               "download_s": round(t1 - t0), "read_s": round(t2 - t1), "detect_s": round(t3 - t2),
               "models": list(models), "land_mask": land is not None}
        with open(sar_dir(region.name) / "manifest.jsonl", "a") as f:
            f.write(json.dumps(row) + "\n")
        out_rows.append(row)
        log(json.dumps(row))
        if not keep_zip:
            z.unlink(missing_ok=True)
        del sc
    return out_rows


def regate(region, hours: float = 96, log=print) -> int:
    """Apply the reception check to scenes processed before it existed: dark -> no_coverage where the feed
    heard < half the usual ships around the image time. One-way and idempotent. Returns the number changed."""
    from maritime.sar import scene_pipeline as P
    ais_dir = PROCESSED / region.name / "ais"
    since = pd.Timestamp.now("UTC").tz_localize(None) - pd.Timedelta(hours=hours)
    changed = 0
    for f in sorted((sar_dir(region.name) / "detections").glob("date=*/*.parquet")):
        if f.parent.name[5:] < str(since.date()):
            continue
        det = pd.read_parquet(f)
        dark = det["status"] == "dark"
        if not dark.any():
            continue
        g = det[dark]
        t = pd.Timestamp(g["time"].iloc[0])
        t = t.tz_convert(None) if t.tzinfo else t
        box = (g["lon"].min() - 0.5, g["lat"].min() - 0.5, g["lon"].max() + 0.5, g["lat"].max() + 0.5)
        rec = P.reception_cells(ais_dir, t, box)
        low = np.array([rec.get(c, 1.0) < P.RECEPTION_MIN_RATIO for c in P.cells_of(g["lat"], g["lon"])])
        if low.any():
            det.loc[g.index[low], ["status", "tier"]] = ["no_coverage", ""]
            det.to_parquet(f, index=False)
            changed += int(low.sum())
            log(f"{f.stem}: {int(low.sum())} of {len(g)} dark -> no_coverage (low AIS reception)")
    log(f"{changed} detections changed")
    return changed


def debug_dump(region, product: str, log=print) -> Path:
    """Keep a product's metadata XMLs and a thinned DN sample (every 8th pixel, both bands) for offline
    checking of calibration and geolocation: data/raw/sentinel1/debug/<product>/ (~20 MB)."""
    import zipfile

    import rasterio

    from maritime.sar.sentinel1 import _members, download
    found = search_region(region, 24 * 30)
    found = found[found["name"].str.startswith(product.removesuffix(".SAFE"))]
    if found.empty:
        raise SystemExit(f"{product} not found in the last 30 days")
    p = found.iloc[0].to_dict()
    z = download(p, DATA / "raw" / "sentinel1", _token(), log=log)
    out = DATA / "raw" / "sentinel1" / "debug" / z.stem
    out.mkdir(parents=True, exist_ok=True)
    arrays = {}
    with zipfile.ZipFile(z) as zf:
        for n in zf.namelist():
            if n.endswith(".xml") and "/annotation/" in n or n.endswith("manifest.safe"):
                (out / n.replace("/", "__")).write_bytes(zf.read(n))
        for pol in ("vv", "vh"):
            m = _members(zf, pol)
            with rasterio.open(f"/vsizip/{z.resolve()}/{m['tiff']}") as src:
                arrays[f"dn_{pol}"] = src.read(1)[::8, ::8]
                arrays[f"tiff_{pol}"] = np.array(m["tiff"])
    np.savez_compressed(out / "dn_every8.npz", **arrays)
    (out / "product.json").write_text(json.dumps({k: (v if isinstance(v, str | int | float | type(None)) else str(v))
                                                  for k, v in p.items()}, indent=1))
    log(f"debug files in {out}; zip kept at {z}")
    return out


def chip_gallery(region, days: float = 3, out: Path | None = None, max_per_kind: int = 16, log=print) -> Path | None:
    """Contact sheet of radar cut-outs from the last `days`: dark candidates first (high tier, then possible),
    then 'no AIS coverage' ships and AIS-matched ships for comparison. VV on a fixed dB scale, so brightness is
    comparable between chips. """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from maritime.sar.scene_pipeline import STATIC_LOOKBACK_DAYS, refine_status, static_mask
    from maritime.sar.xview3 import DB_MAX, DB_MIN
    now = pd.Timestamp.now("UTC").tz_localize(None)
    since = now - pd.Timedelta(days=days)
    history = now - pd.Timedelta(days=max(days, STATIC_LOOKBACK_DAYS))
    base = sar_dir(region.name)
    frames = []
    for f in sorted((base / "detections").glob("date=*/*.parquet")):
        if f.parent.name[5:] >= str(history.date()):
            frames.append(pd.read_parquet(f).assign(_file=str(f)))
    if not frames:
        log(f"no detections in the last {days:g} days")
        return None
    alld = pd.concat(frames, ignore_index=True)
    alld = refine_status(alld, static_mask(alld))           # current dark rules, incl. the static check
    rows = []
    for fname, d in alld.groupby("_file"):
        f = Path(fname)
        chip_file = base / "chips" / f.parent.name / f"{f.stem}.npz"
        if not chip_file.exists() or f.parent.name[5:] < str(since.date()) or "chip" not in d:
            continue
        chips = np.load(chip_file)["chips"]
        for r in d.itertuples():
            if pd.Timestamp(r.time) >= since and r.status not in ("non_vessel", "outside_area"):
                kind = "dark " + r.tier if r.status == "dark" else r.status
                rows.append((kind, r, chips[int(r.chip)]))
    if not rows:
        log(f"no chips in the last {days:g} days")
        return None
    order = ["dark high", "dark possible", "static", "small", "no_coverage", "ais"]
    picked = []
    for k in order:
        sel = sorted([x for x in rows if x[0] == k], key=lambda x: -x[1].score)[:max_per_kind]
        picked += sel
    ncol = 8
    nrow = int(np.ceil(len(picked) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(ncol * 2.0, nrow * 2.3), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    to_db = lambda q: (q.astype(float) - 1) / 254 * (DB_MAX - DB_MIN) + DB_MIN   # noqa: E731
    for ax, (kind, r, ch) in zip(axes.ravel(), picked, strict=False):
        ax.imshow(to_db(ch[0]), cmap="gray", vmin=-25, vmax=5)
        name = f"{r.mmsi}" if r.status == "ais" else ""
        size = f"~{r.length_rough_m:.0f} m" if r.length_rough_m == r.length_rough_m else "small"
        ax.set_title(f"{kind}\n{r.score:.2f} | {size} {name}\n{str(r.time)[5:16]}", fontsize=7)
    plt.tight_layout()
    out = out or DATA / "sar_qc" / f"gallery_{pd.Timestamp.now():%Y%m%d_%H%M}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=90)
    plt.close(fig)
    log(f"{len(picked)} chips -> {out}")
    return out
