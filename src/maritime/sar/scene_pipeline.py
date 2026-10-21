"""One Sentinel-1 scene -> vessel detections matched against recorded AIS.

process_scene(scene, models, land, ais) does:
  1. skip tiles that are no-data or (almost) all land, run the detector (detector.predict_scene);
  2. keep detections >= SCORE_MIN, convert pixel -> lat/lon and per-line acquisition time;
  3. drop detections on land (OSM land polygons);
  4. estimate every AIS transmitter's position at the image time and match (fusion.dark_candidates);
  5. return one row per detection: status ais / dark / non_vessel, matched mmsi, tier for dark ones.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

SCORE_MIN = 0.45          # detection threshold calibrated on separate xView3 scenes
HIGH_TIER = 0.60          # dark candidates >= 0.6 had precision ~0.86 on xView3
GATE_M = 300.0            # 200 m xView3 match distance + ~100 m for the ~25 s scene duration
AIS_WINDOW = pd.Timedelta("30min")
COVER_CELL = (0.1, 0.2)   # coverage grid, degrees (lat, lon): ~11 x 11 km in the Baltic
COVER_DAYS = 7
COVER_MIN_MMSI = 3        # a cell counts as covered if the recorder heard >= 3 different ships there in 7 days
RECEPTION_DAYS = 3        # reception check: ships heard around a cell +-30 min of the image ...
RECEPTION_MIN_RATIO = 0.5 # ... must be >= half the mean of the same window on the previous 3 days ...
RECEPTION_MIN_USUAL = 3.0 # ... judged only where >= 3 ships are usually heard (fewer: too noisy, not judged)


# ---------------------------------------------------------------- land

def prepare_land(src: Path, bbox: tuple, out: Path) -> int:
    """Clip OSM land polygons (land-polygons-split-4326 shapefile or zip) to bbox (min_lon, min_lat, max_lon,
    max_lat) and save as WKB in a parquet file. Returns the number of polygons."""
    import pyogrio
    import shapely

    src = Path(src)
    path = str(src)
    if src.suffix == ".zip":
        import zipfile
        with zipfile.ZipFile(src) as z:
            shp = next(n for n in z.namelist() if n.endswith(".shp"))
        path = f"/vsizip/{src.resolve()}/{shp}"
    _, _, geom, _ = pyogrio.raw.read(path, bbox=bbox, read_geometry=True, columns=[])
    g = shapely.clip_by_rect(shapely.from_wkb(geom), *bbox)
    g = g[~shapely.is_empty(g)]
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"wkb": shapely.to_wkb(g)}).to_parquet(out, index=False)
    return len(g)


class Land:
    """Land polygons with a spatial index: point-on-land tests and coarse masks in a scene's geometry."""

    def __init__(self, geoms):
        import shapely
        self.geoms = np.asarray(geoms)
        self.tree = shapely.STRtree(self.geoms)

    @classmethod
    def load(cls, path: Path) -> Land:
        import shapely
        return cls(shapely.from_wkb(pd.read_parquet(path)["wkb"].to_numpy()))

    def on_land(self, lat, lon) -> np.ndarray:
        import shapely
        pts = shapely.points(np.asarray(lon, float), np.asarray(lat, float))
        hit = np.zeros(len(pts), bool)
        if len(pts):
            idx = self.tree.query(pts, predicate="within")
            hit[np.unique(idx[0])] = True
        return hit

    def sea_km2(self, wkt: str, bbox: tuple) -> tuple[float, float]:
        """(sea area, total area) in km2 of a footprint polygon inside bbox (degrees scaled by cos(latitude):
        within ~1% over one slice, plenty for choosing scenes)."""
        import shapely
        g = shapely.clip_by_rect(shapely.from_wkt(wkt), *bbox)
        if g.is_empty:
            return 0.0, 0.0
        hits = self.geoms[self.tree.query(g, predicate="intersects")]
        # OSM land-polygons-split pieces do not overlap, so clipped areas can simply be summed
        land_area = float(shapely.area(shapely.intersection(hits, g)).sum()) if len(hits) else 0.0
        k = 111.32 ** 2 * np.cos(np.radians(g.centroid.y))
        return max(g.area - land_area, 0.0) * k, g.area * k

    def mask(self, geo, H: int, W: int, step: int = 16) -> np.ndarray:
        """Boolean land mask on a grid of step x step scene pixels (True = land)."""
        import shapely
        from rasterio.features import rasterize

        lat_g, lon_g = geo.lat, geo.lon
        box = shapely.box(lon_g.min() - 0.1, lat_g.min() - 0.1, lon_g.max() + 0.1, lat_g.max() + 0.1)
        sel = self.geoms[self.tree.query(box, predicate="intersects")]
        shape = (H // step + 1, W // step + 1)
        if len(sel) == 0:
            return np.zeros(shape, bool)
        sel = shapely.clip_by_rect(sel, *box.bounds)

        def to_px(xy):                     # (lon, lat) -> (col, row) on the coarse grid
            r, c = geo.to_rowcol(xy[:, 1], xy[:, 0])
            return np.column_stack([c / step, r / step])
        px = shapely.transform(sel, to_px)
        return rasterize(((g, 1) for g in px if not g.is_empty), out_shape=shape, fill=0, dtype="uint8").astype(bool)


def tile_filter(land_mask: np.ndarray, size: int = 512, step: int = 16, max_land: float = 0.99):
    """keep_tile(origin) for detector.predict_scene: drop tiles that are > max_land land."""
    n = size // step

    def keep(o):
        r, c = o[0] // step, o[1] // step
        return land_mask[r:r + n, c:c + n].mean() <= max_land if land_mask[r:r + n, c:c + n].size else True
    return keep


# ---------------------------------------------------------------- AIS

def load_ais(ais_dir: Path, t0, t1, bounds) -> pd.DataFrame:
    """AIS position reports between t0 and t1 inside bounds (min_lon, min_lat, max_lon, max_lat), from the
    recorder's hive-partitioned parquet (date=YYYY-MM-DD/*.parquet)."""
    import duckdb

    t0, t1 = pd.Timestamp(t0), pd.Timestamp(t1)
    days = pd.date_range(t0.normalize(), t1.normalize(), freq="D")
    files = [str(p) for d in days for p in sorted((Path(ais_dir) / f"date={d.date()}").glob("*.parquet"))]
    cols = ["mmsi", "ts", "lat", "lon", "sog", "cog", "vessel_name", "length", "transceiver_class"]
    if not files:
        return pd.DataFrame(columns=cols)
    q = (f"SELECT {', '.join(cols)} FROM read_parquet(?, union_by_name=true) "
         "WHERE ts BETWEEN ? AND ? AND lon BETWEEN ? AND ? AND lat BETWEEN ? AND ?")
    con = duckdb.connect()
    try:
        return con.execute(q, [files, t0.to_pydatetime(), t1.to_pydatetime(), *bounds[0::2], *bounds[1::2]]).df()
    finally:
        con.close()


def coverage_cells(ais_dir: Path, end, days: int = COVER_DAYS, min_mmsi: int = COVER_MIN_MMSI) -> set:
    """Grid cells (i, j) where the free AIS feed demonstrably hears ships: >= min_mmsi distinct MMSIs with a
    position report in the last `days` days. "No AIS" means nothing outside these cells."""
    import duckdb

    end = pd.Timestamp(end)
    days_list = pd.date_range((end - pd.Timedelta(days=days)).normalize(), end.normalize(), freq="D")
    files = [str(p) for d in days_list for p in sorted((Path(ais_dir) / f"date={d.date()}").glob("*.parquet"))]
    if not files:
        return set()
    a, b = COVER_CELL
    q = (f"SELECT CAST(floor(lat / {a}) AS INT) i, CAST(floor(lon / {b}) AS INT) j, count(DISTINCT mmsi) n "
         "FROM read_parquet(?, union_by_name=true) WHERE ts BETWEEN ? AND ? GROUP BY 1, 2")
    con = duckdb.connect()
    try:
        df = con.execute(q, [files, (end - pd.Timedelta(days=days)).to_pydatetime(), end.to_pydatetime()]).df()
    finally:
        con.close()
    return set(map(tuple, df.loc[df["n"] >= min_mmsi, ["i", "j"]].to_numpy().tolist()))


def cells_of(lat, lon) -> list:
    a, b = COVER_CELL
    return list(zip(np.floor(np.asarray(lat) / a).astype(int).tolist(),
                    np.floor(np.asarray(lon) / b).astype(int).tolist(), strict=True))


def _heard_by_cell(ais: pd.DataFrame) -> pd.Series:
    """Distinct MMSIs heard in each cell's 3 x 3 neighbourhood (index (i, j))."""
    if ais.empty:
        return pd.Series(dtype=float)
    c = np.array(cells_of(ais["lat"], ais["lon"]))
    parts = [pd.DataFrame({"i": c[:, 0] + di, "j": c[:, 1] + dj, "mmsi": ais["mmsi"].to_numpy()})
             for di in (-1, 0, 1) for dj in (-1, 0, 1)]
    return pd.concat(parts).groupby(["i", "j"])["mmsi"].nunique().astype(float)


def reception_cells(ais_dir: Path, t, bounds, days: int = RECEPTION_DAYS) -> dict:
    """{cell: ratio} of ships heard around the cell within +-AIS_WINDOW of t vs the mean of the same window on the
    previous `days` days, for cells where >= RECEPTION_MIN_USUAL ships are usually heard. A partial receiver outage
    keeps a cell "live" (someone is still heard) while most ships go unheard; this catches it."""
    t = pd.Timestamp(t)
    now = _heard_by_cell(load_ais(ais_dir, t - AIS_WINDOW, t + AIS_WINDOW, bounds))
    past = [_heard_by_cell(load_ais(ais_dir, t - pd.Timedelta(days=k) - AIS_WINDOW,
                                    t - pd.Timedelta(days=k) + AIS_WINDOW, bounds)) for k in range(1, days + 1)]
    past = [p for p in past if len(p)]
    if not past:
        return {}
    usual = pd.concat(past, axis=1).fillna(0).mean(axis=1)
    usual = usual[usual >= RECEPTION_MIN_USUAL]
    ratio = now.reindex(usual.index).fillna(0) / usual
    return {k: float(v) for k, v in ratio.items()}


def covered(lat, lon, cover: set, active: set, reception: dict | None = None) -> np.ndarray:
    """True where the cell is covered over the last week AND the receivers were live around the image time
    (some report within the AIS window in this cell or its 8 neighbours) AND, where `reception` is given and
    judged, the feed heard at least RECEPTION_MIN_RATIO of the usual number of ships there."""
    out = []
    for i, j in cells_of(lat, lon):
        live = any((i + di, j + dj) in active for di in (-1, 0, 1) for dj in (-1, 0, 1))
        ok = reception is None or reception.get((i, j), 1.0) >= RECEPTION_MIN_RATIO
        out.append((i, j) in cover and live and ok)
    return np.array(out, bool)


# ---------------------------------------------------------------- the scene

def load_models(paths, device="cpu"):
    import torch

    from maritime.sar import detector as D
    models = []
    for p in paths:
        m = D.build_model("resnet18", pretrained=False, in_ch=2)
        m.load_state_dict(torch.load(p, map_location=device, weights_only=True))
        m = m.to(device).eval()
        if str(device) == "cpu":
            m = m.to(memory_format=torch.channels_last)
        models.append(m)
    return models


def process_scene(scene, models, land: Land | None, ais: pd.DataFrame, batch: int = 8,
                  cover: set | None = None, reception: dict | None = None, log=print) -> pd.DataFrame:
    """Detections of one GRDScene with AIS status. `ais`: recorder rows around the image time (load_ais);
    `cover`: coverage_cells() of the week before; without it every unmatched vessel counts as dark."""
    import time

    from maritime.fusion import ais_at_time, dark_candidates, doppler_shift_m, to_local_m
    from maritime.sar import detector as D
    img, geo = scene.image, scene.geo
    H, W = img.shape[1:]
    t0 = time.time()
    keep = tile_filter(land.mask(geo, H, W)) if land is not None else None
    t1 = time.time()
    dets, origins = D.predict_scene(models, img, batch=batch, keep_tile=keep, device="cpu")
    t2 = time.time()
    d = dets[-1]
    log(f"{scene.name}: {len(origins)} tiles, {int((d[:, 2] >= SCORE_MIN).sum()) if len(d) else 0} detections "
        f"(land mask {t1 - t0:.0f} s, detector {t2 - t1:.0f} s = {(t2 - t1) / max(len(origins), 1):.2f} s/tile)")
    cols = ["row", "col", "score", "length_m", "p_vessel", "p_fishing"]
    out = pd.DataFrame(d, columns=cols)
    out = out[out["score"] >= SCORE_MIN].reset_index(drop=True)
    if out.empty:
        return _finish(out, scene)
    out["lat"], out["lon"] = geo.to_latlon(out["row"].to_numpy(), out["col"].to_numpy())
    out["time"] = geo.times(out["row"].to_numpy())
    if land is not None:
        on = land.on_land(out["lat"], out["lon"])
        out = out[~on].reset_index(drop=True)
        log(f"  {int(on.sum())} on land dropped")

    lat0, lon0 = float(np.mean(geo.lat)), float(np.mean(geo.lon))
    out["x"], out["y"] = to_local_m(out["lat"], out["lon"], lat0, lon0)
    t_mid = scene.start + (scene.stop - scene.start) / 2
    pos = pd.DataFrame(columns=["mmsi", "x", "y", "dt_s", "unc_m", "sog", "cog"])
    along = geo.along_track(lat0, lon0)
    if len(ais):
        a = ais.copy()
        a["x"], a["y"] = to_local_m(a["lat"], a["lon"], lat0, lon0)
        pos = ais_at_time(a, pd.Timestamp(t_mid), max_age_s=AIS_WINDOW.total_seconds())
    if len(pos):
        # Doppler: moving ships appear shifted along the track by (R / V) x radial speed
        plat = lat0 + pos["y"].to_numpy() / 6_371_000 * 180 / np.pi
        plon = lon0 + pos["x"].to_numpy() / (6_371_000 * np.cos(np.radians(lat0))) * 180 / np.pi
        pr, pc = geo.to_rowcol(plat, plon)
        r_over_v = (float(np.nanmean(geo.slant_range)) if geo.slant_range is not None else 850e3) / scene.speed_ms
        pos["along_extra_m"] = np.minimum(
            doppler_shift_m(pos["sog"], pos["cog"], along, geo.incidence_at(pr, pc), r_over_v) + 100.0, 2000.0)
    res = dark_candidates(out, pos, score_min=SCORE_MIN, base_gate_m=GATE_M, along=along)
    res["mmsi"] = np.where(res["ais_idx"] >= 0, pos["mmsi"].to_numpy()[np.maximum(res["ais_idx"], 0)]
                           if len(pos) else -1, -1)
    res["ais_dt_s"] = np.where(res["ais_idx"] >= 0, pos["dt_s"].to_numpy()[np.maximum(res["ais_idx"], 0)]
                               if len(pos) else np.nan, np.nan)
    if cover is not None:
        active = set(cells_of(ais["lat"], ais["lon"])) if len(ais) else set()
        unc = (res["status"] == "dark") & ~covered(res["lat"], res["lon"], cover, active, reception)
        res.loc[unc, "status"] = "no_coverage"
    res = refine_status(res)
    log(f"  AIS transmitters at image time: {len(pos)}; matched {int((res.status == 'ais').sum())}, "
        f"dark {int((res.status == 'dark').sum())} (high {int((res.tier == 'high').sum())}), "
        f"no AIS coverage {int((res.status == 'no_coverage').sum())}, "
        f"non-vessel {int((res.status == 'non_vessel').sum())}")
    return _finish(res, scene)


# ---------------------------------------------------------------- what counts as "dark"
# AIS is mandatory for SOLAS ships (>= 300 GT, roughly >= 35-45 m); smaller craft need not carry it, so "no AIS"
# means nothing for them. On Baltic images the model's length output is compressed and ~4x too short
# (a 183 m ship reads ~48), so the cut-off is set on the model's own scale, calibrated with AIS-reported lengths:
# 16 keeps 93% of AIS ships >= 45 m and removes ~80% of AIS boats < 30 m.
DARK_MIN_MODEL_LENGTH = 16.0   # fallback when a detection has no length_est_m (no length head / no cut-out)
DARK_MIN_LENGTH_M = 50.0       # with the Baltic length head, small craft = estimated < 50 m
# rough metres from the model's length (power law fitted on 1,029 AIS-matched Baltic ships); only meant
# for display of ship-sized targets: it overestimates small boats
LEN_CAL_A, LEN_CAL_B = 8.54, 0.753


def length_rough_m(model_length):
    return LEN_CAL_A * np.power(np.maximum(np.asarray(model_length, float), 0.0), LEN_CAL_B)
# Inside the region's bbox but not the target sea: Norway's coast/fjords and the North Sea strip west of Jutland.
EXCLUDE_BOXES = [(8.0, 59.5, 12.5, 66.0), (8.0, 53.0, 8.7, 57.2)]   # (min_lon, min_lat, max_lon, max_lat)
STATIC_RADIUS_M = 100.0    # same spot on two passes ...
STATIC_MIN_HOURS = 6.0     # ... at least this far apart in time = a fixed structure (turbine, pier, rock)
STATIC_LOOKBACK_DAYS = 30


def static_mask(det: pd.DataFrame, radius_m: float = STATIC_RADIUS_M, min_hours: float = STATIC_MIN_HOURS) -> np.ndarray:
    """True for detections with another detection from a different image, >= min_hours apart, within radius_m.
    Ships move between passes; turbines, bridge piers, platforms and rocks do not. Uses all statuses (a
    structure can be AIS-less or carry an AIS aid-to-navigation)."""
    from scipy.spatial import cKDTree
    out = np.zeros(len(det), bool)
    if len(det) < 2:
        return out
    lat0 = float(det["lat"].mean())
    xy = np.column_stack([det["lon"].to_numpy() * 111_320 * np.cos(np.radians(lat0)), det["lat"].to_numpy() * 110_540])
    t = pd.to_datetime(det["time"]).to_numpy()
    prod = det["product"].to_numpy()
    for i, j in cKDTree(xy).query_pairs(radius_m, output_type="ndarray"):
        if prod[i] != prod[j] and abs((t[i] - t[j]) / np.timedelta64(1, "h")) >= min_hours:
            out[i] = out[j] = True
    return out


def refine_status(det: pd.DataFrame, static: np.ndarray | None = None) -> pd.DataFrame:
    """Apply the "what counts as dark" rules on top of the scene statuses (ais / dark / no_coverage / non_vessel):
    static spot -> "static"; outside the target sea -> "outside_area" (for unmatched ones); unmatched vessel
    below DARK_MIN_MODEL_LENGTH (model scale) -> "small"; tier only for what stays dark; length_rough_m for display. Applied when results are read, so
    old results follow the current rules."""
    d = det.copy()
    if d.empty:
        return d
    unmatched = d["status"].isin(["dark", "no_coverage"])
    out_area = np.zeros(len(d), bool)
    for x0, y0, x1, y1 in EXCLUDE_BOXES:
        out_area |= d["lon"].between(x0, x1).to_numpy() & d["lat"].between(y0, y1).to_numpy()
    d.loc[unmatched & out_area, "status"] = "outside_area"
    est = d["length_est_m"] if "length_est_m" in d else pd.Series(np.nan, index=d.index)
    small = np.where(est.notna(), est < DARK_MIN_LENGTH_M, d["length_m"] < DARK_MIN_MODEL_LENGTH)
    d.loc[(d["status"] == "dark") & small, "status"] = "small"
    if static is not None:
        d.loc[static & (d["status"] != "ais"), "status"] = "static"
    d["tier"] = np.where(d["status"] != "dark", "", np.where(d["score"] >= HIGH_TIER, "high", "possible"))
    rough = np.where(d["length_m"] >= DARK_MIN_MODEL_LENGTH, length_rough_m(d["length_m"]).round(), np.nan)
    d["length_rough_m"] = np.where(est.notna(), np.where(~small, est.round(), np.nan), rough)
    return d


CHIP_HALF = 48            # 96 x 96 px cut-outs (~1 km at 10 m pixels) around each detection


def cut_chips(img: np.ndarray, rows, cols, half: int = CHIP_HALF) -> np.ndarray:
    """(n, 2, 2*half, 2*half) uint8 VV/VH cut-outs around detections (zero-padded at the scene edge)."""
    rows = np.asarray(rows, float).round().astype(int)
    cols = np.asarray(cols, float).round().astype(int)
    _, H, W = img.shape
    s = 2 * half
    chips = np.zeros((len(rows), 2, s, s), np.uint8)
    for i, (r, c) in enumerate(zip(rows, cols, strict=True)):
        r0, c0 = r - half, c - half
        rr0, cc0 = max(r0, 0), max(c0, 0)
        rr1, cc1 = min(r0 + s, H), min(c0 + s, W)
        if rr1 > rr0 and cc1 > cc0:
            chips[i, :, rr0 - r0:rr1 - r0, cc0 - c0:cc1 - c0] = img[:, rr0:rr1, cc0:cc1]
    return chips


def save_chips(img: np.ndarray, rows, cols, out: Path, half: int = CHIP_HALF, chips=None) -> int:
    """Save a (n, 2, 2*half, 2*half) uint8 VV/VH cut-out around each detection (radar geometry, zero-padded at
    the scene edge) to `out` (.npz, key "chips"). Row i of the detections file = chip i. The scene itself is
    deleted after processing, so these are what is left for visual review and later validation."""
    rows = np.asarray(rows, float).round().astype(int)
    cols = np.asarray(cols, float).round().astype(int)
    chips = cut_chips(img, rows, cols, half) if chips is None else chips
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, chips=chips, rows=rows, cols=cols)
    return len(chips)


def _finish(df: pd.DataFrame, scene) -> pd.DataFrame:
    keep = ["time", "lat", "lon", "score", "p_vessel", "p_fishing", "length_m", "status", "tier", "mmsi",
            "ais_dt_s", "row", "col"]
    df = df.reindex(columns=keep)
    df.insert(0, "product", scene.name)
    df.insert(1, "platform", scene.platform)
    return df
