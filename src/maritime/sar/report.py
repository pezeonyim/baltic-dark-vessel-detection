"""Validation report for the live Baltic SAR results, using AIS as a partial answer key.

For every processed scene:
  - recall: of the AIS ships inside the image, in water where the feed hears ships, with a fresh report
    (<= 10 min from the image time), how many did the detector find (status "ais" with that MMSI)?
    Split by AIS-reported length, class A/B, moving/stationary and wind (scene VV median).
  - false-alarm ceiling: among confident vessel detections in covered water, the share left without AIS
    ("dark" after the dark rules). Each is a real dark ship or a false alarm, so this bounds the false-alarm share.
  - length check: estimated length vs AIS-reported length for matched ships (bias, and how often a ship
    >= 45 m long is estimated under the 30 m "small craft" cut-off).

    maritime sar report --hours 72        (on the server: it can fetch missing scene footprints from Copernicus)
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

LENGTH_BANDS = [0, 30, 60, 120, 1000]
LENGTH_LABELS = ["<30 m", "30-60 m", "60-120 m", ">=120 m"]
FRESH_S = 600          # an AIS position estimated from a report more than 10 min away is not used for recall


def corners_wkt(geo, H: int, W: int) -> str:
    """Scene outline as a WKT polygon from the four image corners."""
    lat, lon = geo.to_latlon([0, 0, H - 1, H - 1], [0, W - 1, W - 1, 0])
    pts = list(zip(lon, lat, strict=True)) + [(lon[0], lat[0])]
    return "POLYGON((" + ",".join(f"{x:.5f} {y:.5f}" for x, y in pts) + "))"


def load_manifest(sar_dir: Path) -> pd.DataFrame:
    man = sar_dir / "manifest.jsonl"
    if not man.exists():
        return pd.DataFrame()
    rows = {}
    for line in man.read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            rows[r["product"]] = r                   # a re-run replaces the earlier line
    return pd.DataFrame(list(rows.values()))


def fill_footprints(man: pd.DataFrame, search_fn, log=print) -> pd.DataFrame:
    """Add footprints for scenes processed before they were saved, from the Copernicus catalogue."""
    man = man.copy()
    if "footprint" not in man:
        man["footprint"] = None
    missing = man["footprint"].isna()
    if missing.any():
        try:
            found = search_fn()
            fp = dict(zip(found["name"].str.removesuffix(".SAFE"), found["footprint"], strict=True))
            man.loc[missing, "footprint"] = man.loc[missing, "product"].map(fp)
        except Exception as e:                      # no network (e.g. on the Mac): report what we can
            log(f"could not fetch footprints for {int(missing.sum())} older scenes ({type(e).__name__})")
    return man


def scene_recall(det: pd.DataFrame, ais_pos: pd.DataFrame, footprint: str, cover: set, active: set,
                 land=None) -> pd.DataFrame:
    """One row per AIS ship that should be visible in the image, with found = matched by the detector.
    ais_pos: mmsi, lat, lon, dt_s, sog, length, transceiver_class (positions at the image time)."""
    import shapely

    from maritime.sar.scene_pipeline import covered
    if ais_pos.empty or not isinstance(footprint, str):
        return pd.DataFrame()
    poly = shapely.from_wkt(footprint)
    a = ais_pos[ais_pos["dt_s"] <= FRESH_S].copy()
    if a.empty:
        return a
    inside = shapely.contains_xy(poly, a["lon"].to_numpy(), a["lat"].to_numpy())
    a = a[inside]
    if land is not None and len(a):
        a = a[~land.on_land(a["lat"], a["lon"])]
    if a.empty:
        return a
    a = a[covered(a["lat"], a["lon"], cover, active)]
    matched = set(det.loc[det["status"] == "ais", "mmsi"].astype("int64"))
    a["found"] = a["mmsi"].astype("int64").isin(matched)
    a["length_band"] = pd.cut(a["length"], LENGTH_BANDS, labels=LENGTH_LABELS, right=False)
    a["moving"] = a["sog"].fillna(0) >= 1.0
    return a


def summarise(recall: pd.DataFrame, det: pd.DataFrame, lengths: pd.DataFrame) -> dict:
    """Headline numbers. recall: rows from scene_recall (+ vv_db); det: refined detections in covered water;
    lengths: matched detections with est_length and ais_length."""
    out = {}
    if len(recall):
        r = recall
        out["ais_ships_expected"] = int(len(r))
        out["recall_all"] = round(float(r["found"].mean()), 3)
        cls_a = r["transceiver_class"] == "A"
        out["recall_class_a_moving_>=60m"] = _rate(r[cls_a & r["moving"] & (r["length"] >= 60)])
        out["recall_by_length"] = {str(k): _rate(g) for k, g in r.groupby("length_band", observed=True)}
        out["recall_by_class"] = {str(k): _rate(g) for k, g in r.groupby("transceiver_class")}
        out["recall_moving_vs_stationary"] = {"moving": _rate(r[r["moving"]]), "stationary": _rate(r[~r["moving"]])}
        wind = pd.cut(r["vv_db"], [-40, -18, -14, 0], labels=["calm (VV < -18 dB)", "moderate", "windy (VV > -14 dB)"])
        out["recall_by_wind"] = {str(k): _rate(g) for k, g in r.groupby(wind, observed=True)}
    if len(det):
        conf = det[det["status"].isin(["ais", "dark"]) & (det["length_m"] >= 30)]
        n_dark = int((conf["status"] == "dark").sum())
        out["confident_vessels_in_covered_water"] = int(len(conf))
        out["no_ais_share_ceiling"] = round(n_dark / max(len(conf), 1), 3)
        out["dark_high"] = int(((det["status"] == "dark") & (det["tier"] == "high")).sum())
    if len(lengths):
        ratio = lengths["est_length"] / lengths["ais_length"]
        big = lengths[lengths["ais_length"] >= 45]
        out["length_ratio_median"] = round(float(ratio.median()), 2)
        out["length_abs_error_median_m"] = round(float((lengths["est_length"] - lengths["ais_length"]).abs().median()), 1)
        out["ships_>=45m_estimated_<30m"] = _share(big["est_length"] < 30) if len(big) else None
        from maritime.sar.scene_pipeline import DARK_MIN_MODEL_LENGTH
        small = lengths[lengths["ais_length"] < 30]
        out["ships_>=45m_below_small_cutoff"] = _share(big["est_length"] < DARK_MIN_MODEL_LENGTH) if len(big) else None
        out["boats_<30m_above_small_cutoff"] = _share(small["est_length"] >= DARK_MIN_MODEL_LENGTH) if len(small) else None
    return out


def _rate(g: pd.DataFrame) -> dict:
    return {"n": int(len(g)), "recall": round(float(g["found"].mean()), 3) if len(g) else None}


def _share(mask: pd.Series) -> dict:
    return {"n": int(len(mask)), "share": round(float(mask.mean()), 3)}


def run_report(region, hours: float = 72, log=print) -> dict:
    """Build the report from the processed scenes of the last `hours`; writes report/<timestamp>.json + CSVs."""
    from maritime.fusion import ais_at_time, to_local_m
    from maritime.paths import PROCESSED
    from maritime.sar import run as R
    from maritime.sar import scene_pipeline as P

    base = R.sar_dir(region.name)
    man = load_manifest(base)
    if man.empty:
        log("no scenes processed yet")
        return {}
    man["start_ts"] = pd.to_datetime(man["start"])
    since = pd.Timestamp.now("UTC").tz_localize(None) - pd.Timedelta(hours=hours)
    man = man[man["start_ts"] >= since]
    man = fill_footprints(man, lambda: R.search_region(region, hours + 48), log=log)
    land = P.Land.load(R.land_path(region.name)) if R.land_path(region.name).exists() else None
    ais_dir = PROCESSED / region.name / "ais"

    # all detections of the window (+ history for the static check), current dark rules
    files = sorted((base / "detections").glob("date=*/*.parquet"))
    hist = since - pd.Timedelta(days=P.STATIC_LOOKBACK_DAYS)
    frames = [pd.read_parquet(f) for f in files if f.parent.name[5:] >= str(hist.date())]
    alld = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if alld.empty:
        log("no detections")
        return {}
    alld = P.refine_status(alld, P.static_mask(alld))

    rec_rows, det_rows, len_rows = [], [], []
    for sc in man.itertuples():
        det = alld[alld["product"] == sc.product]
        t0 = pd.Timestamp(sc.start_ts)
        t_mid = t0 + pd.Timedelta(seconds=12)
        if not isinstance(sc.footprint, str):
            continue
        import shapely
        b = shapely.from_wkt(sc.footprint).bounds
        ais = P.load_ais(ais_dir, t0 - P.AIS_WINDOW, t0 + P.AIS_WINDOW, (b[0] - .2, b[1] - .2, b[2] + .2, b[3] + .2))
        if ais.empty:
            continue
        lat0, lon0 = float(ais["lat"].mean()), float(ais["lon"].mean())
        a = ais.copy()
        a["x"], a["y"] = to_local_m(a["lat"], a["lon"], lat0, lon0)
        pos = ais_at_time(a, t_mid, max_age_s=P.AIS_WINDOW.total_seconds())
        pos["lat"] = lat0 + pos["y"] / 6_371_000 * 180 / np.pi
        pos["lon"] = lon0 + pos["x"] / (6_371_000 * np.cos(np.radians(lat0))) * 180 / np.pi
        static = ais.sort_values("ts").groupby("mmsi")[["length", "transceiver_class", "vessel_name"]].last()
        pos = pos.join(static, on="mmsi")
        cover = P.coverage_cells(ais_dir, t0)
        active = set(P.cells_of(ais["lat"], ais["lon"]))
        r = scene_recall(det, pos, sc.footprint, cover, active, land)
        if len(r):
            rec_rows.append(r.assign(product=sc.product, vv_db=getattr(sc, "vv_db", np.nan)))
        if len(det):
            cov = P.covered(det["lat"], det["lon"], cover, active)
            det_rows.append(det[cov | (det["status"] == "ais")])
            m = det[det["status"] == "ais"].join(static["length"].rename("ais_length"), on="mmsi")
            m = m[m["ais_length"].notna() & (m["ais_length"] > 0)]
            len_rows.append(pd.DataFrame({"product": sc.product, "mmsi": m["mmsi"], "est_length": m["length_m"],
                                          "ais_length": m["ais_length"]}))
    recall = pd.concat(rec_rows, ignore_index=True) if rec_rows else pd.DataFrame()
    dets = pd.concat(det_rows, ignore_index=True) if det_rows else pd.DataFrame()
    lengths = pd.concat(len_rows, ignore_index=True) if len_rows else pd.DataFrame()
    summary = {"hours": hours, "scenes": int(len(man)),
               "scenes_with_footprint": int(man["footprint"].notna().sum()), **summarise(recall, dets, lengths)}
    out = base / "report"
    out.mkdir(parents=True, exist_ok=True)
    stamp = pd.Timestamp.now("UTC").strftime("%Y%m%d_%H%M")
    (out / f"report_{stamp}.json").write_text(json.dumps(summary, indent=1, default=str))
    if len(recall):
        recall.to_csv(out / f"recall_{stamp}.csv", index=False)
    if len(lengths):
        lengths.to_csv(out / f"lengths_{stamp}.csv", index=False)
    log(json.dumps(summary, indent=1, default=str))
    log(f"written to {out}")
    return summary
