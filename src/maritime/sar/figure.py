"""Visual proof for the README: one processed Sentinel-1 scene re-downloaded and drawn with what the pipeline
concluded about every detection, plus a full-resolution zoom with each AIS ship's position at the image time.

    maritime sar figure --product S1C_IW_GRDH_1SDV_...     (on the server; writes sar/report/figures/*.png)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

STYLE = {   # status -> (label, marker, colour, size)
    "ais": ("matched to its AIS", "o", "#3ec6f0", 70),
    "ais_gap": ("AIS briefly missed (explained)", "o", "#5fe0a0", 70),
    "dark": ("dark candidate", "D", "#ff8c1e", 60),
    "no_coverage": ("ship, no AIS coverage", "s", "#bdbdbd", 30),
    "small": ("small craft", ".", "#9a9a9a", 30),
    "static": ("fixed structure", "x", "#ffffff", 30),
}


def _db(u8):
    return np.where(u8 > 0, (u8.astype(np.float32) - 1) / 254 * 60.0 - 50.0, np.nan)


def window_for(rows, cols, weights, size: int, H: int, W: int) -> tuple[int, int]:
    """Top-left (row, col) of the size x size window, centred on one of the points, with the largest weight sum."""
    rows, cols, weights = np.asarray(rows, float), np.asarray(cols, float), np.asarray(weights, float)
    best, score = (0, 0), -1.0
    for r, c in zip(rows, cols, strict=True):
        r0, c0 = int(np.clip(r - size / 2, 0, H - size)), int(np.clip(c - size / 2, 0, W - size))
        inside = (rows >= r0) & (rows < r0 + size) & (cols >= c0) & (cols < c0 + size)
        if weights[inside].sum() > score:
            best, score = (r0, c0), float(weights[inside].sum())
    return best


def best_window(det: pd.DataFrame, size: int, H: int, W: int) -> tuple[int, int]:
    """Top-left (row, col) of the size x size window with the most ship detections, preferring mixed statuses."""
    pts = det[det["status"].isin(["ais", "ais_gap", "dark", "no_coverage"])]
    best, score = (0, 0), -1
    for r0, c0 in zip(pts["row"] - size / 2, pts["col"] - size / 2, strict=True):
        r0, c0 = int(np.clip(r0, 0, H - size)), int(np.clip(c0, 0, W - size))
        inside = pts[(pts["row"].between(r0, r0 + size)) & (pts["col"].between(c0, c0 + size))]
        s = len(inside) + 3 * inside["status"].nunique() + 4 * inside["status"].isin(["dark", "ais_gap"]).sum()
        if s > score:
            best, score = (r0, c0), s
    return best


def make(region, product: str, zoom_px: int = 700, log=print) -> list:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from maritime.fusion import ais_at_time, to_local_m
    from maritime.paths import DATA, PROCESSED
    from maritime.sar import run as R
    from maritime.sar import scene_pipeline as P
    from maritime.sar.sentinel1 import download, read_grd
    from maritime.web.app import apply_evidence

    files = list((R.sar_dir(region.name) / "detections").glob(f"date=*/{product}*.parquet"))
    if not files:
        raise SystemExit(f"no detections for {product}")
    det = pd.read_parquet(files[0])
    found = R.search_region(region, 24 * 30)
    found = found[found["name"].str.startswith(product)]
    if found.empty:
        raise SystemExit(f"{product} not in the catalogue for the last 30 days")
    from maritime.sar.sentinel1 import Geo, GRDScene
    cache = DATA / "raw" / "sentinel1" / f"figure_{product[:40]}.npz"
    if cache.exists():
        c = np.load(cache)
        geo = Geo(c["g_line"], c["g_pixel"], c["g_lat"], c["g_lon"], c["g_time"])
        sc = GRDScene(product, c["image"], geo, c["start"][()], c["stop"][()], str(c["platform"]), "")
        log(f"using cached scene {cache}")
    else:
        z = download(found.iloc[0].to_dict(), DATA / "raw" / "sentinel1", R._token(), log=log)
        sc = read_grd(z, log=log)
        z.unlink(missing_ok=True)
        g = sc.geo
        np.savez(cache, image=sc.image, g_line=g.line, g_pixel=g.pixel, g_lat=g.lat, g_lon=g.lon, g_time=g.time,
                 start=np.datetime64(sc.start), stop=np.datetime64(sc.stop), platform=np.array(sc.platform))
    # current rules + evidence verdicts, as on the map (static check needs the other scenes)
    allf = sorted((R.sar_dir(region.name) / "detections").glob("date=*/*.parquet"))
    alld = pd.concat([pd.read_parquet(f) for f in allf], ignore_index=True)
    alld = P.refine_status(alld, P.static_mask(alld))
    det = alld[alld["product"] == det["product"].iloc[0]].copy()
    det = apply_evidence(det, R.sar_dir(region.name) / "evidence" / "latest.parquet")
    land = P.Land.load(R.land_path(region.name)) if R.land_path(region.name).exists() else None
    H, W = sc.image.shape[1:]
    out_dir = R.sar_dir(region.name) / "report" / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    t_img = pd.Timestamp(sc.start) + (pd.Timestamp(sc.stop) - pd.Timestamp(sc.start)) / 2
    when = f"{t_img:%Y-%m-%d %H:%M} UTC"
    written = []

    # 1) whole scene, 1/8 resolution
    step = 8
    fig, ax = plt.subplots(figsize=(10, 10 * H / W + 0.6), facecolor="#111111")
    ax.imshow(_db(sc.image[0, ::step, ::step]), cmap="gray", vmin=-26, vmax=-4, extent=(0, W, H, 0))
    for st, (lab, m, c, s) in STYLE.items():
        d = det[det["status"] == st]
        if len(d):
            ax.scatter(d["col"], d["row"], marker=m, s=s * 0.6, facecolors="none" if m in "os" else c,
                       edgecolors=c, linewidths=1.2, label=f"{lab} ({len(d)})")
    ax.set_axis_off()
    ax.legend(loc="lower left", fontsize=8, facecolor="#222222", edgecolor="#444444", labelcolor="#eeeeee")
    ax.set_title(f"{product[:3]} {when}: {len(det)} detections at sea", color="#eeeeee", fontsize=11, loc="left")
    fig.tight_layout()
    p1 = out_dir / f"{product[:40]}_scene.png"
    fig.savefig(p1, dpi=130, facecolor="#111111")
    plt.close(fig)
    written.append(p1)

    # AIS positions at the image time over the whole scene ("ground truth" for ships that broadcast)
    lat_g, lon_g = sc.geo.lat, sc.geo.lon
    lat0, lon0 = float(np.mean(lat_g)), float(np.mean(lon_g))
    ais = P.load_ais(PROCESSED / region.name / "ais", t_img - P.AIS_WINDOW, t_img + P.AIS_WINDOW,
                     (lon_g.min() - 0.1, lat_g.min() - 0.1, lon_g.max() + 0.1, lat_g.max() + 0.1))
    gt = pd.DataFrame(columns=["mmsi", "row", "col", "name", "length"])
    if len(ais):
        a = ais.copy()
        a["x"], a["y"] = to_local_m(a["lat"], a["lon"], lat0, lon0)
        pos = ais_at_time(a, t_img)
        if len(pos):
            plat = lat0 + pos["y"].to_numpy() / 6_371_000 * 180 / np.pi
            plon = lon0 + pos["x"].to_numpy() / (6_371_000 * np.cos(np.radians(lat0))) * 180 / np.pi
            pr, pc = sc.geo.to_rowcol(plat, plon)
            last = ais.sort_values("ts").groupby("mmsi")[["vessel_name", "length"]].last()
            gt = pd.DataFrame({"mmsi": pos["mmsi"].to_numpy(), "row": pr, "col": pc})
            gt["name"] = gt["mmsi"].map(last["vessel_name"])
            gt["length"] = gt["mmsi"].map(last["length"])
            gt = gt[(gt["row"] >= 0) & (gt["row"] < H) & (gt["col"] >= 0) & (gt["col"] < W)]
            if land is not None and len(gt):
                glat, glon = sc.geo.to_latlon(gt["row"].to_numpy(), gt["col"].to_numpy())
                gt = gt[~land.on_land(glat, glon)].copy()
                glat, glon = sc.geo.to_latlon(gt["row"].to_numpy(), gt["col"].to_numpy())
                from maritime.risk import _land_km
                gt["land_km"] = _land_km(land, glat, glon)
    matched = set(det.loc[det["status"] == "ais", "mmsi"].astype("int64"))
    gt["found"] = gt["mmsi"].astype("int64").isin(matched)
    names = dict(zip(gt["mmsi"].astype("int64"), gt["name"], strict=True))

    def zoom(r0, c0, title, fname, label_misses=False):
        win = _db(sc.image[0, r0:r0 + zoom_px, c0:c0 + zoom_px])
        fig, ax = plt.subplots(figsize=(9, 9), facecolor="#111111")
        ax.imshow(win, cmap="gray", vmin=-26, vmax=-2, extent=(c0, c0 + zoom_px, r0 + zoom_px, r0))
        g = gt[gt["row"].between(r0, r0 + zoom_px) & gt["col"].between(c0, c0 + zoom_px)]
        for found, lab, col in ((True, "AIS position, ship detected", "#3ec6f0"),
                                (False, "AIS position, ship missed", "#ff5a5a")):
            gg = g[g["found"] == found]
            if len(gg):
                ax.scatter(gg["col"], gg["row"], marker="+", s=110, color=col, linewidths=1.6, label=lab)
            if not found and label_misses:
                for m in gg.itertuples():
                    t = "  ".join(x for x in ((m.name or "").strip() if isinstance(m.name, str) else "",
                                              f"{int(m.length)} m (AIS)" if m.length == m.length and m.length else "")
                                  if x)
                    if t:
                        ax.annotate(t, (m.col, m.row), xytext=(8, 6), textcoords="offset points", color="#ffb0b0",
                                    fontsize=7.5)
        d = det[det["row"].between(r0, r0 + zoom_px) & det["col"].between(c0, c0 + zoom_px)]
        for st, (lab, m, c, sz) in STYLE.items():
            dd = d[d["status"] == st]
            if len(dd):
                ax.scatter(dd["col"], dd["row"], marker=m, s=sz * 4, facecolors="none", edgecolors=c,
                           linewidths=1.6, label=f"detection: {lab}")
        for r in d.itertuples():
            if r.status == "ais":
                name = names.get(int(r.mmsi))
            elif r.status == "ais_gap":
                name = getattr(r, "ev_name", None)
            else:
                name = None              # weak links etc. are not identifications
            name = name if isinstance(name, str) and name.strip() else ""
            length = f"{int(r.length_rough_m)} m" if r.length_rough_m == r.length_rough_m and r.length_rough_m else ""
            txt = "  ".join(t for t in (name.strip(), length) if t)
            if txt:
                ax.annotate(txt, (r.col, r.row), xytext=(10, -12), textcoords="offset points", color="#eeeeee",
                            fontsize=8)
        ax.plot([c0 + 30, c0 + 130], [r0 + zoom_px - 30] * 2, color="#eeeeee", linewidth=2)
        ax.text(c0 + 30, r0 + zoom_px - 42, "1 km", color="#eeeeee", fontsize=9)
        ax.set_axis_off()
        ax.legend(loc="upper right", fontsize=8, facecolor="#222222", edgecolor="#444444", labelcolor="#eeeeee")
        ax.set_title(title, color="#eeeeee", fontsize=11, loc="left")
        fig.tight_layout()
        p = out_dir / fname
        fig.savefig(p, dpi=130, facecolor="#111111")
        plt.close(fig)
        written.append(p)

    km = zoom_px * 10 / 1000
    # (a) matching: AIS-matched ships with their AIS crosses
    m = det[det["status"].isin(["ais", "ais_gap"])]
    if len(m):
        w = np.where(m["status"] == "ais_gap", 1.5, 1.0)          # many matched ships; explained gaps a bonus
        r0, c0 = window_for(m["row"], m["col"], w, zoom_px, H, W)
        zoom(r0, c0, f"Radar detections matched to AIS, {km:.0f} x {km:.0f} km, {when}", f"{product[:40]}_zoom.png")
    # (b) dark candidates
    d = det[det["status"] == "dark"]
    if len(d):
        r0, c0 = window_for(d["row"], d["col"], np.ones(len(d)), zoom_px, H, W)
        zoom(r0, c0, f"Dark candidates, {km:.0f} x {km:.0f} km, {when}", f"{product[:40]}_dark.png")
    # (c) failure cases at sea (>= 1 km from land): AIS ships the detector missed
    missed = gt[~gt["found"]]
    if "land_km" in missed:
        missed = missed[missed["land_km"] >= 1.0]
    if len(missed):
        r0, c0 = window_for(missed["row"], missed["col"], np.ones(len(missed)), zoom_px, H, W)
        zoom(r0, c0, f"Missed: AIS ships at sea the detector did not find, {km:.0f} x {km:.0f} km, {when}",
             f"{product[:40]}_misses.png", label_misses=True)
    log(f"AIS ships in the scene at sea: {len(gt)}, detected {int(gt['found'].sum())}")
    log(f"status counts: {det['status'].value_counts().to_dict()}; zoom window row {r0}, col {c0}")
    log("written: " + ", ".join(str(p) for p in written))
    return written
