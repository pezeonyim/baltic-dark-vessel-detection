"""AIS evidence for dark candidates: which AIS ship could this be?

A dark candidate is a radar-detected ship with no AIS at the image time, in water where the feed hears ships.
If a ship-sized AIS vessel has a reporting gap that brackets the image time AND its track through the gap passes
the candidate (straight line from its last to its next report, or dead reckoning from the last report if it has
not reappeared), it is a plausible identity: the ship that went silent. Mere reachability is not enough: with gaps
of hours almost any ship could have been anywhere (tested 2026-10-14: 34-411 "reachable" ships per candidate). Linking an anonymous detection to an MMSI / name is what
lets it be checked against the sanctions watchlist.

Each candidate gets a verdict from its best link (each AIS ship is given to at most one candidate per image):
  explained   track passes within EXPLAINED_KM and the gap is < LONG_GAP_H: a ship whose reports simply were not
              received around the image time (ordinary reception gap). Not dark; it also confirms the detection.
  went_silent track passes within EXPLAINED_KM but the gap is >= LONG_GAP_H: a known ship placed at the spot while
              silent for hours. The interesting case; check the watchlist column.
  low_reception like went_silent, but the feed heard < half the usual number of ships around the spot at that
              hour (seen 2026-10-14 15:56 Gulf of Finland: 12 ships vs 31 the day before).
  usual_gap   like went_silent, but a silence that long is not unusual where it happened: gap surprise < SURPRISE_MIN
              from the learned coverage model (features/coverage_map.py) fitted on the last 7 days of both
              feeds. Catches chronic reception holes the reception ratio cannot see (always poor, so "normal").
  crowded     like went_silent, but 3+ ships fit within EXPLAINED_KM (an anchorage): which ship it is is a guess.
  feed_outage like went_silent, but >= OUTAGE_MIN_SHIPS other ships nearby went quiet within OUTAGE_WINDOW_MIN of it:
              the shore receiver stopped, not the ship (seen 2026-10-14 in the Gulf of Finland: ~15 ships last
              heard 22:04-22:28 UTC, image 6 h later).
  weak_link   only looser links (track miss up to TRACK_TOL). Barely above chance (null test below): not evidence.
  unexplained no AIS ship fits. Stays dark: a ship that never transmits, or a false alarm.

Null test (2026-10-14, 66 candidates, 10-11..10-13): the same candidates moved 15 km in a random direction get
2 tight links (<= 2 km) per run against 15 at the real positions, and 14-17 weak links against 21. So tight links
are real (~7x chance; expect ~2 of them to be coincidence), weak links are mostly coincidence.

    maritime sar evidence --hours 72      (needs the watchlist: `maritime sanctions update`)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

MAX_SPEED_KN = 16.0          # few laden tankers/bulkers exceed this
SLACK_M = 2_000.0            # position + timing slack
MIN_GAP_MIN = 45.0           # a reporting gap shorter than this is normal AIS jitter, not "silent"
MAX_GAP_H = 18.0             # longer gaps: the track through the gap is too uncertain to place the ship
TRACK_TOL_M = 5_000.0        # predicted position within 5 km ...
TRACK_TOL_FRAC = 0.10        # ... or 10% of the distance travelled through the gap, whichever is larger
LENGTH_RATIO = (0.5, 2.0)    # radar length / AIS length must lie in here (keeps 95% of true matches)
MIN_SHIP_LENGTH_M = 45.0     # class B or AIS length < 45 m cannot be a ship-sized dark candidate
EXPLAINED_KM = 2.0           # track passes within 2 km of the detection: it is that ship
LONG_GAP_H = 6.0             # silent this long around the image: more than a reception hiccup
OUTAGE_RADIUS_KM = 50.0      # ships within this distance of the candidate ...
OUTAGE_WINDOW_MIN = 30.0     # ... that went quiet within 30 min of the linked ship ...
SURPRISE_MIN = 4.0           # went_silent only if a silence this long happens < 1 in 10,000 times there
COVER_FIT_DAYS = 7
RECEPTION_RADIUS_KM = 30.0   # reception check: distinct ships heard within 30 km, +-30 min of the image ...
RECEPTION_MIN_RATIO = 0.5    # ... below half of the same hour on the previous 3 days: reception was degraded
OUTAGE_MIN_SHIPS = 4         # ... 4 or more of them: the receiver went down, not the ship's transponder
BEFORE_H, AFTER_H = 12.0, 24.0
KNOT_MS = 0.514444


def _dist_m(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * 6_371_000 * np.arcsin(np.sqrt(a))


def silent_links(cand: pd.Series, ais: pd.DataFrame, max_speed_kn: float = MAX_SPEED_KN,
                 min_gap_min: float = MIN_GAP_MIN) -> pd.DataFrame:
    """AIS ships whose reporting gap brackets the candidate's time and that could have been at its position.
    cand: time, lat, lon. ais: mmsi, ts, lat, lon (+ vessel_name, length, sog), around the candidate.
    Returns one row per plausible MMSI with gap timing, distances and a slack ratio (smaller = tighter fit)."""
    t = pd.Timestamp(cand["time"])
    est = cand.get("length_est_m", np.nan)
    v = max_speed_kn * KNOT_MS
    out = []
    for mmsi, g in ais.sort_values("ts").groupby("mmsi"):
        last_len = g["length"].dropna().iloc[-1] if "length" in g and g["length"].notna().any() else np.nan
        cls = g["transceiver_class"].dropna().iloc[-1] if "transceiver_class" in g and g["transceiver_class"].notna().any() else None
        if cls == "B" or (last_len == last_len and last_len < MIN_SHIP_LENGTH_M):
            continue
        if est == est and est and last_len == last_len and last_len > 0 and \
                not LENGTH_RATIO[0] <= est / last_len <= LENGTH_RATIO[1]:
            continue                               # a 110 m radar target is not a 366 m ship
        ts = g["ts"].to_numpy()
        tt = np.datetime64(t.to_datetime64())
        i = np.searchsorted(ts, tt)                # reports before: [:i], after: [i:]
        if i == 0:
            continue                               # never heard before the image: no "went silent"
        prev = g.iloc[i - 1]
        nxt = g.iloc[i] if i < len(g) else None
        gap_start = pd.Timestamp(prev["ts"])
        gap_end = pd.Timestamp(nxt["ts"]) if nxt is not None else None
        before_s = (t - gap_start).total_seconds()
        after_s = (gap_end - t).total_seconds() if gap_end is not None else None
        gap_min = ((gap_end - gap_start).total_seconds() / 60) if gap_end is not None else before_s / 60
        if gap_min < min_gap_min or before_s < 10 * 60 or gap_min > MAX_GAP_H * 60:
            continue                               # still reporting around the image time, or gap too long
        d_prev = float(_dist_m(prev["lat"], prev["lon"], cand["lat"], cand["lon"]))
        reach_prev = v * before_s + SLACK_M
        if d_prev > reach_prev:
            continue
        d_next, reach_next = np.nan, np.nan
        if nxt is not None:
            d_next = float(_dist_m(cand["lat"], cand["lon"], nxt["lat"], nxt["lon"]))
            reach_next = v * after_s + SLACK_M
            if d_next > reach_next:
                continue
        # where the track says the ship was at the image time
        if nxt is not None:
            w = before_s / (before_s + after_s)
            plat = prev["lat"] + w * (nxt["lat"] - prev["lat"])
            plon = prev["lon"] + w * (nxt["lon"] - prev["lon"])
            travelled = float(_dist_m(prev["lat"], prev["lon"], nxt["lat"], nxt["lon"]))
        else:
            sog = float(prev.get("sog")) if pd.notna(prev.get("sog")) else 0.0
            cog = np.radians(float(prev.get("cog"))) if pd.notna(prev.get("cog")) else 0.0
            dist = sog * KNOT_MS * before_s
            plat = prev["lat"] + dist * np.cos(cog) / 111_000
            plon = prev["lon"] + dist * np.sin(cog) / (111_000 * np.cos(np.radians(prev["lat"])))
            travelled = dist
        d_pred = float(_dist_m(plat, plon, cand["lat"], cand["lon"]))
        tol = max(TRACK_TOL_M, TRACK_TOL_FRAC * travelled)
        if d_pred > tol:
            continue
        slack = d_pred / tol
        out.append({"mmsi": int(mmsi), "vessel_name": prev.get("vessel_name"), "length": prev.get("length"),
                    "last_lat": float(prev["lat"]), "last_lon": float(prev["lon"]),
                    "silent_from": gap_start, "silent_until": gap_end, "gap_h": round(gap_min / 60, 1),
                    "dist_last_km": round(d_prev / 1000, 1),
                    "dist_next_km": round(d_next / 1000, 1) if nxt is not None else None,
                    "reappeared": nxt is not None, "track_miss_km": round(d_pred / 1000, 1), "fit": round(slack, 2)})
    cols = ["mmsi", "vessel_name", "length", "last_lat", "last_lon", "silent_from", "silent_until", "gap_h", "dist_last_km",
            "dist_next_km", "reappeared", "track_miss_km", "fit"]
    return pd.DataFrame(out, columns=cols).sort_values("fit") if out else pd.DataFrame(columns=cols)


def _ts(t) -> pd.Timestamp:
    t = pd.Timestamp(t)
    return t.tz_convert(None) if t.tzinfo else t


def assign_links(links: pd.DataFrame) -> pd.DataFrame:
    """One AIS ship can be at most one detection of an image: greedy by track miss (links: cand, product, ...)."""
    if links.empty:
        return links
    links = links.sort_values(["track_miss_km", "fit"])
    used_ship, used_cand, keep = set(), set(), []
    for i, r in links.iterrows():
        ship, cand = (r["product"], r["mmsi"]), r["cand"]
        if ship in used_ship or cand in used_cand:
            continue
        used_ship.add(ship)
        used_cand.add(cand)
        keep.append(i)
    return links.loc[keep]


def quiet_together(ais: pd.DataFrame, t: pd.Timestamp, lat: float, lon: float, since: pd.Timestamp,
                   mmsi: int) -> int:
    """How many other ships within OUTAGE_RADIUS_KM were silent at t and last heard within OUTAGE_WINDOW_MIN of
    `since` (the linked ship's last report). Many at once means the receiver, not the ships, went quiet."""
    if ais.empty:
        return 0
    before = ais[ais["ts"] < t]
    last = before.sort_values("ts").groupby("mmsi").tail(1)
    last = last[last["mmsi"] != mmsi]
    near = _dist_m(last["lat"].to_numpy(), last["lon"].to_numpy(), lat, lon) <= OUTAGE_RADIUS_KM * 1000
    quiet = (t - last["ts"]).dt.total_seconds() >= MIN_GAP_MIN * 60
    close = (last["ts"] - since).abs().dt.total_seconds() <= OUTAGE_WINDOW_MIN * 60
    return int((near & quiet.to_numpy() & close.to_numpy()).sum())


def coverage_model(ais_dir, end, days: int = COVER_FIT_DAYS) -> dict | None:
    """Fit the gap-surprise coverage model on the last `days` of recorded AIS (all feeds)."""
    from pathlib import Path

    from maritime.features import coverage_map as C
    from maritime.features.coverage import outages, vessel_volume
    end = pd.Timestamp(end)
    files = [str(f) for d in pd.date_range((end - pd.Timedelta(days=days)).normalize(), end.normalize())
             for f in sorted((Path(ais_dir) / f"date={d.date()}").glob("*.parquet"))]
    if not files:
        return None
    from maritime.risk import duck
    con = duck()                                                 # memory-capped, no progress bar
    try:
        con.execute("CREATE TABLE cov_ais AS SELECT DISTINCT mmsi, ts, lat, lon, sog, nav_status, transceiver_class "
                    "FROM read_parquet(?, union_by_name=true)", [files])
        out = outages(vessel_volume(con, "cov_ais"))
        stats, excess = C.interval_stats(con, "cov_ais", "aisstream", out)
    finally:
        con.close()
    return C.fit(stats, excess) if len(excess) > 100 else None


def gap_surprise(model: dict, points, gap_h: float, expected_s: float = 60.0) -> float:
    """-log10 P(a silence this long) at the worse-covered of `points` [(lat, lon), ...].
    expected_s: the report interval the ratio is measured against (60 s floor for class A under way)."""
    from maritime.features import coverage_map as C
    p = max(C.cell_p(model, la, lo)[0] for la, lo in points)
    prob = C.tail_probability(model, p, gap_h * 3600 / expected_s)
    return float(-np.log10(max(prob, 1e-300)))


def reception_ratio(ais_dir, t: pd.Timestamp, lat: float, lon: float, days: int = 3) -> float:
    """Distinct ships heard within RECEPTION_RADIUS_KM, +-30 min of t, relative to the mean of the same window on
    the previous `days` days (nan if there is no history)."""
    from maritime.sar.scene_pipeline import load_ais
    dl = RECEPTION_RADIUS_KM / 111.0
    box = (lon - dl / np.cos(np.radians(lat)), lat - dl, lon + dl / np.cos(np.radians(lat)), lat + dl)

    def heard(tc):
        a = load_ais(ais_dir, tc - pd.Timedelta("30min"), tc + pd.Timedelta("30min"), box)
        if a.empty:
            return 0
        a = a[_dist_m(a["lat"].to_numpy(), a["lon"].to_numpy(), lat, lon) <= RECEPTION_RADIUS_KM * 1000]
        return a["mmsi"].nunique()
    past = [heard(t - pd.Timedelta(days=k)) for k in range(1, days + 1)]
    past = [n for n in past if n > 0]
    return heard(t) / float(np.mean(past)) if past else float("nan")


def verdict(best: pd.Series | None) -> str:
    if best is None:
        return "unexplained"
    if best["track_miss_km"] <= EXPLAINED_KM:
        if best["gap_h"] < LONG_GAP_H:
            return "explained"
        if best.get("quiet_together", 0) >= OUTAGE_MIN_SHIPS:
            return "feed_outage"
        if best.get("reception", 1.0) < RECEPTION_MIN_RATIO:
            return "low_reception"
        if best.get("surprise", np.inf) < SURPRISE_MIN:
            return "usual_gap"
        return "crowded" if best.get("n_tight", 1) >= 3 else "went_silent"
    return "weak_link"


def evidence_for(dark: pd.DataFrame, ais_dir, watchlist=None, cover=None, log=print) -> pd.DataFrame:
    """One row per dark candidate: verdict, the assigned AIS ship (if any), how many ships fitted, watchlist flag."""
    from maritime.sar.scene_pipeline import load_ais
    dark = dark.reset_index(drop=True)
    all_links = []
    r_deg = (MAX_SPEED_KN * KNOT_MS * AFTER_H * 3600 + SLACK_M) / 111_000
    for _, grp in dark.groupby("product"):           # one AIS query per image, covering all its candidates
        t = _ts(grp["time"].iloc[0])
        lat_pad = r_deg
        lon_pad = r_deg / np.cos(np.radians(float(grp["lat"].max())))
        box = (grp["lon"].min() - lon_pad, grp["lat"].min() - lat_pad, grp["lon"].max() + lon_pad,
               grp["lat"].max() + lat_pad)
        ais = load_ais(ais_dir, t - pd.Timedelta(hours=BEFORE_H), t + pd.Timedelta(hours=AFTER_H), box)
        for k, c in grp.iterrows():
            near = ais[(ais["lat"] - c.lat).abs() < lat_pad]
            links = silent_links(pd.Series({"time": _ts(c.time), "lat": c.lat, "lon": c.lon,
                                            "length_est_m": c.get("length_est_m", np.nan)}), near)
            if len(links):
                links["quiet_together"] = [quiet_together(near, _ts(c.time), c.lat, c.lon, pd.Timestamp(f), m)
                                           for f, m in zip(links["silent_from"], links["mmsi"], strict=True)]
                all_links.append(links.assign(cand=k, product=c["product"], n_fits=len(links),
                                              n_tight=int((links["track_miss_km"] <= EXPLAINED_KM).sum())))
    all_links = [x for x in all_links if len(x)]
    links = pd.concat(all_links, ignore_index=True) if all_links else pd.DataFrame()
    best = assign_links(links).set_index("cand") if len(links) else pd.DataFrame()
    rows = []
    for k, c in enumerate(dark.itertuples()):
        b = best.loc[k] if k in best.index else None
        if b is not None and b["track_miss_km"] <= EXPLAINED_KM and b["gap_h"] >= LONG_GAP_H:
            b = b.copy()
            b["reception"] = round(reception_ratio(ais_dir, _ts(c.time), c.lat, c.lon), 2)
            if cover is not None:
                b["surprise"] = round(gap_surprise(cover, [(b["last_lat"], b["last_lon"]), (c.lat, c.lon)],
                                                   float(b["gap_h"])), 2)
        row = {"product": c.product, "chip": getattr(c, "chip", np.nan), "time": _ts(c.time),
               "lat": c.lat, "lon": c.lon, "score": c.score,
               "tier": c.tier, "length_rough_m": getattr(c, "length_rough_m", np.nan), "verdict": verdict(b),
               "n_fits": int(b["n_fits"]) if b is not None else 0,
               "n_tight": int(b["n_tight"]) if b is not None else 0}
        if b is not None:
            row.update({f: b[f] for f in ("mmsi", "vessel_name", "length", "silent_from", "silent_until", "gap_h",
                                          "reappeared", "track_miss_km", "quiet_together")})
            row["reception"] = b.get("reception", np.nan)
            row["surprise"] = b.get("surprise", np.nan)
            if watchlist is not None and len(watchlist):
                m = watchlist.match(None, int(b["mmsi"]))
                row["watchlist"] = m.get("listed_category_label") if m else None
        rows.append(row)
    out = pd.DataFrame(rows)
    if len(out):
        log(f"{len(out)} dark candidates: " + ", ".join(f"{k} {v}" for k, v in out["verdict"].value_counts().items()))
    return out


def run_evidence(region, hours: float = 72, log=print) -> pd.DataFrame:
    """Evidence for the dark candidates of the last `hours` (current dark rules); writes report/evidence_*.csv."""
    from datetime import datetime, timezone

    from maritime.paths import PROCESSED
    from maritime.sar import run as R
    from maritime.web.app import load_sar
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    det, _ = load_sar(region.name, hours / 24, now=now, with_evidence=False)
    dark = det[det["status"] == "dark"] if len(det) else det
    if dark.empty:
        log("no dark candidates in the window")
        return pd.DataFrame()
    wl = None
    try:
        from maritime import sanctions
        wl = sanctions.load()
    except Exception as e:                           # no watchlist (e.g. on the Mac): identities without the flag
        log(f"watchlist not available ({type(e).__name__}); skipping the watchlist check")
    ais_dir = PROCESSED / region.name / "ais"
    cover = coverage_model(ais_dir, now)
    if cover is None:
        log("not enough AIS to fit the coverage model; went_silent is not checked for usual gaps")
    out = evidence_for(dark, ais_dir, watchlist=wl, cover=cover, log=log)
    d = R.sar_dir(region.name) / "report"
    d.mkdir(parents=True, exist_ok=True)
    f = d / f"evidence_{pd.Timestamp.now('UTC').strftime('%Y%m%d_%H%M')}.csv"
    out.to_csv(f, index=False)
    latest = R.sar_dir(region.name) / "evidence" / "latest.parquet"      # read by the map (web/app.py load_sar)
    latest.parent.mkdir(parents=True, exist_ok=True)
    out.assign(evaluated=pd.Timestamp.now("UTC").tz_localize(None)).to_parquet(latest, index=False)
    show = out[out["verdict"].isin(["went_silent", "usual_gap", "low_reception", "crowded", "feed_outage"])]
    cols = [c for c in ["time", "lat", "lon", "tier", "verdict", "vessel_name", "length", "gap_h", "track_miss_km",
                        "quiet_together", "reception", "surprise", "n_tight", "n_fits", "watchlist"] if c in show]
    if len(show):
        log(show.sort_values("verdict", ascending=False)[cols].to_string(index=False))
    if "watchlist" in out and out["watchlist"].notna().any():
        log("WATCHLIST HITS:\n" + out[out["watchlist"].notna()][cols].to_string(index=False))
    log(f"written to {f}")
    return out
