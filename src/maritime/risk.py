"""Vessel risk scoring v1: behaviour features per vessel over a time window, scored by a logistic
regression trained on vessels already listed as sanctioned or shadow fleet.

The score says "behaves like listed vessels", with reasons. It is not an accusation: unlisted vessels are
"unlabelled", not "innocent", and a high score is a lead to look at.

    maritime risk --days 7          (writes data/processed/<region>/risk/latest.parquet + a CV report)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

# Russian Baltic oil / bulk export ports (approximate terminal positions)
RUSSIAN_PORTS = {"Primorsk": (60.35, 28.65), "Ust-Luga": (59.68, 28.40), "Vysotsk": (60.62, 28.57),
                 "St Petersburg": (59.90, 30.21), "Kaliningrad": (54.70, 20.42), "Baltiysk": (54.65, 19.90)}
PORT_RADIUS_KM = 15.0
GAP_MIN_H = 0.5
MOVED_KM = 5.0                 # a silence counts as "moved while silent" if the ship was >= 5 km away afterwards
SURPRISE_MIN = 4.0             # same bar as the evidence check
JUMP_KN, JUMP_KM = 50.0, 5.0   # consecutive reports implying > 50 kn over > 5 km: position jump / spoofing
JUMP_AREA_SHIPS = 5            # ... counted only where fewer than 5 ships jumped in the window: where many do, it is
                               # area-wide GNSS interference (eastern Gulf of Finland, 2026-10), not the ship
LOITER_KN, LOITER_LAND_KM = 1.5, 10.0
MEET_M, MEET_KN, MEET_SLOTS = 200.0, 2.0, 6    # alongside (< 200 m), both slow, for 6 x 10 min = 1 h
MEET_LAND_KM = 3.0             # not inside a harbour
DWELL_CAP_H = 1.0              # time credited to one report (stationary hours), capped
LISTED_POSITIVE = ("sanctioned", "shadow_fleet")

FEATURES = ["silent_moved_n", "silent_moved_max_surprise", "russian_port_h", "loiter_open_sea_h", "meetings",
            "names_n", "imos_n", "no_imo_large", "jumps_n", "radar_links", "tanker", "length_m"]
# Scored inputs. russian_port_h is computed and shown as context but not scored: with it, Russian domestic coastal
# tankers crowd the top of the list; without it, average precision is higher on all 10 repeated CV splits
# (0.294 vs 0.263, top-50 precision 0.46 vs 0.41) at a slightly lower ROC-AUC (0.882 vs 0.903).
MODEL_FEATURES = [c for c in FEATURES if c != "russian_port_h"]
REASONS = {
    "silent_moved_n": "AIS went silent unusually long for the area while the ship moved",
    "silent_moved_max_surprise": "long AIS silences while moving",
    "russian_port_h": "time at Russian Baltic oil/export ports",
    "loiter_open_sea_h": "loitering in open sea",
    "meetings": "stopped alongside another ship away from port (possible ship-to-ship transfer)",
    "names_n": "several names on one transponder",
    "imos_n": "several IMO numbers on one transponder",
    "no_imo_large": "large ship without an IMO number",
    "jumps_n": "impossible position jumps",
    "radar_links": "linked to radar dark-ship candidates",
    "tanker": "tanker",
    "length_m": "ship size",
}

_HAV = ("2 * 6371 * asin(sqrt(pow(sin(radians({la2} - {la1}) / 2), 2) + cos(radians({la1})) * cos(radians({la2})) "
        "* pow(sin(radians({lo2} - {lo1}) / 2), 2)))")


def duck():
    """DuckDB connection capped at a third of RAM: its default (80%) leaves no room for Python on small machines."""
    import os

    import duckdb
    con = duckdb.connect()
    try:
        ram_mb = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2**20
        con.execute(f"SET memory_limit = '{max(512, int(ram_mb / 3))}MB'")
    except (ValueError, OSError):
        pass
    con.execute("SET preserve_insertion_order = false")
    con.execute("SET enable_progress_bar = false")
    return con


def _hav(la1, lo1, la2, lo2) -> str:
    return _HAV.format(la1=la1, lo1=lo1, la2=la2, lo2=lo2)


def load_window(con, ais_dir: Path, start, end) -> None:
    """Table `p`: distinct class A reports in [start, end) from all feeds."""
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    files = [str(f) for d in pd.date_range(start.normalize(), end.normalize())
             for f in sorted((Path(ais_dir) / f"date={d.date()}").glob("*.parquet"))]
    if not files:
        raise SystemExit(f"no AIS in {ais_dir} for {start}..{end}")
    con.execute("SET enable_progress_bar = false")
    con.execute("""CREATE OR REPLACE TABLE p AS
        SELECT DISTINCT mmsi, ts, lat, lon, sog, nav_status, vessel_type, length, upper(trim(vessel_name)) AS name,
               nullif(trim(imo), '') AS imo
        FROM read_parquet(?, union_by_name = true)
        WHERE transceiver_class = 'A' AND ts >= ? AND ts < ? AND mmsi BETWEEN 200000000 AND 799999999""",
                [files, start.to_pydatetime(), end.to_pydatetime()])
    con.execute(f"""CREATE OR REPLACE TABLE pp AS
        SELECT *, epoch(ts - p_ts) AS dt_s, {_hav('p_lat', 'p_lon', 'lat', 'lon')} AS dist_km FROM (
          SELECT mmsi, ts, lat, lon, sog, nav_status,
                 lag(ts) OVER w AS p_ts, lag(lat) OVER w AS p_lat, lag(lon) OVER w AS p_lon,
                 lag(sog) OVER w AS p_sog, lag(nav_status) OVER w AS p_nav,
                 lead(ts) OVER w AS n_ts
          FROM p WINDOW w AS (PARTITION BY mmsi ORDER BY ts))""")


def static_features(con) -> pd.DataFrame:
    return con.sql("""SELECT mmsi, count(*) AS reports,
            arg_max(name, ts) FILTER (WHERE name IS NOT NULL AND name <> '') AS vessel_name,
            arg_max(imo, ts) FILTER (WHERE imo IS NOT NULL) AS imo,
            arg_max(vessel_type, ts) FILTER (WHERE vessel_type IS NOT NULL) AS vessel_type,
            arg_max(length, ts) FILTER (WHERE length > 0) AS length_m,
            count(DISTINCT name) FILTER (WHERE name IS NOT NULL AND name <> '') AS names_n,
            count(DISTINCT imo) AS imos_n
        FROM p GROUP BY mmsi""").df()


def jump_features(con) -> pd.DataFrame:
    """Position jumps the ship itself causes: jumps in 0.5 x 1 degree cells where < JUMP_AREA_SHIPS ships jumped."""
    return con.sql(f"""WITH j AS (
            SELECT mmsi, floor(lat / 0.5) AS ci, floor(lon / 1.0) AS cj FROM pp
            WHERE dt_s > 0 AND dist_km > {JUMP_KM} AND dist_km / (dt_s / 3600) / 1.852 > {JUMP_KN}),
        area AS (SELECT ci, cj FROM j GROUP BY 1, 2 HAVING count(DISTINCT mmsi) < {JUMP_AREA_SHIPS})
        SELECT mmsi, count(*) AS jumps_n FROM j JOIN area USING (ci, cj) GROUP BY mmsi""").df()


def port_features(con) -> pd.DataFrame:
    near = " OR ".join(f"{_hav('lat', 'lon', la, lo)} < {PORT_RADIUS_KM}" for la, lo in RUSSIAN_PORTS.values())
    return con.sql(f"""SELECT mmsi, sum(least(coalesce(epoch(n_ts - ts), 0) / 3600, {DWELL_CAP_H})) AS russian_port_h
        FROM pp WHERE coalesce(sog, 0) < 1 AND ({near}) GROUP BY mmsi""").df()


def silence_features(con, cover) -> pd.DataFrame:
    """Silences >= GAP_MIN_H that are unusual for the area (gap surprise >= SURPRISE_MIN) and with >= MOVED_KM
    displacement."""
    from maritime.features import coverage_map as C
    g = con.sql(f"""SELECT mmsi, p_lat, p_lon, lat, lon, dt_s, dist_km, p_sog, p_nav FROM pp
        WHERE dt_s >= {GAP_MIN_H * 3600} AND dist_km >= {MOVED_KM}""").df()
    if g.empty or cover is None:
        return pd.DataFrame(columns=["mmsi", "silent_moved_n", "silent_moved_max_surprise"])
    expected = np.where((g["p_sog"].fillna(0) < 3) & g["p_nav"].isin([1, 5]), 180.0, 60.0)
    s = []
    for r, e in zip(g.itertuples(), expected, strict=True):
        p = max(C.cell_p(cover, r.p_lat, r.p_lon)[0], C.cell_p(cover, r.lat, r.lon)[0])
        s.append(-np.log10(max(C.tail_probability(cover, p, r.dt_s / e), 1e-300)))
    g["surprise"] = s
    hit = g[g["surprise"] >= SURPRISE_MIN]
    out = g.groupby("mmsi")["surprise"].max().rename("silent_moved_max_surprise").to_frame()
    out["silent_moved_n"] = hit.groupby("mmsi").size()
    return out.fillna({"silent_moved_n": 0}).reset_index()


def load_land_coarse(path: Path, tol_deg: float = 0.005):
    """Land polygons simplified to ~500 m (enough for the 3 / 10 km distance tests), built in chunks so the full
    182 MB OSM extract never has to be expanded at once; cached next to the source."""
    import pyarrow.parquet as pq
    import shapely

    from maritime.sar.scene_pipeline import Land
    cache = path.with_name(path.stem + f"_coarse{int(tol_deg * 1000)}.parquet")
    if not cache.exists():
        parts = []
        for batch in pq.ParquetFile(path).iter_batches(batch_size=20_000, columns=["wkb"]):
            g = shapely.simplify(shapely.from_wkb(batch.column(0).to_numpy(zero_copy_only=False)), tol_deg)
            g = g[~shapely.is_empty(g)]                                # keep small islands (archipelagos)
            parts.append(shapely.to_wkb(g))
        pd.DataFrame({"wkb": np.concatenate(parts)}).to_parquet(cache, index=False)
    return Land.load(cache)


def _land_km(land, lat, lon) -> np.ndarray:
    """Distance to the nearest land polygon, km (local degree scaling)."""
    import shapely
    pts = shapely.points(np.asarray(lon, float), np.asarray(lat, float))
    if not len(pts):
        return np.zeros(0)
    idx, dist = land.tree.query_nearest(pts, return_distance=True, all_matches=False)
    out = np.full(len(pts), np.inf)
    out[idx[0]] = dist
    return out * 111.0 * np.sqrt(np.cos(np.radians(np.asarray(lat, float))))   # mix of lat and lon degrees


def loiter_meet_features(con, land) -> pd.DataFrame:
    """Open-sea loitering hours and ship-to-ship meetings, on 10-minute slots of slow reports."""
    near = " OR ".join(f"{_hav('lat', 'lon', la, lo)} < {PORT_RADIUS_KM}" for la, lo in RUSSIAN_PORTS.values())
    slow = con.sql(f"""SELECT mmsi, time_bucket(INTERVAL 10 MINUTE, ts) AS slot, avg(lat) AS lat, avg(lon) AS lon,
                              max(sog) AS sog
        FROM p WHERE coalesce(sog, 0) < {max(LOITER_KN, MEET_KN)} AND NOT ({near})
        GROUP BY 1, 2""").df()
    cols = ["mmsi", "loiter_open_sea_h", "meetings"]
    if slow.empty or land is None:
        return pd.DataFrame(columns=cols)
    slow["land_km"] = _land_km(land, slow["lat"], slow["lon"])
    loiter = slow[(slow["sog"] < LOITER_KN) & (slow["land_km"] > LOITER_LAND_KM)]
    loiter_h = (loiter.groupby("mmsi").size() / 6.0).rename("loiter_open_sea_h")
    s = slow[(slow["sog"] < MEET_KN) & (slow["land_km"] > MEET_LAND_KM)]
    # neighbours only: grid cells of ~330 x 330 m (0.003 deg lat, 0.006 deg lon at ~60 N), equality join on the
    # cell and its 8 neighbours (a plain join on the 10-min slot alone compares every pair of anchored ships)
    s = s.assign(ci=np.floor(s["lat"] / 0.003).astype("int64"), cj=np.floor(s["lon"] / 0.006).astype("int64"))
    con.register("slow_df", s)
    pairs = con.sql(f"""WITH b9 AS (SELECT b.*, b.ci + di AS nci, b.cj + dj AS ncj FROM slow_df b,
                               (SELECT unnest([-1, 0, 1]) AS di), (SELECT unnest([-1, 0, 1]) AS dj))
        SELECT a.mmsi, b9.mmsi AS other, count(*) AS slots FROM slow_df a JOIN b9
        ON a.slot = b9.slot AND a.ci = b9.nci AND a.cj = b9.ncj AND a.mmsi <> b9.mmsi
        WHERE {_hav('a.lat', 'a.lon', 'b9.lat', 'b9.lon')} * 1000 < {MEET_M}
        GROUP BY 1, 2 HAVING count(*) >= {MEET_SLOTS}""").df()
    meetings = pairs.groupby("mmsi")["other"].nunique().rename("meetings")
    return pd.concat([loiter_h, meetings], axis=1).fillna(0).rename_axis("mmsi").reset_index()


def radar_features(evidence_path: Path) -> pd.DataFrame:
    if not evidence_path.exists():
        return pd.DataFrame(columns=["mmsi", "radar_links"])
    ev = pd.read_parquet(evidence_path)
    if "mmsi" not in ev:
        return pd.DataFrame(columns=["mmsi", "radar_links"])
    ev = ev[ev["mmsi"].notna() & ~ev["verdict"].isin(["explained", "unexplained"])]
    return ev.groupby(ev["mmsi"].astype("int64")).size().rename("radar_links").rename_axis("mmsi").reset_index()


def build_features(ais_dir: Path, start, end, cover=None, land=None, evidence_path: Path | None = None,
                   log=print) -> pd.DataFrame:
    con = duck()
    try:
        load_window(con, ais_dir, start, end)
        f = static_features(con)
        for part in (jump_features(con), port_features(con), silence_features(con, cover),
                     loiter_meet_features(con, land)):
            f = f.merge(part, on="mmsi", how="left")
    finally:
        con.close()
    if evidence_path is not None:
        f = f.merge(radar_features(evidence_path), on="mmsi", how="left")
    for c in FEATURES:
        if c not in f:
            f[c] = np.nan
    f["tanker"] = f["vessel_type"].between(80, 89).astype(float)
    f["no_imo_large"] = (f["imo"].isna() & (f["length_m"] >= 100)).astype(float)
    fill0 = [c for c in FEATURES if c not in ("length_m", "silent_moved_max_surprise")]
    f[fill0] = f[fill0].fillna(0)
    f["silent_moved_max_surprise"] = f["silent_moved_max_surprise"].fillna(0)
    log(f"{len(f):,} class A vessels; {int((f['russian_port_h'] > 0).sum())} at Russian ports, "
        f"{int((f['silent_moved_n'] > 0).sum())} moved while unusually silent, {int((f['meetings'] > 0).sum())} with "
        f"meetings, {int((f['jumps_n'] > 0).sum())} with position jumps")
    return f


# ---------------------------------------------------------------- model

LOG_COLS = ["silent_moved_n", "russian_port_h", "loiter_open_sea_h", "meetings", "jumps_n", "radar_links"]


def design(f: pd.DataFrame) -> pd.DataFrame:
    """Model inputs: log-scaled counts/hours (a ship with 3,000 position jumps is not 3,000x worse), capped values."""
    X = pd.DataFrame(index=f.index)
    for c in LOG_COLS:
        X[c] = np.log1p(f[c].clip(lower=0))
    X["silent_moved_max_surprise"] = f["silent_moved_max_surprise"].clip(0, 10)
    X["names_n"] = (f["names_n"] - 1).clip(0, 3)
    X["imos_n"] = (f["imos_n"] - 1).clip(0, 2)
    X["no_imo_large"] = f["no_imo_large"]
    X["tanker"] = f["tanker"]
    X["length_m"] = f["length_m"].clip(0, 400).fillna(f["length_m"].clip(0, 400).median()) / 100
    return X[MODEL_FEATURES]


def labels(f: pd.DataFrame, watchlist) -> pd.DataFrame:
    """listed_category per vessel (IMO first, then MMSI); y = 1 for sanctioned / shadow fleet."""
    cats, labs = [], []
    for r in f.itertuples():
        m = watchlist.match(r.imo if isinstance(r.imo, str) else None, int(r.mmsi)) if watchlist is not None else None
        cats.append(m["listed_category"] if m else None)
        labs.append(m.get("listed_category_label") if m else None)
    out = f.assign(listed_category=cats, listed_label=labs)
    out["y"] = out["listed_category"].isin(LISTED_POSITIVE).astype(int)
    return out


def _model(seed=0):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    return make_pipeline(StandardScaler(), LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000,
                                                              random_state=seed))


def _prec_at(y, s, k):
    k = min(k, len(y))
    return float(np.asarray(y)[np.argsort(-np.asarray(s))[:k]].mean()) if k else float("nan")


def evaluate(f: pd.DataFrame, folds: int = 5) -> dict:
    """Out-of-fold scores (stratified CV) vs the 'tanker at a Russian port' rule and an Isolation Forest."""
    from sklearn.ensemble import IsolationForest
    from sklearn.metrics import average_precision_score, roc_auc_score
    from sklearn.model_selection import StratifiedKFold
    X, y = design(f), f["y"].to_numpy()
    oof = np.zeros(len(y))
    for tr, te in StratifiedKFold(folds, shuffle=True, random_state=0).split(X, y):
        oof[te] = _model().fit(X.iloc[tr], y[tr]).predict_proba(X.iloc[te])[:, 1]
    rule = f["tanker"].to_numpy() * 2 + (f["russian_port_h"] > 0).to_numpy() * 2 + np.log1p(f["russian_port_h"])
    iso = -IsolationForest(random_state=0).fit(X).score_samples(X)
    res = {"vessels": int(len(y)), "positives": int(y.sum())}
    for name, s in (("model_cv", oof), ("rule_tanker_russian_port", rule), ("isolation_forest", iso)):
        res[name] = {"roc_auc": round(float(roc_auc_score(y, s)), 3),
                     "avg_precision": round(float(average_precision_score(y, s)), 3),
                     "precision_top50": round(_prec_at(y, s, 50), 3),
                     "precision_top100": round(_prec_at(y, s, 100), 3)}
    res["base_rate"] = round(float(y.mean()), 4)
    return res, oof


def fit_and_explain(f: pd.DataFrame, top: int = 3) -> pd.DataFrame:
    """Fit on all vessels; risk = model probability; reasons = largest positive contributions (coef x z)."""
    X = design(f)
    m = _model().fit(X, f["y"].to_numpy())
    scaler, lr = m.steps[0][1], m.steps[1][1]
    contrib = scaler.transform(X) * lr.coef_[0]
    cols = list(X.columns)
    out = f.copy()
    out["risk"] = m.predict_proba(X)[:, 1]          # ranking score: class_weight="balanced" inflates it, so it is
    out["risk_top_pct"] = (out["risk"].rank(ascending=False, method="min") / len(out) * 100).round(2)  # shown
    reasons = []
    for row in contrib:
        idx = [i for i in np.argsort(-row) if row[i] > 0.25 and cols[i] != "length_m"][:top]
        txt = []
        for i in idx:
            t = REASONS[cols[i]]
            if cols[i] == "silent_moved_max_surprise":
                srp = float(f["silent_moved_max_surprise"].iloc[len(reasons)])
                t = ("an AIS silence rarer than 1 in 10,000 for its area, while moving" if srp >= SURPRISE_MIN
                     else f"AIS silences while moving (rarest about 1 in {10 ** srp:,.0f} for its area)")
            txt.append(t)
        reasons.append("; ".join(txt))
    out["reasons"] = reasons
    out["context"] = np.where(out["russian_port_h"] > 0,
                              "called at Russian Baltic ports (" + out["russian_port_h"].round().astype(int).astype(str)
                              + " h)", "")
    out.attrs["coef"] = dict(zip(cols, np.round(lr.coef_[0], 3), strict=True))
    return out


def run_risk(region, days: float = 7, end=None, log=print) -> pd.DataFrame:
    """Features for the last `days`, CV evaluation against the watchlist, fit, write risk/latest.parquet."""
    import json

    from maritime import sanctions
    from maritime.paths import PROCESSED
    from maritime.sar import run as RUN
    from maritime.sar.evidence import coverage_model
    end = pd.Timestamp(end) if end is not None else pd.Timestamp.now("UTC").tz_localize(None)
    start = end - pd.Timedelta(days=days)
    ais_dir = PROCESSED / region.name / "ais"
    wl = sanctions.load()
    if not len(wl):
        raise SystemExit("no watchlist: run `maritime sanctions update`")
    lp = RUN.land_path(region.name)
    land = load_land_coarse(lp) if lp.exists() else None
    if land is None:
        log(f"no land polygons ({lp}): loitering and meetings are not computed")
    cover = coverage_model(ais_dir, end, days=7)
    f = build_features(ais_dir, start, end, cover=cover, land=land,
                       evidence_path=RUN.sar_dir(region.name) / "evidence" / "latest.parquet", log=log)
    f = labels(f, wl)
    res, oof = evaluate(f)
    log(json.dumps(res, indent=1))
    out = fit_and_explain(f)
    out["risk_cv"] = oof                     # out-of-fold score, for honest inspection of listed vessels
    res["coef"] = {k: float(v) for k, v in out.attrs["coef"].items()}
    res.update({"start": str(start), "end": str(end)})
    d = PROCESSED / region.name / "risk"
    d.mkdir(parents=True, exist_ok=True)
    out.attrs = {}
    out.to_parquet(d / "latest.parquet", index=False)
    (d / "latest_report.json").write_text(json.dumps(res, indent=1, default=str))
    show = ["vessel_name", "mmsi", "imo", "risk", "listed_label", "reasons", "context"]
    log("\nhighest risk, not on any list:")
    log(out[out["listed_category"].isna()].nlargest(15, "risk")[show].to_string(index=False))
    log("\nhighest risk overall:")
    log(out.nlargest(10, "risk")[show].to_string(index=False))
    log(f"written to {d}")
    return out


def evaluate_later_week(region, split, days: float = 7, log=print) -> dict:
    """The honest test: fit on the `days` before `split`, score the `days` after it. Reported for all
    vessels in the test week and for vessels not seen in the training week (the model never saw them at all)."""
    import json

    from sklearn.metrics import average_precision_score, roc_auc_score

    from maritime import sanctions
    from maritime.paths import PROCESSED
    from maritime.sar import run as RUN
    from maritime.sar.evidence import coverage_model
    split = pd.Timestamp(split)
    ais_dir = PROCESSED / region.name / "ais"
    wl = sanctions.load()
    lp = RUN.land_path(region.name)
    land = load_land_coarse(lp) if lp.exists() else None
    ev = RUN.sar_dir(region.name) / "evidence" / "latest.parquet"
    weeks = {}
    for name, a, b in (("train", split - pd.Timedelta(days=days), split),
                       ("test", split, split + pd.Timedelta(days=days))):
        cover = coverage_model(ais_dir, b, days=7)
        weeks[name] = labels(build_features(ais_dir, a, b, cover=cover, land=land, evidence_path=ev, log=log), wl)
    tr, te = weeks["train"], weeks["test"]
    model = _model().fit(design(tr), tr["y"].to_numpy())
    te = te.assign(score=model.predict_proba(design(te))[:, 1],
                   rule=te["tanker"] * 2 + (te["russian_port_h"] > 0) * 2 + np.log1p(te["russian_port_h"]))
    res = {"split": str(split), "days": days, "train_vessels": int(len(tr)), "train_positives": int(tr["y"].sum())}
    for subset, d in (("all_test_vessels", te), ("vessels_not_in_train_week", te[~te["mmsi"].isin(tr["mmsi"])])):
        y = d["y"].to_numpy()
        r = {"vessels": int(len(d)), "positives": int(y.sum()), "base_rate": round(float(y.mean()), 4)}
        if 0 < y.sum() < len(y):
            for name in ("score", "rule"):
                r[name] = {"roc_auc": round(float(roc_auc_score(y, d[name])), 3),
                           "avg_precision": round(float(average_precision_score(y, d[name])), 3),
                           "precision_top50": round(_prec_at(y, d[name].to_numpy(), 50), 3)}
        res[subset] = r
    out = PROCESSED / region.name / "risk" / f"later_week_{split:%Y%m%d}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1))
    log(json.dumps(res, indent=1))
    log(f"written to {out}")
    return res
