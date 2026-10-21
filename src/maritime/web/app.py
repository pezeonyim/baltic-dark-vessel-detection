"""Map app v1 backend.

FastAPI serving:
  GET  /api/summary                 region, bbox, vessel counts by flag group, freshness
  GET  /api/vessels                 last known position of every vessel, with labels
  GET  /api/vessels/{mmsi}/track    recent track from the Parquet archive
  GET  /api/recent                  last few minutes of positions for moving vessels (wake trails)
  GET  /api/sar                     Sentinel-1 radar detections of the last few days + scene list
  WS   /ws                          pushes vessels whose position changed, every few seconds
  GET  /                            the map page (static/)

Reads the recorder's outputs only (SQLite last positions + Parquet), so it can run next
to the recorder or against previously recorded data.
"""

from __future__ import annotations

import asyncio
import sqlite3
from datetime import datetime, timedelta, timezone
from importlib import resources
from pathlib import Path

import duckdb
import pandas as pd
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from maritime import sanctions
from maritime.config import load_region
from maritime.labels import add_labels
from maritime.live.aisstream import live_db_path
from maritime.paths import PROCESSED, region_ais_dir
from maritime.web import guard

QUIET_AFTER_MIN = 30   # no position for this long -> shown as "last known", faded
# shown on the map; non_vessel, static (same spot on several passes) and outside_area are counted, not drawn
SAR_STATUSES = ("dark", "no_coverage", "ais", "ais_gap", "small")
# evidence verdicts (sar/evidence.py) applied to dark candidates on the map
EVIDENCE_TO_STATUS = {"explained": "ais_gap", "feed_outage": "no_coverage", "low_reception": "no_coverage"}


def sar_dir(region_name: str) -> Path:
    return PROCESSED / region_name / "sar"


def load_sar(region_name: str, days: float, now: datetime | None = None,
             with_evidence: bool = True) -> tuple[pd.DataFrame, list[dict]]:
    """Radar detections (dark / no_coverage / ais) from images taken in the last `days`, and the scene list."""
    import json

    from maritime.sar.scene_pipeline import STATIC_LOOKBACK_DAYS, refine_status, static_mask

    now = now or _now()
    since = now - timedelta(days=days)
    history = now - timedelta(days=max(days, STATIC_LOOKBACK_DAYS))     # the static check needs older passes too
    base = sar_dir(region_name)
    files = [p for d in sorted((base / "detections").glob("date=*"))
             if d.name[5:] >= history.date().isoformat() for p in sorted(d.glob("*.parquet"))]
    frames = [pd.read_parquet(p) for p in files]
    det = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if len(det):
        det = refine_status(det, static_mask(det))
        det = det[pd.to_datetime(det["time"]) >= since]
        if with_evidence:
            det = apply_evidence(det, base / "evidence" / "latest.parquet")
        det.attrs["all_counts"] = det["status"].value_counts().to_dict()
        det = det[det["status"].isin(SAR_STATUSES)].copy()
        det["age_h"] = ((now - pd.to_datetime(det["time"])).dt.total_seconds() / 3600).round(1)
        det["time"] = pd.to_datetime(det["time"]).dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    scenes = {}
    man = base / "manifest.jsonl"
    if man.exists():
        for line in man.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                if pd.Timestamp(r["start"]) >= pd.Timestamp(since):
                    scenes[r["product"]] = r              # a re-run replaces the earlier line
    keep = ["product", "start", "platform", "orbit", "detections", "ais_matched", "dark", "dark_high",
            "no_coverage", "vv_db"]
    return det, [{k: r.get(k) for k in keep} for r in sorted(scenes.values(), key=lambda r: r["start"])]

EVIDENCE_COLS = {"verdict": "verdict", "vessel_name": "ev_name", "mmsi": "ev_mmsi", "length": "ev_length",
                 "gap_h": "ev_gap_h", "watchlist": "ev_watchlist"}


def apply_evidence(det: pd.DataFrame, path: Path) -> pd.DataFrame:
    """Join the latest evidence verdicts (sar evidence, run after every SAR job) onto dark candidates by
    (product, chip). explained -> ais_gap; feed_outage / low_reception -> no_coverage; others stay dark with the
    verdict and the linked ship for the tooltip."""
    for c in EVIDENCE_COLS.values():
        det[c] = None
    if not path.exists() or "chip" not in det or det.empty:
        return det
    ev = pd.read_parquet(path)
    if ev.empty or "chip" not in ev:
        return det
    ev = ev.dropna(subset=["chip"]).astype({"chip": "int64"})
    cols = [c for c in EVIDENCE_COLS if c in ev]
    ev = ev[["product", "chip", *cols]].rename(columns=EVIDENCE_COLS).drop_duplicates(["product", "chip"])
    d = det.drop(columns=list(EVIDENCE_COLS.values())).reset_index(drop=True)
    d["_chip"] = pd.to_numeric(d["chip"], errors="coerce").astype("Int64")
    d = d.merge(ev.rename(columns={"chip": "_chip"}), on=["product", "_chip"], how="left").drop(columns="_chip")
    for c in EVIDENCE_COLS.values():
        if c not in d:
            d[c] = None
    dark = d["status"] == "dark"
    d.loc[~dark, list(EVIDENCE_COLS.values())] = None           # verdicts only describe dark candidates
    for verdict, status in EVIDENCE_TO_STATUS.items():
        hit = dark & (d["verdict"] == verdict)
        d.loc[hit, "status"] = status
        if status != "ais_gap":
            d.loc[hit, "tier"] = ""
    d.loc[d["status"] == "ais_gap", "tier"] = ""
    return d


TYPE_NAMES = [
    (30, 30, "Fishing"), (31, 32, "Towing"), (33, 33, "Dredging"), (34, 34, "Diving"),
    (35, 35, "Military"), (36, 36, "Sailing"), (37, 37, "Pleasure craft"),
    (40, 49, "High-speed craft"), (50, 50, "Pilot"), (51, 51, "Search and rescue"),
    (52, 52, "Tug"), (53, 53, "Port tender"), (55, 55, "Law enforcement"),
    (58, 58, "Medical"), (60, 69, "Passenger"), (70, 79, "Cargo"), (80, 89, "Tanker"),
]


def type_name(code) -> str:
    if code is None or pd.isna(code):
        return "Unknown"
    c = int(code)
    for lo, hi, name in TYPE_NAMES:
        if lo <= c <= hi:
            return name
    return "Other"


def _now() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


def _clean(df: pd.DataFrame) -> list[dict]:
    """JSON-safe records (NaN -> None)."""
    return df.astype(object).where(df.notna(), None).to_dict(orient="records")


class WatchlistCache:
    """Loads the OpenSanctions list once and reloads it when the nightly job replaces it."""

    def __init__(self, path: Path = sanctions.LIST_PATH):
        self.path, self._mtime, self.list = path, None, sanctions.Watchlist([])

    def get(self) -> sanctions.Watchlist:
        mtime = self.path.stat().st_mtime if self.path.exists() else None
        if mtime != self._mtime:
            try:
                self.list = sanctions.load(self.path)
            except (OSError, ValueError) as e:
                print(f"watchlist not loaded: {e}")
            self._mtime = mtime
        return self.list


LISTED_COLUMNS = ["listed", "listed_category", "listed_category_label", "listed_match",
                  "listed_name", "listed_risks", "listed_sources", "listed_url"]


def add_listed(df: pd.DataFrame, wl: sanctions.Watchlist) -> pd.DataFrame:
    matches = [wl.match(i, m) if len(wl) else None for i, m in zip(df["imo"], df["mmsi"], strict=True)]
    labels = pd.DataFrame([m or {"listed": False} for m in matches], index=df.index)
    for c in LISTED_COLUMNS:
        if c not in labels:
            labels[c] = None
    labels["listed"] = labels["listed"].fillna(False).astype(bool)
    return df.join(labels[LISTED_COLUMNS])


class VesselStore:
    """Read-only view over the recorder's SQLite last-position table."""

    COLUMNS = ["mmsi", "ts", "lat", "lon", "sog", "cog", "heading", "nav_status",
               "vessel_name", "imo", "vessel_type", "length", "transceiver_class", "updated_at"]

    def __init__(self, db_path: Path, watchlist: WatchlistCache | None = None):
        self.db_path = db_path
        self.watchlist = watchlist or WatchlistCache()

    def query(self, where: str = "", params: tuple = ()) -> pd.DataFrame:
        if not self.db_path.exists():
            return pd.DataFrame(columns=self.COLUMNS)
        uri = f"file:{self.db_path}?mode=ro"
        with sqlite3.connect(uri, uri=True, timeout=5) as con:
            return pd.read_sql(f"SELECT * FROM last_position {where}", con, params=params)

    def vessels(self, max_age_min: float | None = None, changed_since: str | None = None) -> pd.DataFrame:
        clauses, params = [], []
        if max_age_min is not None:
            clauses.append("ts >= ?")
            params.append((_now() - timedelta(minutes=max_age_min)).isoformat())
        if changed_since:
            clauses.append("updated_at > ?")
            params.append(changed_since)
        where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
        df = self.query(where, tuple(params))
        if df.empty:
            return df
        df = add_labels(df)
        df = add_listed(df, self.watchlist.get())
        ts = pd.to_datetime(df["ts"], format="ISO8601")
        df["age_min"] = ((_now() - ts).dt.total_seconds() / 60).round(1)
        df["quiet"] = df["age_min"] > QUIET_AFTER_MIN
        df["type_name"] = df["vessel_type"].map(type_name)
        return df


def create_app(region_name: str, public: bool | None = None) -> FastAPI:
    """`public` (default: env MARITIME_PUBLIC): rate limits, response caching, a live-connection cap and no API docs,
    for serving the map to the internet through a Cloudflare Tunnel (see web/guard.py)."""
    public = guard.is_public() if public is None else public
    region = load_region(region_name)
    store = VesselStore(live_db_path(region.name))
    ais_dir = region_ais_dir(region.name)
    docs = {"docs_url": None, "redoc_url": None, "openapi_url": None} if public else {"docs_url": "/api/docs"}
    app = FastAPI(title="maritime-anomaly map", **docs)
    app.add_middleware(guard.Guard, public=public)
    ws_limit = guard.WsLimiter()

    def cached(seconds):
        return guard.ttl_cache(seconds, enabled=public)

    @app.get("/api/summary")
    @cached(15)
    def summary():
        df = store.vessels(max_age_min=24 * 60)
        b = region.bbox
        return {
            "region": region.name,
            "display_name": region.display_name,
            "bbox": [b.min_lon, b.min_lat, b.max_lon, b.max_lat],
            "vessels": int(len(df)),
            "live": int((~df["quiet"]).sum()) if len(df) else 0,
            "by_group": df["flag_group"].value_counts().to_dict() if len(df) else {},
            "newest_ts": df["ts"].max() if len(df) else None,
            "listed": int(df["listed"].sum()) if len(df) else 0,
            "by_list_category": df["listed_category"].value_counts().to_dict() if len(df) else {},
            "watchlist": store.watchlist.get().meta,
            "quiet_after_min": QUIET_AFTER_MIN,
        }

    @app.get("/api/vessels")
    @cached(15)
    def vessels(max_age_hours: float = 24.0):
        max_age_hours = min(max(max_age_hours, 0.1), 48.0)
        return _clean(store.vessels(max_age_min=max_age_hours * 60))

    @app.get("/api/vessels/{mmsi}/track")
    @cached(60)
    def track(mmsi: int, hours: float = 12.0):
        hours = min(max(hours, 0.5), 72.0)
        since = _now() - timedelta(hours=hours)
        if not any(ais_dir.glob("*/*.parquet")):
            raise HTTPException(404, "no recorded data for this region yet")
        day_list = _days_since(since)
        con = duckdb.connect()
        try:
            rows = con.execute(f"""
                SELECT DISTINCT lon, lat, epoch(ts) AS t, sog
                FROM read_parquet('{ais_dir}/*/*.parquet', hive_partitioning = true,
                                  union_by_name = true)
                WHERE date IN ({day_list}) AND mmsi = ? AND ts >= ?
                ORDER BY t
            """, [mmsi, since]).fetchall()
        finally:
            con.close()
        return {"mmsi": mmsi, "hours": hours,
                "points": [[r[0], r[1], r[2], r[3]] for r in rows]}

    def _days_since(since: datetime) -> str:
        days = [(since + timedelta(days=i)).date().isoformat()
                for i in range((_now().date() - since.date()).days + 1)]
        return ", ".join(f"'{d}'" for d in days)

    @app.get("/api/recent")
    @cached(30)
    def recent(minutes: float = 10.0):
        """{mmsi: [[lon, lat, t_ms], ...]} for vessels moving >= 0.5 kn in the window."""
        minutes = min(max(minutes, 1.0), 60.0)
        since = _now() - timedelta(minutes=minutes)
        if not any(ais_dir.glob("*/*.parquet")):
            return {}
        con = duckdb.connect()
        try:
            rows = con.execute(f"""
                SELECT mmsi, list([lon, lat, epoch_ms(ts)::DOUBLE] ORDER BY ts) AS pts
                FROM (SELECT DISTINCT mmsi, ts, lon, lat
                      FROM read_parquet('{ais_dir}/*/*.parquet', hive_partitioning = true,
                                        union_by_name = true)
                      WHERE date IN ({_days_since(since)}) AND ts >= ? AND sog >= 0.5)
                GROUP BY mmsi
            """, [since]).fetchall()
        finally:
            con.close()
        return {str(m): pts[-40:] for m, pts in rows}

    @app.get("/api/sar")
    @cached(120)
    def sar(days: float = 3.0):
        days = min(max(days, 0.5), 14.0)
        det, scenes = load_sar(region.name, days)
        if len(det):
            ais_ids = [int(m) for m in det.loc[det["status"] == "ais", "mmsi"].unique()]
            names = {}
            if ais_ids:
                q = store.query(f"WHERE mmsi IN ({','.join('?' * len(ais_ids))})", tuple(ais_ids))
                names = dict(zip(q["mmsi"], q["vessel_name"], strict=True))
            det["vessel_name"] = det["mmsi"].map(names)
            cols = ["time", "age_h", "lat", "lon", "score", "length_rough_m", "status", "tier", "mmsi", "vessel_name",
                    "product", "platform", *EVIDENCE_COLS.values()]
            det = det.reindex(columns=cols)
        all_counts = det.attrs.get("all_counts", {}) if len(det) else {}
        return {"days": days, "scenes": scenes, "detections": _clean(det) if len(det) else [],
                "counts": det["status"].value_counts().to_dict() if len(det) else {},
                "all_counts": all_counts,
                "dark_high": int((det["tier"] == "high").sum()) if len(det) else 0}

    @app.websocket("/ws")
    async def ws(socket: WebSocket):
        ip = guard.client_ip(socket.headers.items(), socket.client.host if socket.client else None)
        if public and not ws_limit.acquire(ip):
            await socket.close(code=1013)      # try again later: the map falls back to polling
            return
        await socket.accept()
        last = _now().isoformat(timespec="seconds")
        try:
            while True:
                await asyncio.sleep(5)
                df = await asyncio.to_thread(store.vessels, None, last)
                if len(df):
                    last = df["updated_at"].max()
                    await socket.send_json({"type": "update", "vessels": _clean(df)})
                else:
                    await socket.send_json({"type": "heartbeat", "at": _now().isoformat()})
        except (WebSocketDisconnect, RuntimeError):
            return
        finally:
            if public:
                ws_limit.release(ip)

    static = resources.files("maritime.web").joinpath("static")
    app.mount("/static", StaticFiles(directory=str(static)), name="static")

    @app.get("/api/risk")
    @cached(300)
    def risk(limit: int = 20, unlisted: bool = False):
        """Vessel risk scores, highest first, with reasons; written nightly by `maritime risk`."""
        import json
        d = PROCESSED / region.name / "risk"
        f = d / "latest.parquet"
        if not f.exists():
            return {"vessels": [], "report": None}
        r = pd.read_parquet(f)
        if unlisted:
            r = r[r["listed_category"].isna()]
        r = r.nlargest(max(1, min(limit, 200)), "risk")
        cols = ["mmsi", "imo", "vessel_name", "risk", "risk_top_pct", "listed_category", "listed_label", "reasons",
                "context"]
        r = r.reindex(columns=cols).assign(risk=lambda x: x["risk"].round(3))
        rep = json.loads((d / "latest_report.json").read_text()) if (d / "latest_report.json").exists() else {}
        m = rep.get("model_cv", {})
        report = {"start": rep.get("start"), "end": rep.get("end"), "vessels": rep.get("vessels"),
                  "positives": rep.get("positives"), "base_rate": rep.get("base_rate"),
                  "precision_top50": m.get("precision_top50"), "avg_precision": m.get("avg_precision")}
        return {"vessels": _clean(r), "report": report}

    @app.get("/")
    def index():
        return FileResponse(str(static.joinpath("index.html")))

    return app
