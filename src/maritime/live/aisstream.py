"""Live AIS recorder for aisstream.io (https://aisstream.io/documentation).

Subscribes to a region's bbox and, until stopped:
- converts position reports to the canonical schema (src/maritime/schema.py) and
  appends them to Parquet under data/processed/<region>/ais/date=YYYY-MM-DD/
  (same layout as historical data, so every downstream tool works on both);
- keeps static data (name, IMO, type, dimensions, draught) per MMSI and attaches it
  to positions; also archives raw static messages under .../static/;
- upserts the latest position of every vessel into a SQLite table for the live map.

Run with:  maritime stream --region baltic
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sqlite3
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import websockets

from maritime.config import Region
from maritime.paths import DATA, PROCESSED, region_ais_dir
from maritime.schema import AIS_COLUMNS, COG_NA, HEADING_NA, SOG_NA

log = logging.getLogger("maritime.stream")

URL = "wss://stream.aisstream.io/v0/stream"
POSITION_TYPES = {
    "PositionReport": "A",
    "StandardClassBPositionReport": "B",
    "ExtendedClassBPositionReport": "B",
}
STATIC_TYPES = {"ShipStaticData", "StaticDataReport", "ExtendedClassBPositionReport"}
SUBSCRIBED = sorted(set(POSITION_TYPES) | STATIC_TYPES)


def live_db_path(region_name: str) -> Path:
    return DATA / "live" / f"{region_name}.sqlite"


# --------------------------------------------------------------------------- parsing

def parse_time_utc(s: str | None) -> datetime | None:
    """aisstream MetaData.time_utc looks like '2024-02-24 17:23:34.153395866 +0000 UTC'."""
    if not s:
        return None
    try:
        date_part, time_part = s.split()[:2]
        if "." in time_part:
            hms, frac = time_part.split(".")
            time_part = f"{hms}.{frac[:6]}"
        fmt = "%Y-%m-%d %H:%M:%S.%f" if "." in time_part else "%Y-%m-%d %H:%M:%S"
        return datetime.strptime(f"{date_part} {time_part}", fmt)  # naive UTC
    except ValueError:
        return None


def _clean(v: Any, na: float | None = None, upper: float | None = None) -> float | None:
    if v is None:
        return None
    v = float(v)
    if v < 0 or (na is not None and v >= na) or (upper is not None and v > upper):
        return None
    return v


def _dims(d: dict | None) -> tuple[float | None, float | None]:
    if not d:
        return None, None
    length = (d.get("A") or 0) + (d.get("B") or 0)
    width = (d.get("C") or 0) + (d.get("D") or 0)
    return (float(length) or None), (float(width) or None)


@dataclass
class StaticInfo:
    vessel_name: str | None = None
    imo: str | None = None
    callsign: str | None = None
    vessel_type: int | None = None
    length: float | None = None
    width: float | None = None
    draught: float | None = None


def update_static(cache: dict[int, StaticInfo], msg: dict) -> dict | None:
    """Merge a static message into the cache. Returns a flat record for archiving."""
    mtype = msg["MessageType"]
    meta = msg.get("MetaData", {})
    body = msg["Message"][mtype]
    mmsi = int(meta.get("MMSI") or body.get("UserID") or 0)
    if not mmsi:
        return None
    info = cache.setdefault(mmsi, StaticInfo())

    if mtype == "ShipStaticData":
        imo = body.get("ImoNumber")
        info.imo = str(imo) if imo else info.imo
        info.callsign = (body.get("CallSign") or "").strip() or info.callsign
        info.vessel_name = (body.get("Name") or "").strip() or info.vessel_name
        info.vessel_type = body.get("Type") or info.vessel_type
        length, width = _dims(body.get("Dimension"))
        info.length, info.width = length or info.length, width or info.width
        info.draught = _clean(body.get("MaximumStaticDraught")) or info.draught
    elif mtype == "StaticDataReport":
        a, b = body.get("ReportA") or {}, body.get("ReportB") or {}
        if a.get("Valid"):
            info.vessel_name = (a.get("Name") or "").strip() or info.vessel_name
        if b.get("Valid"):
            info.vessel_type = b.get("ShipType") or info.vessel_type
            info.callsign = (b.get("CallSign") or "").strip() or info.callsign
            length, width = _dims(b.get("Dimension"))
            info.length, info.width = length or info.length, width or info.width
    elif mtype == "ExtendedClassBPositionReport":
        info.vessel_name = (body.get("Name") or "").strip() or info.vessel_name
        info.vessel_type = body.get("Type") or info.vessel_type
        length, width = _dims(body.get("Dimension"))
        info.length, info.width = length or info.length, width or info.width

    ts = parse_time_utc(meta.get("time_utc")) or datetime.now(timezone.utc).replace(tzinfo=None)
    return {"mmsi": mmsi, "ts": ts,
            "message_type": mtype, "raw": json.dumps(body)}


def position_row(msg: dict, cache: dict[int, StaticInfo], received: datetime) -> dict | None:
    """Convert a position message to a canonical-schema row, or None if unusable."""
    mtype = msg["MessageType"]
    if mtype not in POSITION_TYPES:
        return None
    meta = msg.get("MetaData", {})
    body = msg["Message"][mtype]
    mmsi = int(meta.get("MMSI") or body.get("UserID") or 0)
    lat, lon = body.get("Latitude"), body.get("Longitude")
    if not mmsi or lat is None or lon is None or not (-90 <= lat <= 90) or not (-180 <= lon <= 180):
        return None  # 91/181 are the AIS "position not available" sentinels
    info = cache.get(mmsi, StaticInfo())
    return {
        "mmsi": mmsi,
        "ts": parse_time_utc(meta.get("time_utc")) or received,
        "lat": float(lat),
        "lon": float(lon),
        "sog": _clean(body.get("Sog"), na=SOG_NA),
        "cog": _clean(body.get("Cog"), na=COG_NA),
        "heading": _clean(body.get("TrueHeading"), na=HEADING_NA),
        "draught": info.draught,
        "vessel_name": info.vessel_name or (meta.get("ShipName") or "").strip() or None,
        "imo": info.imo,
        "callsign": info.callsign,
        "vessel_type": info.vessel_type,
        "nav_status": body.get("NavigationalStatus"),
        "length": info.length,
        "width": info.width,
        "transceiver_class": POSITION_TYPES[mtype],
        "source": "aisstream",
    }


# --------------------------------------------------------------------------- storage

LAST_POSITION_DDL = """
CREATE TABLE IF NOT EXISTS last_position (
    mmsi INTEGER PRIMARY KEY, ts TEXT, lat REAL, lon REAL, sog REAL, cog REAL,
    heading REAL, nav_status INTEGER, vessel_name TEXT, imo TEXT, vessel_type INTEGER,
    length REAL, transceiver_class TEXT, updated_at TEXT
)"""


def open_live_db(region_name: str) -> sqlite3.Connection:
    p = live_db_path(region_name)
    p.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(p)
    con.execute("PRAGMA journal_mode=WAL")  # lets the map read while we write
    con.execute(LAST_POSITION_DDL)
    return con


def upsert_last_positions(con: sqlite3.Connection, rows: list[dict]) -> None:
    latest: dict[int, dict] = {}
    for r in rows:
        if r["mmsi"] not in latest or r["ts"] >= latest[r["mmsi"]]["ts"]:
            latest[r["mmsi"]] = r
    now = datetime.now(timezone.utc).replace(tzinfo=None).isoformat(timespec="seconds")
    con.executemany(
        """INSERT INTO last_position VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
           ON CONFLICT(mmsi) DO UPDATE SET
             ts=excluded.ts, lat=excluded.lat, lon=excluded.lon, sog=excluded.sog,
             cog=excluded.cog, heading=excluded.heading, nav_status=excluded.nav_status,
             vessel_name=coalesce(excluded.vessel_name, vessel_name),
             imo=coalesce(excluded.imo, imo),
             vessel_type=coalesce(excluded.vessel_type, vessel_type),
             length=coalesce(excluded.length, length),
             transceiver_class=excluded.transceiver_class, updated_at=excluded.updated_at
           WHERE excluded.ts >= last_position.ts""",
        [(r["mmsi"], r["ts"].isoformat(), r["lat"], r["lon"], r["sog"], r["cog"],
          r["heading"], r["nav_status"], r["vessel_name"], r["imo"], r["vessel_type"],
          r["length"], r["transceiver_class"], now) for r in latest.values()],
    )
    con.commit()


def write_parquet(rows: list[dict], base: Path, columns: list[str] | None = None) -> list[Path]:
    """Append rows as new files partitioned by UTC date. Returns written paths."""
    if not rows:
        return []
    df = pd.DataFrame(rows, columns=columns)
    df["ts"] = pd.to_datetime(df["ts"]).astype("datetime64[us]")  # match historical data
    stamp = time.strftime("%H%M%S")
    written = []
    for day, part in df.groupby(df["ts"].dt.date):
        out = base / f"date={day.isoformat()}" / f"stream-{stamp}-{uuid.uuid4().hex[:8]}.parquet"
        out.parent.mkdir(parents=True, exist_ok=True)
        part.sort_values(["mmsi", "ts"]).to_parquet(out, index=False, compression="zstd")
        written.append(out)
    return written


# --------------------------------------------------------------------------- recorder

@dataclass
class Recorder:
    region: Region
    api_key: str
    flush_seconds: int = 60
    cache: dict[int, StaticInfo] = field(default_factory=dict)
    positions: list[dict] = field(default_factory=list)
    statics: list[dict] = field(default_factory=list)
    n_msgs: int = 0
    n_rows_total: int = 0

    def subscription(self) -> str:
        b = self.region.bbox
        return json.dumps({
            "APIKey": self.api_key,
            "BoundingBoxes": [[[b.min_lat, b.min_lon], [b.max_lat, b.max_lon]]],  # [lat, lon]
            "FilterMessageTypes": SUBSCRIBED,
        })

    def handle(self, raw: str | bytes) -> None:
        msg = json.loads(raw)
        if "error" in msg:
            raise RuntimeError(f"aisstream error: {msg['error']}")
        self.n_msgs += 1
        mtype = msg.get("MessageType")
        if mtype in STATIC_TYPES:
            rec = update_static(self.cache, msg)
            if rec:
                self.statics.append(rec)
        row = position_row(msg, self.cache, datetime.now(timezone.utc).replace(tzinfo=None))
        if row:
            self.positions.append(row)

    def flush(self, con: sqlite3.Connection) -> None:
        rows, statics = self.positions, self.statics
        self.positions, self.statics = [], []
        write_parquet(rows, region_ais_dir(self.region.name), columns=list(AIS_COLUMNS))
        write_parquet(statics, PROCESSED / self.region.name / "static")
        if rows:
            upsert_last_positions(con, rows)
        self.n_rows_total += len(rows)
        vessels = con.execute("SELECT count(*) FROM last_position").fetchone()[0]
        log.info("flushed %d positions (%d total), %d msgs, %d vessels known",
                 len(rows), self.n_rows_total, self.n_msgs, vessels)

    async def run(self) -> None:
        con = open_live_db(self.region.name)
        backoff = 1
        try:
            while True:
                try:
                    async with websockets.connect(URL, compression="deflate",
                                                  ping_interval=20, max_size=2**22) as ws:
                        await ws.send(self.subscription())
                        log.info("subscribed to %s bbox", self.region.name)
                        backoff = 1
                        last_flush = time.monotonic()
                        async for raw in ws:
                            self.handle(raw)
                            if time.monotonic() - last_flush >= self.flush_seconds:
                                self.flush(con)
                                last_flush = time.monotonic()
                except (websockets.ConnectionClosed, OSError) as e:
                    self.flush(con)
                    log.warning("connection lost (%s); reconnecting in %ds", e, backoff)
                    await asyncio.sleep(backoff)
                    backoff = min(backoff * 2, 60)
        finally:
            self.flush(con)
            con.close()


def load_api_key() -> str:
    from dotenv import load_dotenv

    from maritime.paths import ROOT
    load_dotenv(ROOT / ".env")
    key = os.environ.get("AISSTREAM_API_KEY", "").strip()
    if not key:
        raise SystemExit("AISSTREAM_API_KEY is not set (put it in .env at the repo root)")
    return key
