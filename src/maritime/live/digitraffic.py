"""Second live AIS source: Fintraffic Digitraffic, https://www.digitraffic.fi/en/marine-traffic/

Finland's open AIS feed from its shore stations (class A only, fishing vessels removed). Strongest where aisstream
is weakest: Gulf of Finland, Archipelago Sea, Gulf of Bothnia, northern Baltic.
Licence CC BY 4.0: "Source: Fintraffic / digitraffic.fi, license CC 4.0 BY".

Polls the REST API (no extra dependency): vessel locations every POLL_S seconds, vessel metadata every META_S.
Writes canonical rows (source='digitraffic') into the same date partitions and SQLite table as the aisstream
recorder, so everything downstream reads both feeds.

    maritime digitraffic --region baltic          (what the systemd service runs)
    maritime digitraffic --region baltic --once   (one poll, print a summary: a connectivity check)
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone

import requests

from maritime.config import Region
from maritime.live.aisstream import StaticInfo, open_live_db, upsert_last_positions, write_parquet
from maritime.paths import region_ais_dir
from maritime.schema import AIS_COLUMNS, COG_NA, HEADING_NA, SOG_NA

log = logging.getLogger("maritime.digitraffic")

BASE = "https://meri.digitraffic.fi/api/ais/v1"
USER = "maritime-anomaly/0.1"      # Digitraffic-User header: app name, no personal data (their rule)
POLL_S = 60                        # responses are cached ~1 min, polling faster returns the same data
META_S = 600
OVERLAP_MS = 120_000               # re-ask the last 2 min each poll; exact duplicates are dropped
FLUSH_S = 300
ATTRIBUTION = "Source: Fintraffic / digitraffic.fi, license CC 4.0 BY"


def _utc(ms) -> datetime | None:
    if ms is None:
        return None
    return datetime.fromtimestamp(int(ms) / 1000, tz=timezone.utc).replace(tzinfo=None)


def _na(v, na: float | None = None, upper: float | None = None) -> float | None:
    if v is None:
        return None
    v = float(v)
    if v < 0 or (na is not None and v >= na) or (upper is not None and v > upper):
        return None
    return v


def static_from_vessel(v: dict) -> tuple[int, StaticInfo] | None:
    """One /vessels record -> (mmsi, StaticInfo). Dimensions: A+B = length, C+D = width (metres)."""
    mmsi = int(v.get("mmsi") or 0)
    if not mmsi:
        return None
    a, b = v.get("referencePointA") or 0, v.get("referencePointB") or 0
    c, d = v.get("referencePointC") or 0, v.get("referencePointD") or 0
    draught = v.get("draught") or 0
    imo = v.get("imo")
    return mmsi, StaticInfo(
        vessel_name=(v.get("name") or "").strip() or None,
        imo=str(imo) if imo else None,
        callsign=(v.get("callSign") or "").strip() or None,
        vessel_type=v.get("shipType") or None,
        length=float(a + b) or None,
        width=float(c + d) or None,
        draught=draught / 10 if 0 < draught < 255 else None)


def position_from_feature(f: dict, cache: dict[int, StaticInfo], bbox) -> dict | None:
    """One /locations GeoJSON feature -> canonical row (None if unusable or outside the region)."""
    p = f.get("properties") or {}
    mmsi = int(f.get("mmsi") or p.get("mmsi") or 0)
    coords = (f.get("geometry") or {}).get("coordinates") or []
    if not mmsi or len(coords) < 2:
        return None
    lon, lat = float(coords[0]), float(coords[1])
    if not (bbox.min_lat <= lat <= bbox.max_lat and bbox.min_lon <= lon <= bbox.max_lon):
        return None
    ts = _utc(p.get("timestampExternal"))
    if ts is None:
        return None
    info = cache.get(mmsi, StaticInfo())
    return {
        "mmsi": mmsi, "ts": ts, "lat": lat, "lon": lon,
        "sog": _na(p.get("sog"), na=SOG_NA),
        "cog": _na(p.get("cog"), na=COG_NA),
        "heading": _na(p.get("heading"), na=HEADING_NA),
        "draught": info.draught, "vessel_name": info.vessel_name, "imo": info.imo, "callsign": info.callsign,
        "vessel_type": info.vessel_type, "nav_status": p.get("navStat"),
        "length": info.length, "width": info.width,
        "transceiver_class": "A",               # Digitraffic publishes class A messages only
        "source": "digitraffic",
    }


@dataclass
class Poller:
    region: Region
    session: requests.Session = field(default_factory=requests.Session)
    cache: dict[int, StaticInfo] = field(default_factory=dict)
    since_ms: int | None = None
    meta_since_ms: int | None = None
    seen: set = field(default_factory=set)          # (mmsi, ts) already written, for the overlap window
    buffer: list[dict] = field(default_factory=list)
    n_total: int = 0

    def __post_init__(self):
        self.session.headers.update({"Digitraffic-User": USER, "Accept-Encoding": "gzip"})

    def _get(self, path: str, **params):
        r = self.session.get(f"{BASE}/{path}", params=params, timeout=60)
        if r.status_code == 429:
            raise RuntimeError("Digitraffic rate limit (429)")
        r.raise_for_status()
        return r.json()

    def poll_metadata(self) -> int:
        now_ms = int(time.time() * 1000)
        params = {"from": self.meta_since_ms} if self.meta_since_ms else {}
        n = 0
        for v in self._get("vessels", **params):
            got = static_from_vessel(v)
            if got:
                self.cache[got[0]] = got[1]
                n += 1
        self.meta_since_ms = now_ms - OVERLAP_MS
        return n

    def poll_locations(self) -> list[dict]:
        now_ms = int(time.time() * 1000)
        frm = self.since_ms or now_ms - 10 * 60_000
        data = self._get("locations", **{"from": frm})
        rows = []
        for f in data.get("features", []):
            r = position_from_feature(f, self.cache, self.region.bbox)
            if r is None or (r["mmsi"], r["ts"]) in self.seen:
                continue
            self.seen.add((r["mmsi"], r["ts"]))
            rows.append(r)
        newest = max((int(f["properties"]["timestampExternal"]) for f in data.get("features", [])
                      if (f.get("properties") or {}).get("timestampExternal")), default=frm)
        self.since_ms = max(frm, newest - OVERLAP_MS)
        cutoff = _utc(self.since_ms - OVERLAP_MS)
        self.seen = {k for k in self.seen if k[1] >= cutoff}
        self.buffer.extend(rows)
        return rows

    def flush(self, con) -> int:
        rows, self.buffer = self.buffer, []
        write_parquet(rows, region_ais_dir(self.region.name), columns=list(AIS_COLUMNS))
        if rows:
            upsert_last_positions(con, rows)
        self.n_total += len(rows)
        return len(rows)

    def run(self, once: bool = False) -> None:
        con = open_live_db(self.region.name)
        con.execute("PRAGMA busy_timeout = 30000")    # the aisstream recorder writes the same table
        last_meta = last_flush = 0.0
        backoff = POLL_S
        try:
            while True:
                t0 = time.monotonic()
                try:
                    if t0 - last_meta >= META_S:
                        n = self.poll_metadata()
                        last_meta = t0
                        log.info("metadata: %d vessels (%d known)", n, len(self.cache))
                    rows = self.poll_locations()
                    backoff = POLL_S
                    if once:
                        mmsis = {r["mmsi"] for r in rows}
                        print(f"{len(rows)} positions from {len(mmsis)} ships inside {self.region.name}; "
                              f"{len(self.cache)} vessels with metadata; newest {max((r['ts'] for r in rows), default=None)}")
                        if rows:
                            lat = [r["lat"] for r in rows]
                            lon = [r["lon"] for r in rows]
                            print(f"extent lat {min(lat):.2f}..{max(lat):.2f}, lon {min(lon):.2f}..{max(lon):.2f}")
                        return
                    if time.monotonic() - last_flush >= FLUSH_S:
                        n = self.flush(con)
                        last_flush = time.monotonic()
                        log.info("flushed %d positions (%d total)", n, self.n_total)
                except (requests.RequestException, RuntimeError, ValueError) as e:
                    backoff = min(backoff * 2, 900)
                    log.warning("poll failed (%s); retrying in %d s", e, backoff)
                    time.sleep(backoff)
                    continue
                time.sleep(max(0.0, POLL_S - (time.monotonic() - t0)))
        finally:
            if not once:
                self.flush(con)
            con.close()
