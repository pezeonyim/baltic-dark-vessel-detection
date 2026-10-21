"""AIS gap detection.

A gap is a silence that is long *relative to how often the vessel should report*.

Expected interval = max(spec interval, source floor):
- Spec interval (ITU-R M.1371) depends on class, speed and nav status, judged from the
  last ping before the silence: class A moored/at anchor and < 3 kn -> 180 s; otherwise
  10 s (< 14 kn), 6 s (14-23 kn), 2 s (> 23 kn). Class B -> 180 s at <= 2 kn, else 30 s.
- Source floor: providers thin the stream. MarineCadastre keeps at most ~1 ping per
  vessel per minute (5th percentile interval is 61 s in every speed band), so nothing
  can be "expected" faster than 60 s.

A silence is a gap when it exceeds k x expected. With k = 30: 30 min for a ship
underway, 90 min for one at anchor (Gulf week 1, class A underway p99.9 is ~45 min).

Outages: time inside region-wide outages is subtracted first. The gap is kept only if
the silence *outside* outages still exceeds k x expected, so a ship that was simply
unheard during a network failure is not flagged.
"""

from __future__ import annotations

import duckdb
import pandas as pd

SOURCE_FLOOR_SECONDS = {"marinecadastre": 60, "aisstream": 60}  # aisstream: provisional

SPEC_INTERVAL_SQL = """
CASE
  WHEN p_cls = 'B' THEN CASE WHEN coalesce(p_sog, 0) <= 2 THEN 180 ELSE 30 END
  WHEN p_nav IN (1, 5) AND coalesce(p_sog, 0) < 3 THEN 180
  WHEN p_sog > 23 THEN 2
  WHEN p_sog > 14 THEN 6
  ELSE 10
END"""

def haversine_km_sql(lat1: str, lon1: str, lat2: str, lon2: str) -> str:
    """Great-circle distance in km between two points, as a SQL expression."""
    return f"""2 * 6371.0088 * asin(sqrt(
        pow(sin(radians({lat2} - {lat1}) / 2), 2)
      + cos(radians({lat1})) * cos(radians({lat2})) * pow(sin(radians({lon2} - {lon1}) / 2), 2)))"""


def detect_gaps(con: duckdb.DuckDBPyConnection, source: str, source_name: str,
                outage_intervals: pd.DataFrame | None = None, k: float = 30.0,
                classes: tuple[str, ...] = ("A",)) -> pd.DataFrame:
    floor = SOURCE_FLOOR_SECONDS[source_name]
    cls_list = ", ".join(f"'{c}'" for c in classes)

    out = outage_intervals if outage_intervals is not None and len(outage_intervals) else \
        pd.DataFrame({"start": pd.Series(dtype="datetime64[us]"),
                      "end": pd.Series(dtype="datetime64[us]")})
    con.register("outage_df", out[["start", "end"]])

    return con.sql(f"""
        WITH pairs AS (
            SELECT mmsi, ts, lat, lon, sog, nav_status, vessel_type, transceiver_class AS cls,
                   lag(ts)  OVER w AS p_ts,  lag(lat) OVER w AS p_lat, lag(lon) OVER w AS p_lon,
                   lag(sog) OVER w AS p_sog, lag(nav_status) OVER w AS p_nav,
                   lag(transceiver_class) OVER w AS p_cls
            FROM {source}
            WHERE transceiver_class IN ({cls_list})
            WINDOW w AS (PARTITION BY mmsi ORDER BY ts)
        ),
        candidates AS (
            SELECT *, epoch(ts - p_ts) AS silence_s,
                   greatest({SPEC_INTERVAL_SQL}, {floor}) AS expected_s
            FROM pairs
            WHERE p_ts IS NOT NULL
        ),
        long AS (
            SELECT * FROM candidates WHERE silence_s > {k} * expected_s
        ),
        with_outage AS (
            -- CASE guard matters: DuckDB's least()/greatest() skip NULLs, so without it
            -- an unmatched gap would count its whole length as outage time.
            SELECT l.mmsi, l.p_ts, coalesce(sum(CASE WHEN o.start IS NULL THEN 0 ELSE epoch(
                       least(l.ts, o."end") - greatest(l.p_ts, o.start)) END), 0) AS outage_s
            FROM long l LEFT JOIN outage_df o ON o.start < l.ts AND o."end" > l.p_ts
            GROUP BY 1, 2
        )
        SELECT l.mmsi, l.cls AS transceiver_class, l.vessel_type,
               l.p_ts AS start_ts, l.ts AS end_ts,
               l.p_lat AS start_lat, l.p_lon AS start_lon, l.lat AS end_lat, l.lon AS end_lon,
               l.p_sog AS start_sog, l.p_nav AS start_nav_status,
               l.expected_s, l.silence_s / 60 AS silence_min, w.outage_s / 60 AS outage_min,
               (l.silence_s - w.outage_s) / l.expected_s AS ratio_outside_outage,
               {haversine_km_sql('l.p_lat', 'l.p_lon', 'l.lat', 'l.lon')} AS distance_km
        FROM long l JOIN with_outage w USING (mmsi, p_ts)
        WHERE l.silence_s - w.outage_s > {k} * l.expected_s
        ORDER BY start_ts
    """).df().assign(
        implied_speed_kn=lambda d: d["distance_km"] / (d["silence_min"] / 60) / 1.852
    )
