"""The canonical AIS schema.

Every source adapter must produce exactly these columns. Everything downstream
(features, SAR fusion, risk) reads only this schema, which is what makes regions and
data sources swappable.

Sentinel values from the AIS spec ("not available") are converted to NULL at ingest so
no downstream code has to know about them.
"""

from __future__ import annotations

# column name -> DuckDB type
AIS_COLUMNS: dict[str, str] = {
    "mmsi": "BIGINT",            # Maritime Mobile Service Identity (transponder ID, not a ship ID)
    "ts": "TIMESTAMP",           # UTC
    "lat": "DOUBLE",
    "lon": "DOUBLE",
    "sog": "DOUBLE",             # speed over ground, knots
    "cog": "DOUBLE",             # course over ground, degrees
    "heading": "DOUBLE",         # true heading, degrees
    "draught": "DOUBLE",         # meters, self-reported
    "vessel_name": "VARCHAR",
    "imo": "VARCHAR",            # IMO number (hull ID, more stable than MMSI)
    "callsign": "VARCHAR",
    "vessel_type": "INTEGER",    # AIS ship-type code
    "nav_status": "INTEGER",     # AIS navigational status code
    "length": "DOUBLE",          # meters
    "width": "DOUBLE",           # meters
    "transceiver_class": "VARCHAR",  # 'A' (commercial, SOLAS) or 'B' (small craft); NULL if unknown
    "source": "VARCHAR",         # which adapter produced the row
}

# AIS spec "not available" sentinels
SOG_NA = 102.3
COG_NA = 360.0
HEADING_NA = 511.0


def validate_columns(columns: list[str]) -> None:
    """Raise if a produced table does not match the canonical schema exactly."""
    expected = list(AIS_COLUMNS)
    if list(columns) != expected:
        missing = set(expected) - set(columns)
        extra = set(columns) - set(expected)
        raise ValueError(
            f"schema mismatch; missing={sorted(missing)} extra={sorted(extra)} "
            f"(order must also match)"
        )
