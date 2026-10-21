"""Nightly maintenance for live-recorded regions.

1. Compaction: the recorder writes a small Parquet file every minute. For every
   finished UTC day, merge that day's files (plus any earlier compacted file) into one
   deduplicated, sorted `part.parquet`, verify the row count, then remove the small
   files. ~3x smaller and much faster to query.
2. Live DB snapshot: copy the last-known-position SQLite with SQLite's online backup
   API (safe while the recorder is writing). Keeps the most recent 14.
"""

from __future__ import annotations

import sqlite3
from datetime import date, datetime, timezone
from pathlib import Path

import duckdb

from maritime.live.aisstream import live_db_path
from maritime.paths import DATA, PROCESSED

BACKUPS = DATA / "backups"


def _compact_day(day_dir: Path) -> tuple[int, int] | None:
    """Merge one date=YYYY-MM-DD folder. Returns (files merged, rows) or None if nothing to do."""
    streams = sorted(day_dir.glob("stream-*.parquet"))
    if not streams:
        return None
    part = day_dir / "part.parquet"
    inputs = [str(p) for p in streams] + ([str(part)] if part.exists() else [])
    tmp = day_dir / "part.parquet.tmp"
    con = duckdb.connect()
    try:
        src = f"read_parquet({inputs!r}, hive_partitioning = false, union_by_name = true)"
        expected = con.sql(f"SELECT count(*) FROM (SELECT DISTINCT * FROM {src})").fetchone()[0]
        con.execute(f"COPY (SELECT DISTINCT * FROM {src} ORDER BY ALL) "
                    f"TO '{tmp}' (FORMAT parquet, COMPRESSION zstd)")
        written = con.sql(f"SELECT count(*) FROM read_parquet('{tmp}')").fetchone()[0]
    finally:
        con.close()
    if written != expected:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"{day_dir}: wrote {written} rows, expected {expected}; left inputs alone")
    tmp.replace(part)
    for p in streams:
        p.unlink()
    return len(streams), written


def compact_region(region_name: str, today: date | None = None) -> list[str]:
    """Compact every finished UTC day (strictly before today) for ais/ and static/."""
    today = today or datetime.now(timezone.utc).date()
    log = []
    for kind in ("ais", "static"):
        base = PROCESSED / region_name / kind
        for day_dir in sorted(base.glob("date=*")):
            day = date.fromisoformat(day_dir.name.removeprefix("date="))
            if day >= today:
                continue
            res = _compact_day(day_dir)
            if res:
                log.append(f"{kind} {day}: merged {res[0]} files -> {res[1]:,} rows")
    return log


def snapshot_live_db(region_name: str, keep: int = 14) -> Path | None:
    src = live_db_path(region_name)
    if not src.exists():
        return None
    out_dir = BACKUPS / "live"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{region_name}-{datetime.now(timezone.utc):%Y%m%d}.sqlite"
    with sqlite3.connect(src) as s, sqlite3.connect(out) as d:
        s.backup(d)
    snaps = sorted(out_dir.glob(f"{region_name}-*.sqlite"))
    for old in snaps[:-keep]:
        old.unlink()
    return out
