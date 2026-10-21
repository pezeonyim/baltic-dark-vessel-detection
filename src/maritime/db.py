"""DuckDB connection with settings that keep 25M+ row window queries inside a laptop's
memory: a capped memory limit and an on-disk spill directory."""

from __future__ import annotations

import os
from pathlib import Path

import duckdb

from maritime.paths import DATA, region_ais_dir

CACHE = DATA / "cache"


def connect(path: Path | str | None = None, memory_limit: str | None = None) -> duckdb.DuckDBPyConnection:
    CACHE.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(path) if path else ":memory:")
    con.execute(f"SET memory_limit='{memory_limit or os.environ.get('MARITIME_DUCKDB_MEMORY', '2GB')}'")
    con.execute(f"SET temp_directory='{CACHE / 'duckdb_tmp'}'")
    con.execute("SET preserve_insertion_order=false")
    return con


def ais_source(region_name: str) -> str:
    """SQL table expression over a region's canonical AIS Parquet."""
    glob = region_ais_dir(region_name) / "*" / "*.parquet"
    return f"read_parquet('{glob}', hive_partitioning = true, union_by_name = true)"
