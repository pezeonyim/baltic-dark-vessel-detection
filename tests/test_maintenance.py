"""Compaction merges minute files for finished days only and loses nothing."""

from datetime import date

import pandas as pd

import maritime.maintenance as mt


def write(day_dir, name, rows):
    day_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(day_dir / name, index=False)


def test_compaction(tmp_path, monkeypatch):
    monkeypatch.setattr(mt, "PROCESSED", tmp_path)
    old = tmp_path / "r" / "ais" / "date=2026-10-03"
    new = tmp_path / "r" / "ais" / "date=2026-10-04"
    write(old, "stream-1.parquet", [{"mmsi": 1, "ts": pd.Timestamp("2026-10-03 10:00")}])
    write(old, "stream-2.parquet", [{"mmsi": 1, "ts": pd.Timestamp("2026-10-03 10:00")},   # dup
                                    {"mmsi": 2, "ts": pd.Timestamp("2026-10-03 11:00")}])
    write(new, "stream-3.parquet", [{"mmsi": 3, "ts": pd.Timestamp("2026-10-04 01:00")}])

    log = mt.compact_region("r", today=date(2026, 10, 4))
    assert log == ["ais 2026-10-03: merged 2 files -> 2 rows"]
    assert [p.name for p in old.iterdir()] == ["part.parquet"]
    assert len(pd.read_parquet(old / "part.parquet")) == 2
    assert [p.name for p in new.iterdir()] == ["stream-3.parquet"]        # today untouched

    # a late file for an already-compacted day is merged into the existing part
    write(old, "stream-4.parquet", [{"mmsi": 4, "ts": pd.Timestamp("2026-10-03 23:59")}])
    mt.compact_region("r", today=date(2026, 10, 5))
    assert len(pd.read_parquet(old / "part.parquet")) == 3
