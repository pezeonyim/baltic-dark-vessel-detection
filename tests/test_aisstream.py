"""Message parsing for the aisstream recorder (no network)."""

from datetime import datetime

from maritime.live.aisstream import (
    StaticInfo,
    open_live_db,
    parse_time_utc,
    position_row,
    update_static,
    upsert_last_positions,
)
from maritime.schema import AIS_COLUMNS

NOW = datetime(2026, 10, 3, 12, 0, 0)

POS_A = {
    "MessageType": "PositionReport",
    "MetaData": {"MMSI": 273123456, "ShipName": "TEST TANKER  ", "time_utc": "2026-10-03 11:59:58.123456789 +0000 UTC"},
    "Message": {"PositionReport": {"UserID": 273123456, "Latitude": 55.6, "Longitude": 12.7,
                                   "Sog": 11.5, "Cog": 360.0, "TrueHeading": 511, "NavigationalStatus": 0}},
}
STATIC = {
    "MessageType": "ShipStaticData",
    "MetaData": {"MMSI": 273123456, "time_utc": "2026-10-03 11:59:00 +0000 UTC"},
    "Message": {"ShipStaticData": {"UserID": 273123456, "ImoNumber": 9123456, "CallSign": "UBAB2",
                                   "Name": "TEST TANKER", "Type": 80, "MaximumStaticDraught": 14.2,
                                   "Dimension": {"A": 200, "B": 50, "C": 20, "D": 24}}},
}
POS_B_NO_FIX = {
    "MessageType": "StandardClassBPositionReport",
    "MetaData": {"MMSI": 211000001},
    "Message": {"StandardClassBPositionReport": {"UserID": 211000001, "Latitude": 91, "Longitude": 181}},
}


def test_parse_time_utc():
    assert parse_time_utc("2026-10-03 11:59:58.123456789 +0000 UTC") == datetime(2026, 10, 3, 11, 59, 58, 123456)
    assert parse_time_utc("2026-10-03 11:59:00 +0000 UTC") == datetime(2026, 10, 3, 11, 59)
    assert parse_time_utc(None) is None and parse_time_utc("garbage") is None


def test_static_then_position_produces_canonical_row():
    cache: dict[int, StaticInfo] = {}
    update_static(cache, STATIC)
    row = position_row(POS_A, cache, NOW)
    assert list(row) == list(AIS_COLUMNS)
    assert row["imo"] == "9123456" and row["vessel_type"] == 80
    assert row["length"] == 250 and row["width"] == 44 and row["draught"] == 14.2
    assert row["cog"] is None and row["heading"] is None  # sentinels
    assert row["transceiver_class"] == "A" and row["source"] == "aisstream"


def test_position_without_fix_is_dropped():
    assert position_row(POS_B_NO_FIX, {}, NOW) is None


def test_last_position_keeps_newest(tmp_path, monkeypatch):
    monkeypatch.setattr("maritime.live.aisstream.DATA", tmp_path)
    con = open_live_db("t")
    row = position_row(POS_A, {}, NOW)
    upsert_last_positions(con, [row])
    older = dict(row, ts=datetime(2026, 10, 3, 11, 0), lat=0.0)
    upsert_last_positions(con, [older])
    assert con.execute("SELECT lat FROM last_position").fetchone()[0] == 55.6
