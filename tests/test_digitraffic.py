import time
from types import SimpleNamespace

import pandas as pd

from maritime.live import digitraffic as D

BBOX = SimpleNamespace(min_lat=53.0, max_lat=66.0, min_lon=9.0, max_lon=31.0)
REGION = SimpleNamespace(name="testregion", bbox=BBOX)
T_MS = int(time.time() * 1000) - 60_000      # a minute ago (the poller starts from "now - 10 min")


def _feat(mmsi, lon, lat, ms=T_MS, sog=12.3, cog=90.0, heading=88, nav=0):
    return {"type": "Feature", "mmsi": mmsi, "geometry": {"type": "Point", "coordinates": [lon, lat]},
            "properties": {"mmsi": mmsi, "sog": sog, "cog": cog, "heading": heading, "navStat": nav,
                           "timestampExternal": ms}}


def test_static_from_vessel_dimensions_and_draught():
    mmsi, info = D.static_from_vessel({"mmsi": 230000001, "name": "TESTER ", "imo": 9123456, "callSign": "OJAB",
                                       "shipType": 80, "draught": 118, "referencePointA": 148,
                                       "referencePointB": 35, "referencePointC": 23, "referencePointD": 9})
    assert mmsi == 230000001 and info.vessel_name == "TESTER" and info.imo == "9123456"
    assert info.length == 183 and info.width == 32 and info.draught == 11.8 and info.vessel_type == 80
    _, unknown = D.static_from_vessel({"mmsi": 1, "draught": 0})
    assert unknown.length is None and unknown.draught is None


def test_position_row_sentinels_bbox_and_static():
    cache = {230000001: D.StaticInfo(vessel_name="TESTER", length=183.0)}
    r = D.position_from_feature(_feat(230000001, 25.7, 59.84, sog=102.3, cog=360, heading=511), cache, BBOX)
    assert r["sog"] is None and r["cog"] is None and r["heading"] is None
    assert r["vessel_name"] == "TESTER" and r["length"] == 183.0
    assert r["source"] == "digitraffic" and r["transceiver_class"] == "A"
    assert r["ts"] == pd.Timestamp(T_MS, unit="ms").to_pydatetime()
    assert D.position_from_feature(_feat(2, 5.0, 59.0), cache, BBOX) is None          # outside the region


class FakeSession:
    def __init__(self, pages):
        self.pages, self.headers, self.calls = pages, {}, []

    def get(self, url, params=None, timeout=None):
        self.calls.append((url.rsplit("/", 1)[-1], dict(params or {})))
        body = self.pages.pop(0)
        return SimpleNamespace(status_code=200, json=lambda: body, raise_for_status=lambda: None)


def test_poll_dedupes_overlap_and_advances(tmp_path, monkeypatch):
    first = {"features": [_feat(1, 25.0, 60.0, T_MS), _feat(2, 26.0, 60.1, T_MS + 5_000)]}
    second = {"features": [_feat(2, 26.0, 60.1, T_MS + 5_000), _feat(2, 26.01, 60.1, T_MS + 65_000)]}
    p = D.Poller(REGION, session=FakeSession([first, second]))
    assert p.session.headers["Digitraffic-User"] == D.USER
    assert len(p.poll_locations()) == 2
    assert p.since_ms == T_MS + 5_000 - D.OVERLAP_MS
    rows = p.poll_locations()                       # the repeated report is dropped, the new one kept
    assert [(r["mmsi"], r["ts"].second) for r in rows] == [(2, pd.Timestamp(T_MS + 65_000, unit="ms").second)]
    assert p.session.calls[1] == ("locations", {"from": T_MS + 5_000 - D.OVERLAP_MS})

    monkeypatch.setattr(D, "region_ais_dir", lambda name: tmp_path / "ais")
    monkeypatch.setattr("maritime.live.aisstream.live_db_path", lambda name: tmp_path / "live.sqlite")
    con = D.open_live_db("testregion")
    assert p.flush(con) == 3
    df = pd.concat(pd.read_parquet(f) for f in (tmp_path / "ais").glob("date=*/stream-*.parquet"))
    assert len(df) == 3 and set(df["source"]) == {"digitraffic"}
    assert con.execute("SELECT count(*) FROM last_position").fetchone()[0] == 2
