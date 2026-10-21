"""Map API against a synthetic last-position DB and Parquet track."""

from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

import maritime.web.app as webapp  # noqa: E402
from maritime.live.aisstream import open_live_db, upsert_last_positions  # noqa: E402


@pytest.fixture
def client(tmp_path, monkeypatch):
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    monkeypatch.setattr("maritime.live.aisstream.DATA", tmp_path)
    ais = tmp_path / "ais"
    monkeypatch.setattr(webapp, "region_ais_dir", lambda name: ais)

    con = open_live_db("baltic")
    base = dict(sog=10.0, cog=90.0, heading=None, nav_status=0, imo=None,
                length=None, transceiver_class="A")
    upsert_last_positions(con, [
        dict(base, mmsi=273000001, ts=now - timedelta(minutes=2), lat=59.0, lon=24.0,
             vessel_name="<b>RUSSIAN CARGO</b>", vessel_type=70),
        dict(base, mmsi=211000002, ts=now - timedelta(hours=3), lat=54.5, lon=10.0,
             vessel_name="GERMAN NAVY", vessel_type=35),
    ])
    day = ais / f"date={now.date().isoformat()}"
    day.mkdir(parents=True)
    pd.DataFrame({"mmsi": [273000001] * 3, "ts": [now - timedelta(minutes=m) for m in (30, 20, 10)],
                  "lat": [58.9, 58.95, 59.0], "lon": [23.8, 23.9, 24.0], "sog": [10.0] * 3}
                 ).to_parquet(day / "part.parquet")
    wl = tmp_path / "wl.csv"
    wl.write_text("type,caption,imo,risk,mmsi,datasets,url\n"
                  "Vessel,RUSSIAN CARGO,,sanction;mare.shadow,273000001,ua_war_sanctions,"
                  "https://www.opensanctions.org/entities/X/\n")
    monkeypatch.setattr(webapp.sanctions, "LIST_PATH", wl)
    monkeypatch.setattr(webapp.WatchlistCache.__init__, "__defaults__", (wl,))
    return TestClient(webapp.create_app("baltic"))


def test_vessels_have_labels_and_age(client):
    vs = {v["mmsi"]: v for v in client.get("/api/vessels").json()}
    ru, de = vs[273000001], vs[211000002]
    assert ru["flag_group"] == "russia" and ru["type_name"] == "Cargo" and not ru["quiet"]
    assert de["flag_group"] == "nato" and de["is_military"] and de["quiet"]
    assert ru["vessel_name"] == "<b>RUSSIAN CARGO</b>"   # raw; the page renders it as text


def test_summary(client):
    s = client.get("/api/summary").json()
    assert s["vessels"] == 2 and s["live"] == 1
    assert s["by_group"] == {"russia": 1, "nato": 1}


def test_track(client):
    t = client.get("/api/vessels/273000001/track?hours=1").json()
    assert len(t["points"]) == 3 and t["points"][0][:2] == [23.8, 58.9]


def test_index_and_static(client):
    assert "Maritime Watch" in client.get("/").text
    assert client.get("/static/app.js").status_code == 200


def test_recent_trails(client):
    r = client.get("/api/recent?minutes=40").json()
    assert list(r) == ["273000001"] and len(r["273000001"]) == 3
    assert r["273000001"][0][:2] == [23.8, 58.9]


def test_listed_vessels(client):
    vs = {v["mmsi"]: v for v in client.get("/api/vessels").json()}
    ru, de = vs[273000001], vs[211000002]
    assert ru["listed"] and ru["listed_match"] == "mmsi" and "Shadow fleet" in ru["listed_risks"]
    assert not de["listed"]
    assert client.get("/api/summary").json()["listed"] == 1


def test_sar_endpoint(client, tmp_path, monkeypatch):
    import json
    monkeypatch.setattr(webapp, "PROCESSED", tmp_path / "processed")
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    d = tmp_path / "processed" / "baltic" / "sar" / "detections" / f"date={now.date().isoformat()}"
    d.mkdir(parents=True)
    t = now - timedelta(hours=2)
    pd.DataFrame({"product": "S1C_X", "platform": "S1C", "time": [t] * 4, "lat": [59.0, 59.1, 59.2, 59.3],
                  "lon": [24.0] * 4, "score": [0.9, 0.7, 0.5, 0.8], "p_vessel": 0.9, "p_fishing": 0.1,
                  "length_m": 50.0, "status": ["ais", "dark", "no_coverage", "non_vessel"],
                  "tier": ["", "high", "", ""], "mmsi": [273000001, -1, -1, -1], "ais_dt_s": 10.0,
                  "row": 1, "col": 1}).to_parquet(d / "S1C_X.parquet")
    man = tmp_path / "processed" / "baltic" / "sar" / "manifest.jsonl"
    row = {"product": "S1C_X", "start": t.isoformat(), "platform": "S1C", "orbit": "ASCENDING", "detections": 4,
           "ais_matched": 1, "dark": 1, "dark_high": 1, "no_coverage": 1, "vv_db": -18.0}
    man.write_text(json.dumps(dict(row, dark=9)) + "\n" + json.dumps(row) + "\n")   # rerun replaces the first
    r = client.get("/api/sar?days=2").json()
    assert len(r["detections"]) == 3 and r["dark_high"] == 1          # non_vessel not shown
    assert r["counts"] == {"ais": 1, "dark": 1, "no_coverage": 1}
    ais = next(x for x in r["detections"] if x["status"] == "ais")
    assert ais["vessel_name"] == "<b>RUSSIAN CARGO</b>" and ais["age_h"] == 2.0
    assert len(r["scenes"]) == 1 and r["scenes"][0]["dark"] == 1


def test_sar_endpoint_empty(client, tmp_path, monkeypatch):
    monkeypatch.setattr(webapp, "PROCESSED", tmp_path / "nothing")
    r = client.get("/api/sar").json()
    assert r["detections"] == [] and r["scenes"] == [] and r["dark_high"] == 0


def test_apply_evidence_maps_verdicts(tmp_path):
    import pandas as pd

    from maritime.web.app import apply_evidence
    det = pd.DataFrame({"product": "P", "chip": [0, 1, 2, 3, 4], "tier": ["high", "high", "possible", "high", ""],
                        "status": ["dark", "dark", "dark", "dark", "ais"]})
    ev = pd.DataFrame({"product": "P", "chip": [0.0, 1.0, 2.0, 4.0], "verdict": ["explained", "feed_outage",
                       "went_silent", "explained"], "vessel_name": ["A", "B", "C", "D"], "mmsi": [1, 2, 3, 4],
                       "length": [180.0, 90.0, 250.0, 100.0], "gap_h": [2.0, 7.0, 9.0, 1.0],
                       "watchlist": [None, None, "Shadow fleet", None]})
    ev.to_parquet(tmp_path / "latest.parquet")
    out = apply_evidence(det.copy(), tmp_path / "latest.parquet")
    assert list(out["status"]) == ["ais_gap", "no_coverage", "dark", "dark", "ais"]
    assert list(out["tier"]) == ["", "", "possible", "high", ""]
    assert out.loc[2, "ev_name"] == "C" and out.loc[2, "ev_watchlist"] == "Shadow fleet"
    assert out.loc[3, "verdict"] is None or pd.isna(out.loc[3, "verdict"])      # no verdict for chip 3
    assert out.loc[4, "verdict"] is None or pd.isna(out.loc[4, "verdict"])      # not dark: verdict ignored
    missing = apply_evidence(det.copy(), tmp_path / "nope.parquet")
    assert list(missing["status"]) == list(det["status"])


def test_public_mode_hides_docs_and_sets_headers(client, monkeypatch):
    app = webapp.create_app("baltic", public=True)
    c = TestClient(app)
    assert c.get("/api/docs").status_code == 404
    assert c.get("/openapi.json").status_code == 404
    r = c.get("/api/summary")
    assert r.status_code == 200
    assert r.headers["X-Content-Type-Options"] == "nosniff"
    assert r.headers["X-Frame-Options"] == "DENY"


def test_private_mode_keeps_docs(client):
    assert client.get("/api/docs").status_code == 200


def test_rate_limit_per_visitor():
    from maritime.web.guard import TokenBucket
    b = TokenBucket(rate_per_min=60, burst=3)
    assert [b.allow("a", now=0.0) for _ in range(4)] == [True, True, True, False]
    assert b.allow("b", now=0.0)              # other visitors are not affected
    assert b.allow("a", now=1.0)              # refills at 1 per second


def test_public_mode_returns_429(client):
    from maritime.web import guard
    app = webapp.create_app("baltic", public=True)
    app.user_middleware.clear()
    app.add_middleware(guard.Guard, public=True, bucket=guard.TokenBucket(rate_per_min=1, burst=2))
    c = TestClient(app)
    codes = [c.get("/api/summary", headers={"CF-Connecting-IP": "1.2.3.4"}).status_code for _ in range(3)]
    assert codes == [200, 200, 429]
    assert c.get("/api/summary", headers={"CF-Connecting-IP": "5.6.7.8"}).status_code == 200
    assert c.get("/static/app.js").status_code == 200        # static files are not limited


def test_ttl_cache():
    from maritime.web.guard import ttl_cache
    calls = []

    @ttl_cache(60, enabled=True)
    def f(x=1):
        calls.append(x)
        return x

    assert f(x=1) == 1 and f(x=1) == 1 and f(x=2) == 2
    assert calls == [1, 2]


def test_ws_limiter():
    from maritime.web.guard import WsLimiter
    w = WsLimiter(total=2, per_ip=1)
    assert w.acquire("a") and not w.acquire("a") and w.acquire("b") and not w.acquire("c")
    w.release("a")
    assert w.acquire("c")


def test_public_mode_endpoints_work_cached(client):
    c = TestClient(webapp.create_app("baltic", public=True))
    for _ in range(2):
        assert len(c.get("/api/vessels").json()) == 2
        assert len(c.get("/api/vessels/273000001/track").json()["points"]) == 3
        assert c.get("/api/recent").status_code == 200
        assert c.get("/api/risk").status_code == 200
    assert c.get("/api/vessels/notanumber/track").status_code == 422
