"""Gap and outage detection on tiny synthetic tracks."""

from datetime import datetime, timedelta

import duckdb
import pandas as pd

from maritime.features.coverage import outages
from maritime.features.gaps import detect_gaps

T0 = datetime(2023, 1, 1)


def track(mmsi, minutes, sog=10.0, nav=0, lat0=28.0):
    return [dict(mmsi=mmsi, ts=T0 + timedelta(minutes=m), lat=lat0 + i * 0.001, lon=-90.0,
                 sog=sog, nav_status=nav, vessel_type=80, transceiver_class="A")
            for i, m in enumerate(minutes)]


def run(rows, outage_df=None, k=30):
    con = duckdb.connect()
    con.register("pings", pd.DataFrame(rows))
    return detect_gaps(con, "pings", "marinecadastre", outage_df, k=k)


def test_underway_gap_detected_and_short_silence_ignored():
    rows = track(1, [0, 1, 2, 40, 41]) + track(2, [0, 1, 20, 21])  # 38 min vs 19 min
    g = run(rows)
    assert list(g["mmsi"]) == [1]
    assert round(g.iloc[0]["silence_min"]) == 38


def test_moored_vessel_needs_longer_silence():
    # moored: expected 180 s, so threshold is 90 min; a 60 min silence is not a gap
    assert run(track(3, [0, 3, 63, 66], sog=0.0, nav=5)).empty
    assert len(run(track(3, [0, 3, 120, 123], sog=0.0, nav=5))) == 1


def test_gap_explained_by_outage_is_dropped_but_others_kept():
    out = pd.DataFrame({"start": [T0 + timedelta(minutes=5)], "end": [T0 + timedelta(minutes=45)]})
    rows = track(1, [0, 1, 50, 51])           # 49 min silence, 40 min of it in the outage
    rows += track(2, [100, 101, 150, 151])    # 49 min silence, no outage overlap
    g = run(rows, out)
    assert list(g["mmsi"]) == [2]             # vessel 2 must survive (NULL-join regression)
    assert g.iloc[0]["outage_min"] == 0


def test_outages_merge_consecutive_bins():
    bins = pd.date_range(T0, periods=6, freq="10min")
    vol = pd.DataFrame({"bin_start": bins, "bin_end": bins + pd.Timedelta("10min"),
                        "ratio": [1.0, 0.1, 0.05, 1.0, 0.2, 1.0]})
    o = outages(vol)
    assert list(o["minutes"]) == [20, 10]
