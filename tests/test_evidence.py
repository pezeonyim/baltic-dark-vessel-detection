import pandas as pd

from maritime.sar import evidence as E

T = pd.Timestamp("2026-10-12 17:00")


def _ais(mmsi, times, lats, lons, length=150.0, cls="A", sog=12.0, cog=0.0, name="SHIP"):
    return pd.DataFrame({"mmsi": mmsi, "ts": pd.to_datetime(times), "lat": lats, "lon": lons, "sog": sog,
                         "cog": cog, "length": length, "transceiver_class": cls, "vessel_name": name})


def _cand(lat=57.5, lon=10.0):
    return pd.Series({"time": T, "lat": lat, "lon": lon})


def test_ship_whose_track_passes_is_linked():
    # silent 16:00-18:00, moving north 0.2 deg (~22 km) -> at 17:00 it is at 57.5
    a = _ais(1, ["2026-10-12 16:00", "2026-10-12 18:00"], [57.4, 57.6], [10.0, 10.0])
    got = E.silent_links(_cand(), a)
    assert list(got["mmsi"]) == [1]
    assert got["track_miss_km"].iloc[0] == 0.0 and got["gap_h"].iloc[0] == 2.0 and bool(got["reappeared"].iloc[0])


def test_reachable_but_off_track_is_not_linked():
    # same gap, but the track runs 20 km east of the candidate
    a = _ais(1, ["2026-10-12 16:00", "2026-10-12 18:00"], [57.4, 57.6], [10.33, 10.33])
    assert E.silent_links(_cand(), a).empty


def test_still_reporting_small_or_class_b_are_skipped():
    reporting = _ais(1, ["2026-10-12 16:40", "2026-10-12 16:55", "2026-10-12 17:10"], [57.5] * 3, [10.0] * 3)
    small = _ais(2, ["2026-10-12 16:00", "2026-10-12 18:00"], [57.4, 57.6], [10.0, 10.0], length=20.0)
    class_b = _ais(3, ["2026-10-12 16:00", "2026-10-12 18:00"], [57.4, 57.6], [10.0, 10.0], cls="B")
    assert E.silent_links(_cand(), pd.concat([reporting, small, class_b])).empty


def test_dead_reckoning_when_not_reappeared():
    # last report 15:00 at 57.3, 12 kn north: 2 h * 12 kn = 44 km = 0.4 deg -> 57.7, candidate at 57.7
    a = _ais(1, ["2026-10-12 15:00"], [57.3], [10.0])
    got = E.silent_links(_cand(lat=57.7), a)
    assert len(got) == 1 and not bool(got["reappeared"].iloc[0]) and got["track_miss_km"].iloc[0] < 1.0


def test_assign_one_ship_per_image_and_verdicts():
    links = pd.DataFrame({"cand": [0, 1, 1], "product": "P", "mmsi": [7, 7, 8], "track_miss_km": [0.5, 0.2, 3.0],
                          "fit": [0.1, 0.05, 0.6], "gap_h": [2.0, 9.0, 1.0]})
    best = E.assign_links(links).set_index("cand")
    # ship 7 fits candidate 1 best; candidate 0 cannot also be ship 7, and candidate 1 does not also get ship 8
    assert set(best.index) == {1} and best.loc[1, "mmsi"] == 7
    assert E.verdict(best.loc[1]) == "went_silent"
    assert E.verdict(pd.Series({"track_miss_km": 0.5, "gap_h": 2.0})) == "explained"
    assert E.verdict(pd.Series({"track_miss_km": 3.0, "gap_h": 2.0})) == "weak_link"
    assert E.verdict(None) == "unexplained"


def test_many_ships_quiet_at_once_is_a_feed_outage():
    t = pd.Timestamp("2026-10-14 04:41")
    since = pd.Timestamp("2026-10-13 22:20")
    others = pd.concat([_ais(10 + k, [since + pd.Timedelta(minutes=k)], [59.9], [26.0 + 0.01 * k]) for k in range(5)])
    assert E.quiet_together(others, t, 59.9, 26.0, since, mmsi=1) == 5
    far = others.assign(lat=58.0)
    assert E.quiet_together(far, t, 59.9, 26.0, since, mmsi=1) == 0
    assert E.verdict(pd.Series({"track_miss_km": 0.5, "gap_h": 6.4, "quiet_together": 5})) == "feed_outage"
    assert E.verdict(pd.Series({"track_miss_km": 0.5, "gap_h": 6.4, "quiet_together": 1})) == "went_silent"


def test_several_tight_fits_is_crowded():
    b = pd.Series({"track_miss_km": 0.4, "gap_h": 11.0, "quiet_together": 0, "n_tight": 4})
    assert E.verdict(b) == "crowded"
    assert E.verdict(b.drop("n_tight")) == "went_silent"


def test_low_reception_and_ratio(tmp_path):
    b = pd.Series({"track_miss_km": 0.2, "gap_h": 9.0, "quiet_together": 0, "n_tight": 1, "reception": 0.39})
    assert E.verdict(b) == "low_reception"
    t = pd.Timestamp("2026-10-14 16:00")
    for day, n in ((t, 2), (t - pd.Timedelta(days=1), 4)):
        d = tmp_path / f"date={day.date()}"
        d.mkdir()
        _ais(list(range(n)), [day] * n, [59.84] * n, [25.7] * n).to_parquet(d / "a.parquet")
    assert E.reception_ratio(tmp_path, t, 59.84, 25.7) == 0.5


def _flat_model(p=0.05):
    from maritime.features.coverage_map import LEVELS
    return {"cells": {r: pd.DataFrame(columns=["p"]) for r in LEVELS}, "p0": p, "xi": 1.3, "sigma": 42.0}


def test_gap_surprise_and_usual_gap():
    patchy, good = _flat_model(0.07), _flat_model(0.0005)
    s6 = E.gap_surprise(patchy, [(57.6, 19.7)], 6.0)
    assert 1.0 < s6 < E.SURPRISE_MIN < E.gap_surprise(good, [(57.6, 19.7)], 6.0)   # patchy: usual; good: not
    assert E.gap_surprise(patchy, [(57.6, 19.7)], 12.0) > s6               # longer silences are rarer
    assert E.gap_surprise(good, [(57.6, 19.7)], 6.0) > s6                  # and rarer where coverage is good
    b = pd.Series({"track_miss_km": 0.5, "gap_h": 8.0, "quiet_together": 0, "reception": 1.0, "n_tight": 1})
    assert E.verdict(pd.concat([b, pd.Series({"surprise": 2.5})])) == "usual_gap"
    assert E.verdict(pd.concat([b, pd.Series({"surprise": 5.0})])) == "went_silent"
    assert E.verdict(b) == "went_silent"                                     # no coverage model: not checked


def test_length_mismatch_is_not_linked():
    a = _ais(1, ["2026-10-12 16:00", "2026-10-12 18:00"], [57.4, 57.6], [10.0, 10.0], length=366.0)
    c = _cand()
    assert len(E.silent_links(c, a)) == 1                                    # no radar length: not checked
    assert E.silent_links(pd.concat([c, pd.Series({"length_est_m": 113.0})]), a).empty
    assert len(E.silent_links(pd.concat([c, pd.Series({"length_est_m": 300.0})]), a)) == 1
