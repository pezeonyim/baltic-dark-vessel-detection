import numpy as np
import pandas as pd
import pytest

from maritime.fusion import ais_at_time, dark_candidates, match_to_ais, to_local_m
from maritime.sar import fusion_eval


def test_to_local_m_one_degree_north():
    x, y = to_local_m(55.0, 18.0, 54.0, 18.0)
    assert abs(x) < 1e-6 and abs(y - 111_195) < 50


def test_ais_interpolates_between_reports():
    t0 = pd.Timestamp("2026-01-01T00:00:00")
    ais = pd.DataFrame({"mmsi": [1, 1], "ts": [t0, t0 + pd.Timedelta(minutes=10)],
                        "x": [0.0, 1000.0], "y": [0.0, 0.0], "sog": [5.0, 5.0], "cog": [90.0, 90.0]})
    p = ais_at_time(ais, t0 + pd.Timedelta(minutes=4))
    assert len(p) == 1 and abs(p.x.iat[0] - 400) < 1e-6 and p.dt_s.iat[0] == 240


def test_ais_dead_reckons_and_drops_stale():
    t0 = pd.Timestamp("2026-01-01T00:00:00")
    ais = pd.DataFrame({"mmsi": [1, 2], "ts": [t0, t0 - pd.Timedelta(hours=2)],
                        "x": [0.0, 0.0], "y": [0.0, 0.0], "sog": [10.0, 10.0], "cog": [0.0, 0.0]})
    p = ais_at_time(ais, t0 + pd.Timedelta(seconds=100), max_age_s=1800)
    assert list(p.mmsi) == [1]                                   # mmsi 2 is 2 h old: dropped
    assert abs(p.y.iat[0] - 10 * 0.514444 * 100) < 1e-3          # 10 kn north for 100 s
    assert p.unc_m.iat[0] > 50


def test_match_is_one_to_one_and_gated():
    det = np.array([[0, 0], [50, 0], [5000, 0]])
    ais = np.array([[10, 0]])
    m = match_to_ais(det, ais, base_gate_m=200)
    assert list(m) == [0, -1, -1]
    m = match_to_ais(det, ais, base_gate_m=10, ais_unc_m=np.array([0.0]))
    assert list(m) == [0, -1, -1]
    assert list(match_to_ais(det, np.zeros((0, 2)))) == [-1, -1, -1]


def test_dark_candidates_status():
    dets = pd.DataFrame({"x": [0, 1000, 2000, 3000], "y": [0, 0, 0, 0],
                         "score": [0.9, 0.9, 0.9, 0.2], "p_vessel": [0.9, 0.9, 0.1, 0.9]})
    ais = pd.DataFrame({"x": [20.0], "y": [0.0], "unc_m": [0.0]})
    d = dark_candidates(dets, ais, score_min=0.45)
    assert list(d.status) == ["ais", "dark", "non_vessel"]


def _labels():
    rows = [  # row, col, source, is_vessel, confidence
        (100, 100, "ais/manual", True, "HIGH"),
        (100, 400, "manual", True, "MEDIUM"),     # dark vessel, detected
        (100, 700, "manual", True, "HIGH"),       # dark vessel, missed
        (100, 1000, "manual", False, "HIGH"),     # unlit platform
        (100, 1300, "manual", True, "LOW"),       # ignored
    ]
    return pd.DataFrame([{"detect_scene_row": r, "detect_scene_column": c, "source": s, "is_vessel": v,
                          "confidence": q, "distance_from_shore_km": 9999.99, "scene_id": "s"}
                         for r, c, s, v, q in rows])


def test_fusion_eval_end_to_end():
    dets = pd.DataFrame({"scene_id": "s", "row": [101, 100, 100, 100, 100, 500],
                         "col": [101, 402, 1000, 1301, 1600, 500], "score": 0.9,
                         "p_vessel": [0.9, 0.9, 0.8, 0.9, 0.9, 0.9]})
    s, per, c, lab = fusion_eval.run(dets, _labels(), score_min=0.45)
    assert s["ais_matched"] == 1
    assert s["tp"] == 1 and s["fp_infra"] == 1 and s["ignored"] == 1 and s["fp_clutter"] == 2
    assert s["dark_vessels"] == 2 and s["recall"] == 0.5 and s["miss_not_detected"] == 1
    assert abs(s["precision"] - 1 / 4) < 1e-9


def test_doppler_shift_and_stretched_gate():
    from maritime.fusion import doppler_shift_m
    along = np.array([0.0, 1.0])                       # satellite flies north
    # 10 kn due east (fully across track), 40 deg incidence, R/V = 110 s
    s = doppler_shift_m([10.0, 10.0], [90.0, 0.0], along, [40.0, 40.0], 110.0)
    assert s[0] == pytest.approx(110 * 10 * 0.514444 * np.sin(np.radians(40)), rel=1e-6)
    assert s[1] == pytest.approx(0.0, abs=1e-9)        # moving along the track: no shift
    det = np.array([[0.0, 450.0], [450.0, 0.0]])       # 450 m along / across track from the AIS position
    ais = np.array([[0.0, 0.0]])
    assert list(match_to_ais(det, ais, 200)) == [-1, -1]
    m = match_to_ais(det[:1], ais, 200, along=along, along_extra_m=np.array([400.0]))
    assert list(m) == [0]
    m = match_to_ais(det[1:], ais, 200, along=along, along_extra_m=np.array([400.0]))
    assert list(m) == [-1]                             # across-track stays strict


def test_ais_at_time_keeps_speed_and_course():
    t0 = pd.Timestamp("2026-01-01T00:00:00")
    ais = pd.DataFrame({"mmsi": [1, 1], "ts": [t0, t0 + pd.Timedelta(minutes=10)], "x": [0.0, 1000.0],
                        "y": [0.0, 0.0], "sog": [5.0, 6.0], "cog": [90.0, 91.0]})
    p = ais_at_time(ais, t0 + pd.Timedelta(minutes=8))
    assert p.sog.iat[0] == 6.0 and p.cog.iat[0] == 91.0
