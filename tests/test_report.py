import numpy as np
import pandas as pd

from maritime.sar import report as RP
from maritime.sar.scene_pipeline import cells_of
from maritime.sar.sentinel1 import Geo


def test_corners_wkt():
    lines, pixels = np.meshgrid([0, 999], [0, 999], indexing="ij")
    g = Geo(lines.ravel().astype(float), pixels.ravel().astype(float), 59 + lines.ravel() * 1e-4,
            20 + pixels.ravel() * 2e-4, np.array(["2026-10-12T05:00"] * 4, dtype="datetime64[us]"))
    w = RP.corners_wkt(g, 1000, 1000)
    assert w.startswith("POLYGON((20.00000 59.00000,20.19980 59.00000") and w.count(",") == 4


def test_scene_recall_and_summary():
    fp = "POLYGON((19 58,21 58,21 60,19 60,19 58))"
    pos = pd.DataFrame({"mmsi": [1, 2, 3, 4, 5], "lat": [59.0, 59.1, 59.2, 61.0, 59.3],
                        "lon": [20.0, 20.1, 20.2, 20.0, 20.3], "dt_s": [30, 30, 30, 30, 2000],
                        "sog": [10, 0, 12, 10, 10], "length": [150.0, 20.0, 80.0, 100.0, 90.0],
                        "transceiver_class": ["A", "B", "A", "A", "A"]})
    cover = set(cells_of(pos["lat"], pos["lon"]))
    det = pd.DataFrame({"status": ["ais", "dark", "ais", "small"], "mmsi": [1, -1, 3, -1],
                        "length_m": [120.0, 60.0, 70.0, 15.0], "tier": ["", "high", "", ""]})
    r = RP.scene_recall(det, pos, fp, cover, cover)
    # mmsi 4 outside the image, 5 stale (2000 s) -> 3 expected ships, 2 found
    assert sorted(r.mmsi) == [1, 2, 3] and list(r.sort_values("mmsi").found) == [True, False, True]
    lengths = pd.DataFrame({"est_length": [120.0, 70.0, 25.0], "ais_length": [150.0, 80.0, 50.0]})
    s = RP.summarise(r.assign(vv_db=-16.0), det, lengths)
    assert s["ais_ships_expected"] == 3 and s["recall_all"] == round(2 / 3, 3)
    assert s["recall_by_length"][">=120 m"] == {"n": 1, "recall": 1.0}
    assert s["no_ais_share_ceiling"] == round(1 / 3, 3)          # 1 dark of 3 confident vessels >= 30 m
    assert s["ships_>=45m_estimated_<30m"] == {"n": 3, "share": round(1 / 3, 3)}
    assert s["length_ratio_median"] == 0.8               # ratios 0.8, 0.875, 0.5


def test_fill_footprints_from_search_and_offline():
    man = pd.DataFrame({"product": ["A", "B"], "footprint": ["POLYGON((0 0,1 0,1 1,0 0))", None]})
    found = pd.DataFrame({"name": ["B.SAFE"], "footprint": ["POLYGON((2 2,3 2,3 3,2 2))"]})
    out = RP.fill_footprints(man, lambda: found, log=lambda *a: None)
    assert out.footprint[1].startswith("POLYGON((2 2")

    def offline():
        raise OSError("no network")
    out = RP.fill_footprints(man, offline, log=lambda *a: None)
    assert out.footprint.isna().sum() == 1
