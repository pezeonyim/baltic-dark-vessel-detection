import numpy as np
import pandas as pd
import pytest

from maritime import risk as R

T0 = pd.Timestamp("2026-10-15 00:00")


def _track(mmsi, lats, lons, sog, step_min=10, t0=T0, name="SHIP", vtype=70, length=150.0, imo="9000001"):
    n = len(lats)
    return pd.DataFrame({"mmsi": mmsi, "ts": [t0 + pd.Timedelta(minutes=step_min * i) for i in range(n)],
                         "lat": lats, "lon": lons, "sog": sog, "cog": 0.0, "nav_status": 0, "vessel_type": vtype,
                         "length": length, "vessel_name": name, "imo": imo, "transceiver_class": "A",
                         "source": "aisstream"})


def _write(tmp_path, df):
    d = tmp_path / f"date={T0.date()}"
    d.mkdir(parents=True, exist_ok=True)
    df.to_parquet(d / "a.parquet")
    return tmp_path


@pytest.fixture
def ais(tmp_path):
    rows = [
        _track(273000001, [60.35] * 19, [28.66] * 19, 0.1, vtype=80),                       # 3 h at Primorsk
        _track(273000002, [57.0, 57.0, 59.0], [18.0, 18.0, 18.0], 12.0),                   # 222 km in 10 min
        _track(273000003, [57.50] * 13, [19.000] * 13, 0.5, imo="9000003"),                # alongside each other
        _track(273000004, [57.50] * 13, [19.002] * 13, 0.5, imo="9000004"),                # ~120 m apart, 2 h
        _track(273000005, list(np.linspace(55, 55.5, 13)), [15.0] * 13, 12.0, imo=None, length=180.0),
    ]
    return _write(tmp_path, pd.concat(rows))


def _land_far():
    import shapely

    from maritime.sar.scene_pipeline import Land
    return Land([shapely.box(10.0, 50.0, 10.5, 50.5)])                                    # far from everything


def test_features(ais):
    f = R.build_features(ais, T0, T0 + pd.Timedelta(days=1), cover=None, land=_land_far(), log=lambda *a: None)
    f = f.set_index("mmsi")
    assert f.loc[273000001, "russian_port_h"] == pytest.approx(3.0, abs=0.2) and f.loc[273000001, "tanker"] == 1
    assert f.loc[273000002, "jumps_n"] == 1 and f.loc[273000005, "jumps_n"] == 0
    assert f.loc[273000003, "meetings"] == 1 and f.loc[273000004, "meetings"] == 1
    assert f.loc[273000003, "loiter_open_sea_h"] >= 2.0
    assert f.loc[273000005, "no_imo_large"] == 1 and f.loc[273000001, "no_imo_large"] == 0
    assert f.loc[273000001, "meetings"] == 0                       # in port: not a meeting


class FakeWL:
    def __len__(self):
        return 1

    def match(self, imo, mmsi):
        return {"listed_category": "shadow_fleet", "listed_category_label": "Shadow fleet"} if mmsi % 2 else None


def test_labels_design_evaluate_explain():
    rng = np.random.default_rng(0)
    n = 300
    f = pd.DataFrame({"mmsi": np.arange(273000000, 273000000 + n), "imo": None, "vessel_type": 70,
                      "silent_moved_n": 0, "silent_moved_max_surprise": 0.0, "russian_port_h": 0.0,
                      "loiter_open_sea_h": 0.0, "meetings": 0, "names_n": 1, "imos_n": 1, "no_imo_large": 0.0,
                      "jumps_n": 0, "radar_links": 0, "tanker": 0.0, "length_m": 150.0})
    f = R.labels(f, FakeWL())
    pos = f["y"] == 1
    f.loc[pos, "meetings"] = rng.integers(1, 4, pos.sum())              # positives meet other ships at sea
    f.loc[pos, "russian_port_h"] = rng.uniform(5, 50, pos.sum())
    f.loc[pos, "tanker"] = 1.0
    X = R.design(f)
    assert list(X.columns) == R.MODEL_FEATURES and "russian_port_h" not in X
    res, oof = R.evaluate(f)
    assert res["positives"] == pos.sum() and res["model_cv"]["roc_auc"] > 0.95 and len(oof) == n
    out = R.fit_and_explain(f)
    top = out.sort_values("risk", ascending=False).iloc[0]
    assert top["y"] == 1 and "ship-to-ship" in top["reasons"] and "Russian" in top["context"]
