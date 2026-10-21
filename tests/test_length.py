import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from maritime.sar import length as L  # noqa: E402
from maritime.sar import scene_pipeline as P  # noqa: E402


def _model():
    from maritime.sar import detector as D
    return D.build_model("resnet18", pretrained=False, in_ch=2).eval()


def test_features_train_predict_shapes_and_learning():
    m = _model()
    rng = np.random.default_rng(0)
    chips = rng.integers(60, 120, (24, 2, 96, 96), dtype=np.uint8)
    F = L.chip_features(m, chips)
    assert tuple(F.shape) == (24, 64, 3, 3)
    y = np.where(np.arange(24) % 2 == 0, 40.0, 200.0)            # two size classes
    F = F + torch.from_numpy(np.where(y == 40.0, 0.0, 1.0)).float()[:, None, None, None]   # separable features
    head = L.train_head(m, F, y, epochs=200)
    est = L.predict(head, F)
    assert np.median(np.abs(est - y) / y) < 0.25


def test_metrics_counts_cutoff():
    out = L.metrics([20, 60, 100, 300, 40], [25, 50, 110, 280, 120])
    assert out["n"] == 5 and out["n_boats_lt30m"] == 1
    assert out[f"ships_ge45m_below_{P.DARK_MIN_LENGTH_M:g}m_pct"] == 25.0      # 40 m estimate for a 120 m ship
    assert out[f"boats_lt30m_below_{P.DARK_MIN_LENGTH_M:g}m_pct"] == 100.0


def test_refine_status_prefers_length_est():
    d = pd.DataFrame({"status": ["dark", "dark", "dark"], "score": 0.7, "lat": 57.0, "lon": 19.0,
                      "length_m": [30.0, 10.0, 10.0], "length_est_m": [35.0, 120.0, np.nan]})
    r = P.refine_status(d)
    # est 35 m -> small despite a large model length; est 120 m -> dark despite a small one; no est -> fallback
    assert list(r["status"]) == ["small", "dark", "small"]
    assert np.isnan(r["length_rough_m"].iloc[0]) and r["length_rough_m"].iloc[1] == 120.0


def test_cut_chips_matches_saved(tmp_path):
    img = np.zeros((2, 300, 400), np.uint8)
    img[0, 100, 200] = 250
    c = P.cut_chips(img, [100], [200], half=16)
    P.save_chips(img, [100], [200], tmp_path / "x.npz", half=16, chips=c)
    assert np.array_equal(np.load(tmp_path / "x.npz")["chips"], c) and c[0, 0, 16, 16] == 250
