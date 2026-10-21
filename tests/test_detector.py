import numpy as np
import pytest

from maritime.sar import detector as D
from maritime.sar.xview3 import OBJECT_DTYPE

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")


def _tile(points, size=512, seed=0):
    """Speckle-like sea (~ -20 dB) with a bright 3x3 target at each point."""
    rng = np.random.default_rng(seed)
    img = np.clip(rng.normal(150, 4, (2, size, size)), 1, 255).astype(np.uint8)
    objs = np.zeros(len(points), OBJECT_DTYPE)
    for i, (r, c) in enumerate(points):
        img[:, r - 1:r + 2, c - 1:c + 2] = 250
        objs[i] = (0, r, c, 1.0, 0.0, 120.0, "HIGH", 5.0)
    return img, objs


def test_render_targets_peak_and_regression():
    _, objs = _tile([(101, 202)])
    heat, cen = D.render_targets(objs)
    assert heat.shape == (128, 128)
    assert heat[25, 50] == 1.0 and heat.max() == 1.0
    np.testing.assert_allclose(cen[0, :4], [25, 50, 0.25, 0.5])
    assert cen[0, 4] == 120.0


def test_augment_keeps_objects_on_bright_pixels():
    img, objs = _tile([(40, 300), (450, 77)])
    rng = np.random.default_rng(1)
    for _ in range(20):
        a_img, a_objs = D.augment(img, objs, rng)
        for o in a_objs:
            assert a_img[0, o["row"], o["col"]] == 250


def test_model_forward_loss_backward_and_decode():
    torch.manual_seed(0)
    model = D.build_model("resnet18", pretrained=False)
    img, objs = _tile([(100, 100)])
    x = torch.as_tensor(np.stack([img, img])).float()
    out = model(x)
    assert out["heat"].shape == (2, 1, 128, 128)
    assert out["offset"].shape == (2, 2, 128, 128)
    heat, cen = D.render_targets(objs)
    loss, parts = D.compute_loss(out, torch.as_tensor(np.stack([heat, heat])), [cen, cen[:0]])
    assert torch.isfinite(loss) and set(parts) == {"heat", "offset", "length", "vessel", "fishing"}
    loss.backward()
    assert model.heat[-1].weight.grad is not None


def test_decode_finds_planted_peak():
    out = {k: torch.zeros(1, c, 128, 128) for k, c in
           [("heat", 1), ("offset", 2), ("length", 1), ("vessel", 1), ("fishing", 1)]}
    out["heat"][:] = -10
    out["heat"][0, 0, 30, 40] = 5
    out["offset"][0, :, 30, 40] = torch.tensor([0.5, 0.25])
    out["length"][0, 0, 30, 40] = float(np.log1p(80))
    (det,) = D.decode(out, threshold=0.3)
    assert len(det) == 1
    np.testing.assert_allclose(det[0, :2], [122, 161])
    assert abs(det[0, 3] - 80) < 1e-3


def test_cfar_detects_target_not_sea():
    img, _ = _tile([(256, 256)])
    det = D.cfar(img)
    assert len(det) >= 1
    d = np.hypot(det[:, 0] - 256, det[:, 1] - 256)
    assert d.min() < 2
    assert (d < 20).mean() > 0.5        # few false alarms on synthetic sea


def _arr(rows, conf=None, vessel=None):
    dt = [("row", "f4"), ("col", "f4"), ("is_vessel", "f4"), ("is_fishing", "f4"),
          ("length_m", "f4"), ("confidence", "U6")]
    a = np.zeros(len(rows), dt)
    a["row"] = [r for r, _ in rows]
    a["col"] = [c for _, c in rows]
    a["is_vessel"] = vessel if vessel is not None else 1
    a["is_fishing"] = 0
    a["length_m"] = 100
    a["confidence"] = conf if conf is not None else "HIGH"
    return a


def test_match_respects_radius():
    pairs = D.match(np.array([[0, 0], [100, 100]]), np.array([[5, 5], [150, 150]]))
    assert pairs == [(0, 0)]


def test_score_counts_and_ignores_low():
    gts = {"s": _arr([(0, 0), (500, 500), (1000, 1000), (2000, 2000)],
                     conf=["HIGH", "HIGH", "LOW", "LOW"])}
    # hit HIGH at (0,0); hit LOW at (1000,1000) -> ignored; one far false alarm.
    preds = {"s": _arr([(3, 4), (1002, 1000), (3000, 3000)])}
    s = D.score(preds, gts)
    assert (s["tp"], s["fp"], s["fn"]) == (1, 1, 1)
    assert s["f1_detect"] == pytest.approx(0.5)
    assert s["f1_vessel"] == 1.0 and s["length_score"] == 1.0


def _npz(path, n_tiles, objs_rows, anc=True):
    imgs = np.full((n_tiles, 2, 512, 512), 150, np.uint8)
    o = np.zeros(len(objs_rows), OBJECT_DTYPE)
    for i, (t, r, c) in enumerate(objs_rows):
        o[i] = (t, r, c, 1.0, 0.0, 50.0, "HIGH", 3.0)
    extra = {"anc_bathymetry": np.full((40, 40), -200, np.float16),
             "anc_owiWindDirection": np.full((40, 40), 90, np.float16)} if anc else {}
    np.savez_compressed(path, images=imgs, origins=np.array([[0, 512 * t] for t in range(n_tiles)]),
                        objects=o, scene_shape=np.array([2048, 2048]), **extra)


def test_load_tiles_keep_filter_and_ancillary(tmp_path):
    _npz(tmp_path / "s1.npz", 3, [(0, 10, 10), (2, 20, 20), (2, 30, 30)])
    _npz(tmp_path / "s2.npz", 2, [], anc=False)
    ts = D.load_tiles([tmp_path / "s1.npz", tmp_path / "s2.npz"],
                      keep=lambda sid, counts: counts > 0 if sid == "s1" else np.ones(len(counts), bool))
    assert len(ts) == 4 and list(ts.scene) == ["s1", "s1", "s2", "s2"]
    assert [len(o) for o in ts.objects] == [1, 2, 0, 0]
    assert set(ts.ancillary["s1"]) == {"bathymetry", "owiWindDirection"} and ts.ancillary["s2"] == {}
    assert ts.scene_shape["s1"] == (2048, 2048)


def test_make_input_channels():
    img = np.full((2, 512, 512), 150, np.uint8)
    img[0, 0, 0] = 0                                     # one no-data pixel
    anc = {"bathymetry": np.full((40, 40), -200.0), "owiWindDirection": np.full((40, 40), 90.0)}
    x = D.make_input(img, ["VV", "VV_local", "bathymetry", "wind_dir_sin", "wind_dir_cos", "wind_speed"],
                     anc, origin=(512, 512), scene_shape=(2048, 2048))
    assert x.shape == (6, 512, 512) and x.dtype == np.float32 and np.isfinite(x).all()
    assert abs(x[1, 5, 5]) < 1e-6                         # local: tile median removed
    assert x[2, 0, 0] == pytest.approx(np.log1p(200) / 4, rel=1e-3)   # depth positive below sea level
    assert x[3, 0, 0] == pytest.approx(1.0) and abs(x[4, 0, 0]) < 1e-6
    assert (x[5] == 0).all()                              # wind speed layer absent -> zeros
    with pytest.raises(ValueError):
        D.make_input(img, ["nope"])


def test_model_takes_any_number_of_channels():
    m = D.build_model(in_ch=5)
    out = m(torch.zeros(1, 5, 256, 256))
    assert out["heat"].shape == (1, 1, 64, 64)


def test_scene_tiles_cover_scene():
    origins = D.scene_tiles(1100, 1300, 512, 64)
    cover = np.zeros((1100, 1300), bool)
    for r, c in origins:
        cover[r:r + 512, c:c + 512] = True
    assert cover.all()
    assert max(r for r, _ in origins) == 1100 - 512 and max(c for _, c in origins) == 1300 - 512
    assert D.scene_tiles(300, 400) == [(0, 0)]


def test_keep_core_and_merge():
    dets = np.array([[10, 250, .9], [250, 10, .9], [250, 250, .9], [505, 250, .9]], float)
    kept = D.keep_core(dets, (448, 448), (2000, 2000))           # interior tile: margins on all sides
    assert kept[:, :2].tolist() == [[250, 250]]
    kept = D.keep_core(dets, (0, 0), (2000, 2000))               # top-left tile: top/left edges are scene edges
    assert len(kept) == 3
    m = D.merge_close(np.array([[100, 100, .5], [104, 103, .9], [300, 300, .4]], float), 10)
    assert m[:, :3].tolist() == [[104, 103, .9], [300, 300, .4]]


def test_xview3_metrics_shore_and_aggregate():
    def rec(rows, conf="HIGH"):
        a = np.zeros(len(rows), [("row", "f4"), ("col", "f4"), ("is_vessel", "f4"), ("is_fishing", "f4"),
                                 ("length_m", "f4"), ("confidence", "U6")])
        a["row"] = [r for r, _ in rows]
        a["col"] = [c for _, c in rows]
        a["is_vessel"] = 1
        a["is_fishing"] = 0
        a["length_m"] = 50
        a["confidence"] = conf
        return a
    # shoreline along row 0, one vertex per pixel like the real xView3 contours
    shore = {"s": [np.column_stack([np.zeros(5001), np.arange(5001.0)])]}
    gts = {"s": rec([(50, 100), (50, 900), (3000, 500)])}         # two near shore (<200 px), one far
    preds = {"s": rec([(52, 102), (3001, 500), (60, 2000)])}      # near hit, far hit, near false alarm
    m = D.xview3_metrics(preds, gts, shore)
    assert (m["tp"], m["fp"], m["fn"]) == (2, 1, 1)
    assert (m["shore_tp"], m["shore_fp"], m["shore_fn"]) == (1, 1, 1) and m["f1_shore"] == pytest.approx(0.5)
    f1d = m["f1_detect"]
    assert m["aggregate"] == pytest.approx(f1d * (1 + 0.5 + 1.0 + 0.0 + 1.0) / 5)
    assert np.isinf(D.shore_distance_px([[1, 1]], None)).all()


def test_predict_scene_shapes_and_coordinates():
    torch.manual_seed(0)
    models = [D.build_model(in_ch=2).eval(), D.build_model(in_ch=2).eval()]
    img = np.full((2, 700, 900), 150, np.uint8)
    dets, origins = D.predict_scene(models, img, decode_threshold=0.0, max_det=5)
    assert len(dets) == 3 and len(origins) == 4                 # 2 models + ensemble; 2x2 tiles
    for d in dets:
        assert d.shape[1] == 6 and (d[:, 0] < 700).all() and (d[:, 1] < 900).all()


def test_unmatched_and_hard_negative_tiles():
    dets = np.array([[100, 100, .9, 0, 1, 0], [600, 600, .8, 0, 1, 0], [605, 610, .7, 0, 1, 0],
                     [900, 100, .2, 0, 1, 0]], float)
    labels_rc = np.array([[102, 98]])                          # first detection is a real ship
    fa = D.unmatched_detections(dets, labels_rc, min_score=0.3)
    assert fa[:, :2].tolist() == [[600, 600], [605, 610]]      # low score dropped, matched dropped
    img = np.full((2, 1200, 1300), 150, np.uint8)
    lab = np.zeros(1, [("row", "f4"), ("col", "f4"), ("is_vessel", "f4"), ("is_fishing", "f4"),
                       ("length_m", "f4"), ("confidence", "U6"), ("shore_km", "f4")])
    lab[0] = (650, 640, 1, 0, 40, "HIGH", 3)
    ims, origins, objs = D.hard_negative_tiles(img, fa, lab, seed=1)
    assert ims.shape == (1, 2, 512, 512)                       # second false alarm already covered
    r0, c0 = origins[0]
    assert r0 + 64 <= 600 < r0 + 448 and c0 + 64 <= 600 < c0 + 448
    assert len(objs) == 1 and (objs[0]["row"], objs[0]["col"]) == (650 - r0, 640 - c0)
