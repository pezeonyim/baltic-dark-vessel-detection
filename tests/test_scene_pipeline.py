import numpy as np
import pandas as pd
import pytest

from maritime.sar import scene_pipeline as P
from maritime.sar.sentinel1 import Geo, GRDScene


def _geo():
    # 2,000 x 2,000 px scene; lat grows with row (1e-4 deg/px ~ 11 m), lon with col (2e-4 deg/px ~ 11 m at 59N)
    lines, pixels = np.meshgrid([0, 1000, 1999], [0, 1000, 1999], indexing="ij")
    lines, pixels = lines.ravel().astype(float), pixels.ravel().astype(float)
    t = np.datetime64("2026-10-04T16:00:00", "us") + (lines * 10_000).astype("timedelta64[us]")
    return Geo(lines, pixels, 59.0 + lines * 1e-4, 24.0 + pixels * 2e-4, t)


def _scene():
    img = np.full((2, 2000, 2000), 60, np.uint8)
    return GRDScene("S1C_TEST", img, _geo(), np.datetime64("2026-10-04T16:00:00", "us"),
                    np.datetime64("2026-10-04T16:00:20", "us"), "S1C", "Ascending")


def _land():
    import shapely
    # land = everything with lon > 24.3 (cols > 1500)
    return P.Land([shapely.box(24.3, 58.0, 25.0, 60.0)])


def test_land_mask_and_points():
    land = _land()
    m = land.mask(_geo(), 2000, 2000, step=16)
    assert m.shape == (126, 126)
    assert m[:, 100:].mean() > 0.95 and m[:, :90].mean() < 0.01        # col 1500 / 16 ~ 94
    assert list(land.on_land([59.05, 59.05], [24.1, 24.35])) == [False, True]
    keep = P.tile_filter(m)
    assert keep((0, 0)) and not keep((0, 1536))


def test_load_ais_filters_time_and_box(tmp_path):
    d = tmp_path / "date=2026-10-04"
    d.mkdir()
    pd.DataFrame({"mmsi": [1, 2, 3], "ts": pd.to_datetime(["2026-10-04 15:50", "2026-10-04 15:00",
                                                           "2026-10-04 16:05"]),
                  "lat": [59.1, 59.1, 70.0], "lon": [24.1, 24.1, 24.1], "sog": 1.0, "cog": 0.0,
                  "vessel_name": "x", "length": np.nan, "transceiver_class": "A"}).to_parquet(d / "a.parquet")
    a = P.load_ais(tmp_path, "2026-10-04 15:30", "2026-10-04 16:30", (23.9, 58.9, 24.5, 59.3))
    assert list(a.mmsi) == [1]


def test_process_scene_statuses(monkeypatch):
    from maritime.sar import detector as D
    # detections (row, col, score, length, p_vessel, p_fishing)
    dets = np.array([[1000, 500, 0.9, 50, 0.9, 0.1],    # AIS ship here -> ais
                     [1000, 800, 0.7, 30, 0.9, 0.1],    # no AIS -> dark, high
                     [1200, 800, 0.5, 20, 0.9, 0.1],    # no AIS -> dark, possible
                     [1000, 1000, 0.8, 20, 0.1, 0.0],   # not a vessel
                     [1000, 1700, 0.9, 40, 0.9, 0.0],   # on land -> dropped
                     [500, 500, 0.3, 20, 0.9, 0.0]])    # below threshold
    seen = {}

    def fake(models, img, batch, keep_tile, device):
        seen["keep"] = keep_tile
        return [dets], [(0, 0)]
    monkeypatch.setattr(D, "predict_scene", fake)
    sc = _scene()
    lat, lon = sc.geo.to_latlon([1000], [500])
    ais = pd.DataFrame({"mmsi": [111], "ts": [pd.Timestamp("2026-10-04 16:00:05")], "lat": lat, "lon": lon,
                        "sog": [0.0], "cog": [0.0]})
    out = P.process_scene(sc, [], _land(), ais, log=lambda *a: None)
    assert seen["keep"] is not None and not seen["keep"]((0, 1536))
    got = {(int(r.row), int(r.col)): (r.status, r.tier, r.mmsi) for r in out.itertuples()}
    assert got == {(1000, 500): ("ais", "", 111), (1000, 800): ("dark", "high", -1),
                   (1200, 800): ("dark", "possible", -1), (1000, 1000): ("non_vessel", "", -1)}
    assert out["time"].iloc[0] == pd.Timestamp("2026-10-04 16:00:10")        # row 1000 -> +10 s
    assert out["lat"].iloc[0] == pytest.approx(59.1) and (out["product"] == "S1C_TEST").all()


def test_prepare_land_clips_zip(tmp_path):
    import zipfile

    import pyogrio
    import shapely
    g = np.array([shapely.box(20, 55, 22, 57), shapely.box(40, 10, 41, 11)])
    shp_dir = tmp_path / "land-polygons-split-4326"
    shp_dir.mkdir()
    pyogrio.raw.write(str(shp_dir / "land_polygons.shp"), shapely.to_wkb(g), field_data=[], fields=[],
                      crs="EPSG:4326", geometry_type="Polygon", driver="ESRI Shapefile")
    zp = tmp_path / "land.zip"
    with zipfile.ZipFile(zp, "w") as z:
        for f in shp_dir.iterdir():
            z.write(f, f"land-polygons-split-4326/{f.name}")
    out = tmp_path / "baltic_land.parquet"
    assert P.prepare_land(zp, (8.0, 53.0, 30.5, 66.0), out) == 1
    land = P.Land.load(out)
    assert list(land.on_land([56.0, 56.0], [21.0, 25.0])) == [True, False]


def test_processed_products_reads_manifest(tmp_path, monkeypatch):
    from maritime.sar import run as R
    monkeypatch.setattr(R, "PROCESSED", tmp_path)
    assert R.processed_products("baltic") == set()
    d = tmp_path / "baltic" / "sar"
    d.mkdir(parents=True)
    (d / "manifest.jsonl").write_text('{"product": "A"}\n{"product": "B"}\n')
    assert R.processed_products("baltic") == {"A", "B"}


def test_cli_has_sar():
    from maritime.cli import main
    with pytest.raises(SystemExit):
        main(["sar", "--help"])


def test_scene_medians_db_inverts_quantise():
    from maritime.sar import run as R
    from maritime.sar.xview3 import quantise
    img = np.stack([quantise(np.full((64, 64), -20.0)), quantise(np.full((64, 64), -28.0))])
    m = R.scene_medians_db(img)
    assert m["vv_db"] == pytest.approx(-20.0, abs=0.2) and m["vh_db"] == pytest.approx(-28.0, abs=0.2)


def test_sea_area_and_selection():
    import shapely

    from maritime.config import BBox
    from maritime.sar import run as R
    land = P.Land([shapely.box(10.0, 55.0, 11.0, 56.0)])
    sea, total = land.sea_km2("POLYGON((10 55,12 55,12 56,10 56,10 55))", (8.0, 53.0, 30.5, 66.0))
    assert total == pytest.approx(2 * 111.32 ** 2 * np.cos(np.radians(55.5)), rel=1e-6)
    assert sea == pytest.approx(total / 2, rel=1e-6)

    class Reg:
        bbox = BBox(min_lon=8.0, min_lat=53.0, max_lon=30.5, max_lat=66.0)
    df = pd.DataFrame({"footprint": ["POLYGON((10 55,11 55,11 56,10 56,10 55))",
                                     "POLYGON((10 55,12 55,12 56,10 56,10 55))", None]})
    out = R.add_sea_area(df, land, Reg)
    assert out.sea_km2[0] == 0 and out.sea_km2[1] > 6000 and np.isnan(out.sea_km2[2])


def test_coverage_cells_and_no_coverage_status(tmp_path, monkeypatch):
    from maritime.sar import detector as D
    d = tmp_path / "date=2026-10-03"
    d.mkdir()
    # three ships heard near (59.1, 24.1) during the week; nothing near (59.15, 24.3)
    pd.DataFrame({"mmsi": [1, 2, 3], "ts": pd.to_datetime(["2026-10-03 12:00"] * 3), "lat": [59.12, 59.13, 59.11],
                  "lon": [24.05, 24.06, 24.07], "sog": 1.0, "cog": 0.0}).to_parquet(d / "a.parquet")
    cover = P.coverage_cells(tmp_path, "2026-10-04 16:00")
    assert P.cells_of([59.12], [24.05])[0] in cover and P.cells_of([59.15], [24.5])[0] not in cover
    dets = np.array([[1100, 300, 0.9, 30, 0.9, 0.1],     # 59.11, 24.06: covered cell, live receiver -> dark
                     [1500, 1300, 0.9, 30, 0.9, 0.1]])  # 59.15, 24.26: not covered -> no_coverage
    monkeypatch.setattr(D, "predict_scene", lambda *a, **k: ([dets], [(0, 0)]))
    live = pd.DataFrame({"mmsi": [9], "ts": [pd.Timestamp("2026-10-04 16:00:05")], "lat": [59.0],
                         "lon": [24.0], "sog": [12.0], "cog": [90.0]})
    out = P.process_scene(_scene(), [], None, live, cover=cover, log=lambda *a: None)
    assert list(out.status) == ["dark", "no_coverage"] and list(out.tier) == ["high", ""]


def test_save_chips_pads_edges_and_keeps_order(tmp_path):
    img = np.zeros((2, 300, 400), np.uint8)
    img[0, 100, 200] = 250                       # a bright ship
    img[1, 100, 200] = 180
    n = P.save_chips(img, [100, 5], [200, 395], tmp_path / "x.npz", half=16)
    z = np.load(tmp_path / "x.npz")
    assert n == 2 and z["chips"].shape == (2, 2, 32, 32) and z["chips"].dtype == np.uint8
    assert z["chips"][0, 0, 16, 16] == 250 and z["chips"][0, 1, 16, 16] == 180   # centred on the detection
    assert z["chips"][1, 0, :11, :].max() == 0                                     # padded above the top edge
    assert list(z["rows"]) == [100, 5] and list(z["cols"]) == [200, 395]


def test_chip_gallery_renders(tmp_path, monkeypatch):
    from maritime.config import BBox
    from maritime.sar import run as R
    monkeypatch.setattr(R, "PROCESSED", tmp_path)
    monkeypatch.setattr(R, "DATA", tmp_path)
    day = pd.Timestamp.now("UTC").tz_localize(None).date()
    det_dir = tmp_path / "baltic" / "sar" / "detections" / f"date={day}"
    det_dir.mkdir(parents=True)
    t = pd.Timestamp.now("UTC").tz_localize(None) - pd.Timedelta(hours=1)
    pd.DataFrame({"time": [t] * 3, "score": [0.8, 0.5, 0.9], "length_m": 40.0, "status": ["dark", "dark", "ais"],
                  "tier": ["high", "possible", ""], "mmsi": [-1, -1, 123], "chip": [0, 1, 2], "product": "S",
                  "lat": [58.0, 58.1, 58.2], "lon": [20.0, 20.0, 20.0]}).to_parquet(
        det_dir / "S.parquet")
    img = np.full((2, 200, 200), 60, np.uint8)
    P.save_chips(img, [50, 100, 150], [50, 100, 150], tmp_path / "baltic" / "sar" / "chips" / f"date={day}" / "S.npz")

    class Reg:
        name = "baltic"
        bbox = BBox(min_lon=8.0, min_lat=53.0, max_lon=30.5, max_lat=66.0)
    out = R.chip_gallery(Reg, days=1, out=tmp_path / "g.png", log=lambda *a: None)
    assert out is not None and out.exists() and out.stat().st_size > 1000


def test_refine_status_rules_and_static():
    t0 = pd.Timestamp("2026-10-12 05:00")
    det = pd.DataFrame({
        "product": ["A", "A", "A", "A", "B", "C"],
        "time": [t0, t0, t0, t0, t0 + pd.Timedelta(hours=12), t0 + pd.Timedelta(hours=1)],
        "lat": [58.0, 58.5, 64.0, 57.0, 58.0003, 57.0003],
        "lon": [20.0, 20.5, 10.0, 19.0, 20.0, 19.0],
        "length_m": [80.0, 12.0, 90.0, 60.0, 80.0, 60.0],      # model scale: < 16 = small craft
        "score": [0.9, 0.7, 0.8, 0.5, 0.9, 0.5],
        "status": ["dark", "dark", "dark", "dark", "no_coverage", "dark"],
    })
    st = P.static_mask(det)
    # rows 0 and 4: same spot (~35 m), different images 12 h apart -> static; rows 3 and 5 only 1 h apart -> not
    assert list(st) == [True, False, False, False, True, False]
    out = P.refine_status(det, st)
    assert list(out.status) == ["static", "small", "outside_area", "dark", "static", "dark"]
    assert list(out.tier) == ["", "", "", "possible", "", "possible"]


def test_length_rough_m_and_display_column():
    assert P.length_rough_m(48.0) == pytest.approx(8.54 * 48 ** 0.753)
    d = P.refine_status(pd.DataFrame({"product": ["A", "A"], "time": [pd.Timestamp("2026-10-12")] * 2,
                                      "lat": [58.0, 58.1], "lon": [20.0, 20.1], "length_m": [48.0, 10.0],
                                      "score": [0.9, 0.9], "status": ["dark", "dark"]}))
    assert list(d.status) == ["dark", "small"] and d.length_rough_m[0] > 150 and np.isnan(d.length_rough_m[1])


def test_reception_cells_flags_partial_outage(tmp_path):
    t = pd.Timestamp("2026-10-14 16:00")
    for k, n in ((0, 2), (1, 10), (2, 8), (3, 9)):      # today 2 ships heard, usually ~9
        day = t - pd.Timedelta(days=k)
        d = tmp_path / f"date={day.date()}"
        d.mkdir()
        pd.DataFrame({"mmsi": range(n), "ts": [day] * n, "lat": 59.84, "lon": 25.7, "sog": 0.0, "cog": 0.0,
                      "vessel_name": "x", "length": 200.0, "transceiver_class": "A"}).to_parquet(d / "a.parquet")
    rec = P.reception_cells(tmp_path, t, (25.0, 59.5, 26.5, 60.2))
    cell = P.cells_of([59.84], [25.7])[0]
    assert rec[cell] == pytest.approx(2 / 9)
    assert not P.covered([59.84], [25.7], {cell}, {cell}, rec)[0]
    assert P.covered([59.84], [25.7], {cell}, {cell}, None)[0]


def test_regate_marks_low_reception_dark(tmp_path, monkeypatch):
    from maritime.sar import run as R
    t = pd.Timestamp.now("UTC").tz_localize(None).floor("h")
    ais = tmp_path / "ais"
    for k, n in ((0, 1), (1, 9), (2, 9), (3, 9)):
        day = t - pd.Timedelta(days=k)
        d = ais / f"date={day.date()}"
        d.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"mmsi": range(n), "ts": [day] * n, "lat": 59.84, "lon": 25.7, "sog": 0.0, "cog": 0.0,
                      "vessel_name": "x", "length": 200.0, "transceiver_class": "A"}).to_parquet(d / f"a{k}.parquet")
    monkeypatch.setattr(R, "sar_dir", lambda name: tmp_path / "sar")
    f = tmp_path / "sar" / "detections" / f"date={t.date()}" / "S1X.parquet"
    f.parent.mkdir(parents=True)
    pd.DataFrame({"time": [t, t], "lat": [59.84, 59.84], "lon": [25.7, 25.7], "status": ["dark", "ais"],
                  "tier": ["high", ""]}).to_parquet(f)

    class Reg:
        name = "baltic"
    monkeypatch.setattr(P, "load_ais", lambda d, t0, t1, b, _f=P.load_ais: _f(ais, t0, t1, b))  # our AIS dir
    assert R.regate(Reg, hours=24, log=lambda *a: None) == 1
    assert list(pd.read_parquet(f)["status"]) == ["no_coverage", "ais"]
    assert R.regate(Reg, hours=24, log=lambda *a: None) == 0
