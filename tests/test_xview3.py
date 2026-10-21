"""xView3 prep on a synthetic scene: parse links, download (local HTTP), extract, tile."""

import hashlib
import http.server
import tarfile
import threading
from functools import partial

import numpy as np
import pandas as pd
import pytest

rasterio = pytest.importorskip("rasterio")
from maritime.sar import xview3  # noqa: E402


def make_scene(tmp_path, sid="abc123", h=1100, w=1300):
    d = tmp_path / "build" / sid
    d.mkdir(parents=True)
    rng = np.random.default_rng(0)
    for band in xview3.BANDS:
        a = rng.normal(-25, 3, (h, w)).astype("float32")
        a[:, :200] = -32768                       # a no-data strip (land / outside swath)
        a[700, 900] = 5.0                          # a bright target
        with rasterio.open(d / band, "w", driver="GTiff", height=h, width=w, count=1,
                           dtype="float32") as dst:
            dst.write(a, 1)
    # two of the five ancillary layers, at 1/50 resolution like the real 500 m grid
    for name, val in (("bathymetry.tif", -120.0), ("owiWindSpeed.tif", 7.5)):
        a = np.full((h // 50 + 1, w // 50 + 1), val, "float32")
        a[0, 0] = -99999.0                         # no-data sentinel
        with rasterio.open(d / name, "w", driver="GTiff", height=a.shape[0], width=a.shape[1],
                           count=1, dtype="float32") as dst:
            dst.write(a, 1)
    archive = tmp_path / "srv" / f"{sid}.tar.gz"
    archive.parent.mkdir()
    with tarfile.open(archive, "w:gz") as tar:
        tar.add(d, arcname=sid)
    return archive, hashlib.sha1(archive.read_bytes()).hexdigest()


@pytest.fixture
def server(tmp_path):
    handler = partial(http.server.SimpleHTTPRequestHandler, directory=str(tmp_path / "srv"))
    httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}"
    httpd.shutdown()


def test_parse_links(tmp_path):
    p = tmp_path / "l.txt"
    p.write_text("https://x/a/abc123.tar.gz?sig=1\n  out=abc123.tar.gz\n  checksum=sha-1=ABCDEF\n"
                 "https://x/b/def.tar.gz?sig=2\n")
    links = xview3.parse_links(p)
    assert [lk.scene_id for lk in links] == ["abc123", "def"]
    assert links[0].sha1 == "abcdef" and links[1].sha1 is None


def test_prepare_end_to_end(tmp_path, server):
    archive, sha = make_scene(tmp_path)
    links = tmp_path / "tiny.txt"
    links.write_text(f"{server}/abc123.tar.gz\n  out=abc123.tar.gz\n  checksum=sha-1={sha}\n")
    labels = tmp_path / "labels"
    labels.mkdir()
    pd.DataFrame({"scene_id": ["abc123", "other"], "detect_scene_row": [700, 5],
                  "detect_scene_column": [900, 5], "is_vessel": [True, False],
                  "is_fishing": [None, None], "vessel_length_m": [80.0, None],
                  "confidence": ["HIGH", "LOW"], "distance_from_shore_km": [12.0, 1.0]}
                 ).to_csv(labels / "validation.csv", index=False)
    out = tmp_path / "tiles"
    res = xview3.prepare(links, labels, out, tmp_path / "work", log=lambda *a: None)
    assert res[0]["positive_tiles"] == 1 and res[0]["objects"] == 1
    z = np.load(out / "abc123.npz")
    assert z["images"].dtype == np.uint8 and z["images"].shape[1:] == (2, 512, 512)
    o = z["objects"][0]
    t = o["tile"]
    assert tuple(z["origins"][t]) == (512, 512) and (o["row"], o["col"]) == (700 - 512, 900 - 512)
    assert z["images"][t, 0, o["row"], o["col"]] == 233          # 5 dB -> (55/60)*254+1
    assert not (tmp_path / "work" / "downloads" / "abc123.tar.gz").exists()   # raw deleted
    # ancillary layers kept whole; missing ones skipped; sentinels -> NaN
    assert res[0]["ancillary"] == ["bathymetry", "owiWindSpeed"]
    b = z["anc_bathymetry"].astype(np.float32)
    assert b.shape == (23, 27) and np.isnan(b[0, 0]) and b[5, 5] == -120.0
    assert z["anc_owiWindSpeed"].astype(np.float32)[3, 3] == 7.5
    # second run skips the finished scene
    assert xview3.prepare(links, labels, out, tmp_path / "work", log=lambda *a: None) == []


def test_quantise_nodata():
    q = xview3.quantise(np.array([-32768, -50, -20, 10, 30], dtype="float32"))
    assert q[0] == 0 and q[1] == 1 and q[3] == 255 and q[4] == 255


def test_bundle_writes_splits_and_detector(tmp_path):
    from maritime.sar.kaggle_upload import bundle
    tiles, labels = tmp_path / "tiles", tmp_path / "labels"
    tiles.mkdir()
    labels.mkdir()
    (labels / "train.csv").write_text("scene_id,detect_scene_row\naaa,1\naaa,2\n")
    (labels / "validation.csv").write_text("scene_id,detect_scene_row\nbbb,1\n")
    (labels / "GRD_train.csv").write_text("scene_id,x\nbbb,1\n")
    for sid in ("aaa", "bbb", "ccc"):
        (tiles / f"{sid}.npz").write_bytes(b"")
    bundle(tiles, labels, log=lambda *a: None)
    lines = (tiles / "scenes.csv").read_text().splitlines()
    assert lines == ["scene_id,split", "aaa,train", "bbb,validation", "ccc,unknown"]
    assert "def score" in (tiles / "maritime_detector.py").read_text()


def test_prepare_scenes_whole_scene(tmp_path, server):
    archive, sha = make_scene(tmp_path)
    links = tmp_path / "validation.txt"
    links.write_text(f"{server}/abc123.tar.gz\n  out=abc123.tar.gz\n  checksum=sha-1={sha}\n")
    labels = tmp_path / "labels"
    labels.mkdir()
    pd.DataFrame({"scene_id": ["abc123", "other"], "detect_scene_row": [700, 5],
                  "detect_scene_column": [900, 5], "confidence": ["HIGH", "LOW"]}
                 ).to_csv(labels / "validation.csv", index=False)
    shore_dir = tmp_path / "shore" / "validation"
    shore_dir.mkdir(parents=True)
    contour = np.empty(1, object)
    contour[0] = np.array([[0.0, 300.0], [1099.0, 300.0]])
    np.save(shore_dir / "abc123v_shoreline.npy", contour, allow_pickle=True)   # other scene: ignored
    np.save(shore_dir / "abc123_shoreline.npy", contour, allow_pickle=True)
    tar_path = tmp_path / "shoreline.tar.gz"
    with tarfile.open(tar_path, "w:gz") as tar:
        tar.add(tmp_path / "shore" / "validation", arcname="validation")
    out = tmp_path / "scenes"
    res = xview3.prepare_scenes(links, ["abc123"], labels, tar_path, out, tmp_path / "work", log=lambda *a: None)
    assert res[0]["scene_px"] == [1100, 1300] and res[0]["shoreline"]
    z = np.load(out / "abc123.npz")
    assert z["image"].shape == (2, 1100, 1300) and z["image"][0, 700, 900] == 233
    assert (z["image"][:, :, :200] == 0).all()                # no-data strip
    assert sorted(p.name for p in out.glob("*_shoreline.npy")) == ["abc123_shoreline.npy"]
    assert pd.read_csv(out / "labels.csv").scene_id.tolist() == ["abc123"]
    assert xview3.prepare_scenes(links, ["abc123"], labels, tar_path, out, tmp_path / "work",
                                 log=lambda *a: None) == []


def test_kaggle_subtitles_fit_kaggle_limits():
    import re
    from pathlib import Path
    src = Path(xview3.__file__).with_name("kaggle_upload.py").read_text()
    subs = re.findall(r'subtitle = "([^"]+)"', src)
    assert len(subs) == 3
    assert all(20 <= len(s) <= 80 for s in subs), [(s, len(s)) for s in subs]   # Kaggle rejects otherwise
