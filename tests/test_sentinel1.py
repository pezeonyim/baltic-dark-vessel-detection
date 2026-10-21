import zipfile

import numpy as np
import pytest

from maritime.sar import sentinel1 as S1
from maritime.sar.xview3 import quantise

H, W = 300, 400
NAME = "S1C_IW_GRDH_1SDV_20261001T160000_20261001T160025_001000_000100_ABCD"


def _cal_xml(a=500.0):
    vecs = "".join(f"<calibrationVector><azimuthTime>2026-10-01T16:00:{s:02d}.000000</azimuthTime>"
                   f"<line>{ln}</line><pixel count='3'>0 200 399</pixel>"
                   f"<sigmaNought count='3'>{a} {a} {a}</sigmaNought></calibrationVector>"
                   for s, ln in ((0, 0), (10, 150), (20, 299)))
    return f"<calibration><calibrationVectorList>{vecs}</calibrationVectorList></calibration>".encode()


def _noise_xml(rng=1000.0, az=1.0):
    rv = "".join(f"<noiseRangeVector><azimuthTime>x</azimuthTime><line>{ln}</line><pixel count='2'>0 399</pixel>"
                 f"<noiseRangeLut count='2'>{rng} {rng}</noiseRangeLut></noiseRangeVector>" for ln in (0, 299))
    av = (f"<noiseAzimuthVector><swath>IW1</swath><firstAzimuthLine>0</firstAzimuthLine>"
          f"<firstRangeSample>0</firstRangeSample><lastAzimuthLine>{H - 1}</lastAzimuthLine>"
          f"<lastRangeSample>{W - 1}</lastRangeSample><line count='2'>0 {H - 1}</line>"
          f"<noiseAzimuthLut count='2'>{az} {az}</noiseAzimuthLut></noiseAzimuthVector>")
    return (f"<noise><noiseRangeVectorList>{rv}</noiseRangeVectorList>"
            f"<noiseAzimuthVectorList>{av}</noiseAzimuthVectorList></noise>").encode()


def _ann_xml():
    pts = []
    for i, ln in enumerate((0, 150, 299)):
        for px in (0, 200, 399):
            pts.append(f"<geolocationGridPoint><azimuthTime>2026-10-01T16:00:{10 * i:02d}.000000</azimuthTime>"
                       f"<line>{ln}</line><pixel>{px}</pixel><latitude>{59.0 + ln * 1e-3}</latitude>"
                       f"<longitude>{24.0 + px * 1e-3}</longitude><height>0</height>"
                       f"<slantRangeTime>5.5e-03</slantRangeTime><incidenceAngle>{30 + px / 40}</incidenceAngle>"
                       "</geolocationGridPoint>")
    return ("<product><generalAnnotation><orbitList><orbit><velocity><x>6000</x><y>3000</y><z>-3600</z>"
            "</velocity></orbit></orbitList></generalAnnotation><adsHeader><missionId>S1C</missionId><startTime>2026-10-01T16:00:00.000000</startTime>"
            "<stopTime>2026-10-01T16:00:25.000000</stopTime></adsHeader>"
            "<generalAnnotation><productInformation><pass>Ascending</pass></productInformation></generalAnnotation>"
            f"<geolocationGrid><geolocationGridPointList>{''.join(pts)}</geolocationGridPointList>"
            "</geolocationGrid></product>").encode()


@pytest.fixture
def grd_zip(tmp_path):
    import rasterio
    dn = np.full((H, W), 100, np.uint16)
    dn[:10, :] = 0                                  # no-data rows
    dn[150, 300] = 2000                             # a bright ship
    safe = f"{NAME}.SAFE"
    zp = tmp_path / f"{NAME}.zip"
    with zipfile.ZipFile(zp, "w", zipfile.ZIP_STORED) as z:
        for pol in ("vv", "vh"):
            stem = f"s1c-iw-grd-{pol}-20261001t160000-20261001t160025-001000-000100-001"
            tif = tmp_path / f"{stem}.tiff"
            with rasterio.open(tif, "w", driver="GTiff", height=H, width=W, count=1, dtype="uint16") as d:
                d.write(dn, 1)
            z.write(tif, f"{safe}/measurement/{stem}.tiff")
            z.writestr(f"{safe}/annotation/{stem}.xml", _ann_xml())
            z.writestr(f"{safe}/annotation/calibration/calibration-{stem}.xml", _cal_xml())
            z.writestr(f"{safe}/annotation/calibration/noise-{stem}.xml", _noise_xml())
            z.writestr(f"{safe}/annotation/rfi/rfi-{stem}.xml", b"<rfi/>")         # newer products have these
    return zp


def test_lut_interpolation_is_bilinear():
    lines = np.array([0, 10])
    out = S1.lut_rows(lines, [np.array([0.0, 4.0]), np.array([0.0, 4.0])],
                      [np.array([0.0, 4.0]), np.array([10.0, 14.0])], 0, 11, 5)
    assert out.shape == (11, 5)
    assert out[0, 2] == pytest.approx(2.0) and out[10, 2] == pytest.approx(12.0) and out[5, 0] == pytest.approx(5.0)


def test_read_grd_calibrates_and_quantises(grd_zip):
    sc = S1.read_grd(grd_zip, strip=64, log=lambda *a: None)
    assert sc.image.shape == (2, H, W) and sc.platform == "S1C" and sc.orbit_direction == "Ascending"
    expected = quantise(np.array([10 * np.log10((100.0 ** 2 - 1000.0) / 500.0 ** 2)]))[0]
    assert sc.image[0, 100, 100] == expected and sc.image[1, 100, 100] == expected
    assert (sc.image[:, :10] == 0).all()                       # no data stays 0
    assert sc.image[0, 150, 300] > sc.image[0, 150, 299] + 50   # ship is bright


def test_geo_roundtrip_and_time(grd_zip):
    sc = S1.read_grd(grd_zip, log=lambda *a: None)
    lat, lon = sc.geo.to_latlon([150, 75], [200, 100])
    assert lat[0] == pytest.approx(59.15) and lon[0] == pytest.approx(24.2)
    assert lat[1] == pytest.approx(59.075) and lon[1] == pytest.approx(24.1)
    r, c = sc.geo.to_rowcol(lat, lon)
    assert r == pytest.approx([150, 75], abs=1e-6) and c == pytest.approx([200, 100], abs=1e-6)
    r, c = sc.geo.to_rowcol([59.4], [24.5])                     # outside the grid: polynomial fallback
    assert r[0] == pytest.approx(400, abs=1) and c[0] == pytest.approx(500, abs=1)
    t = sc.geo.times([0, 75, 299])
    assert str(t[1]) == "2026-10-01T16:00:05.000000" and str(t[2]) == "2026-10-01T16:00:20.000000"


def test_search_parses_and_pages():
    item = {"Id": "abc", "Name": f"{NAME}.SAFE", "ContentLength": 123,
            "ContentDate": {"Start": "2026-10-01T16:00:00Z", "End": "2026-10-01T16:00:25Z"},
            "Checksum": [{"Algorithm": "MD5", "Value": "ff"}],
            "Attributes": [{"Name": "orbitDirection", "Value": "ASCENDING"},
                           {"Name": "platformSerialIdentifier", "Value": "C"}]}

    class Sess:
        calls = []

        def get(self, url, params, timeout):
            self.calls.append(params)

            class R:
                def raise_for_status(self):
                    pass

                def json(self_inner):
                    return {"value": [item] * (2 if params["$skip"] == 0 else 1)}
            return R()
    s = Sess()
    df = S1.search(S1.bbox_wkt(8, 53, 30.5, 66), "2026-10-01T00:00:00Z", "2026-10-02T00:00:00Z", session=s, page=2)
    assert len(df) == 3 and len(s.calls) == 2 and df.md5[0] == "ff" and df.platform[0] == "C"
    assert "IW_GRDH_1S" in s.calls[0]["$filter"] and "Intersects" in s.calls[0]["$filter"]


def test_download_checks_md5(tmp_path):
    import hashlib
    body = b"x" * 1000

    class Sess:
        def get(self, url, headers, stream, timeout):
            class R:
                def __enter__(self):
                    return self

                def __exit__(self, *a):
                    return False

                def raise_for_status(self):
                    pass

                def iter_content(self, n):
                    yield body
            return R()
    p = {"id": "abc", "name": f"{NAME}.SAFE", "size": 1000, "md5": hashlib.md5(body).hexdigest()}
    out = S1.download(p, tmp_path, lambda: "tok", session=Sess(), log=lambda *a: None)
    assert out.read_bytes() == body
    bad = dict(p, md5="00", name="other.SAFE")
    with pytest.raises(RuntimeError):
        S1.download(bad, tmp_path, lambda: "tok", session=Sess(), retries=1, log=lambda *a: None)


def test_geo_incidence_range_and_speed(grd_zip):
    sc = S1.read_grd(grd_zip, log=lambda *a: None)
    assert sc.speed_ms == pytest.approx(np.sqrt(6000 ** 2 + 3000 ** 2 + 3600 ** 2))
    assert sc.geo.slant_range[0] == pytest.approx(5.5e-3 * 299_792_458 / 2)
    assert sc.geo.incidence_at([150], [200])[0] == pytest.approx(35.0)
    u = sc.geo.along_track(59.15, 24.2)
    assert u[1] == pytest.approx(1.0, abs=1e-6)              # lines go north in the fake grid
