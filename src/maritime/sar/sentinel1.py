"""Sentinel-1 IW GRD: find, download and read products from the Copernicus Data Space Ecosystem.

- `search`   : CDSE OData catalogue (no login needed).
- `download` : one product zip (~1.8 GB), needs a free CDSE account (CDSE_USERNAME / CDSE_PASSWORD in .env).
- `read_grd` : calibrate to sigma0 with thermal-noise removal, quantise to the xView3 dB scale used in training,
               and return the image in its own radar geometry plus a `Geo` object for pixel <-> lat/lon/time.

Why radar geometry: GRDH pixels are 10 m x 10 m, the same scale as the xView3 training grid, and at sea the height is
0, so the product's geolocation grid gives positions without terrain correction or reprojection.
Calibration follows the ESA definition: sigma0 = (DN^2 - noise) / A_sigma^2, where A_sigma comes from the
calibration LUT and noise = range LUT x azimuth LUT, all bilinearly interpolated to every pixel.
"""

from __future__ import annotations

import hashlib
import time
import xml.etree.ElementTree as ET
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

CATALOGUE = "https://catalogue.dataspace.copernicus.eu/odata/v1/Products"
DOWNLOAD = "https://download.dataspace.copernicus.eu/odata/v1/Products({id})/$value"
TOKEN_URL = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
PRODUCT_TYPE = "IW_GRDH_1S"


# ---------------------------------------------------------------- catalogue and download

def bbox_wkt(min_lon, min_lat, max_lon, max_lat) -> str:
    return (f"POLYGON(({min_lon} {min_lat},{max_lon} {min_lat},{max_lon} {max_lat},"
            f"{min_lon} {max_lat},{min_lon} {min_lat}))")


def search(wkt: str, start: str, end: str, session=None, page: int = 100) -> pd.DataFrame:
    """IW GRDH products whose footprint intersects `wkt` and whose sensing starts in [start, end) (ISO UTC).
    Returns id, name, start, end, size, md5, online, orbit_direction, platform; newest first."""
    import requests

    s = session or requests.Session()
    flt = (f"Collection/Name eq 'SENTINEL-1' and "
           f"Attributes/OData.CSC.StringAttribute/any(a:a/Name eq 'productType' and "
           f"a/OData.CSC.StringAttribute/Value eq '{PRODUCT_TYPE}') and "
           f"OData.CSC.Intersects(area=geography'SRID=4326;{wkt}') and "
           f"ContentDate/Start ge {start} and ContentDate/Start lt {end}")
    rows, skip = [], 0
    while True:
        r = s.get(CATALOGUE, params={"$filter": flt, "$top": page, "$skip": skip,
                                     "$orderby": "ContentDate/Start desc", "$expand": "Attributes"},
                  timeout=60)
        r.raise_for_status()
        items = r.json().get("value", [])
        rows += [_product_row(p) for p in items]
        if len(items) < page:
            break
        skip += page
    cols = ["id", "name", "start", "end", "footprint", "size", "md5", "online", "orbit_direction", "platform"]
    return pd.DataFrame(rows, columns=cols)


def _product_row(p: dict) -> dict:
    attrs = {a.get("Name"): a.get("Value") for a in p.get("Attributes", [])}
    md5 = next((c.get("Value") for c in p.get("Checksum", []) if c.get("Algorithm") == "MD5"), None)
    fp = p.get("Footprint") or ""
    fp = fp.split(";", 1)[1].rstrip("'") if ";" in fp else None      # geography'SRID=4326;POLYGON(..)' -> WKT
    return {"id": p["Id"], "name": p["Name"], "start": p["ContentDate"]["Start"], "end": p["ContentDate"]["End"],
            "footprint": fp,
            "size": p.get("ContentLength"), "md5": md5, "online": p.get("Online", True),
            "orbit_direction": attrs.get("orbitDirection"), "platform": attrs.get("platformSerialIdentifier")}


class TokenSource:
    """CDSE access tokens (valid ~10 min) from username/password; refreshed when close to expiry."""

    def __init__(self, username: str, password: str, session=None):
        import requests
        self.username, self.password = username, password
        self.s = session or requests.Session()
        self._tok, self._exp = None, 0.0

    def __call__(self) -> str:
        if self._tok is None or time.time() > self._exp - 60:
            r = self.s.post(TOKEN_URL, data={"grant_type": "password", "client_id": "cdse-public",
                                             "username": self.username, "password": self.password}, timeout=60)
            if r.status_code != 200:
                raise RuntimeError(f"CDSE login failed (HTTP {r.status_code}); check CDSE_USERNAME/CDSE_PASSWORD")
            j = r.json()
            self._tok, self._exp = j["access_token"], time.time() + float(j.get("expires_in", 600))
        return self._tok


def download(product: dict, out_dir: Path, token, session=None, retries: int = 3, chunk: int = 1 << 22,
             log=print) -> Path:
    """Stream one product zip to out_dir/<name>.zip, checking the catalogue MD5. Skips if already complete."""
    import requests

    s = session or requests.Session()
    out_dir.mkdir(parents=True, exist_ok=True)
    name = product["name"].removesuffix(".SAFE")
    dst = out_dir / f"{name}.zip"
    if dst.exists() and (product.get("size") in (None, dst.stat().st_size)):
        return dst
    part = dst.with_suffix(".zip.part")
    url = DOWNLOAD.format(id=product["id"])
    for attempt in range(1, retries + 1):
        try:
            h = hashlib.md5()
            n = 0
            with s.get(url, headers={"Authorization": f"Bearer {token()}"}, stream=True, timeout=120) as r:
                r.raise_for_status()
                with open(part, "wb") as f:
                    for b in r.iter_content(chunk):
                        f.write(b)
                        h.update(b)
                        n += len(b)
            if product.get("md5") and h.hexdigest() != product["md5"].lower():
                raise OSError(f"MD5 mismatch for {name}")
            part.replace(dst)
            log(f"downloaded {name} ({n / 1e9:.2f} GB)")
            return dst
        except (OSError, requests.RequestException) as e:
            log(f"download attempt {attempt}/{retries} failed for {name}: {e}")
            if attempt < retries:
                time.sleep(10 * attempt)
    part.unlink(missing_ok=True)
    raise RuntimeError(f"could not download {name}")


# ---------------------------------------------------------------- reading a GRD product

def _floats(el) -> np.ndarray:
    return np.array(el.text.split(), dtype=np.float64) if el is not None and el.text else np.zeros(0)


def _t(s: str) -> np.datetime64:
    return np.datetime64(s.strip().rstrip("Z"), "us")


def parse_calibration(xml: bytes):
    """-> (lines (n,), [pixels (m_i,)], [sigmaNought (m_i,)])."""
    root = ET.fromstring(xml)
    lines, pix, val = [], [], []
    for v in root.iter("calibrationVector"):
        lines.append(int(v.findtext("line")))
        pix.append(_floats(v.find("pixel")))
        val.append(_floats(v.find("sigmaNought")))
    return np.array(lines), pix, val


def parse_noise(xml: bytes):
    """-> range part (lines, [pixels], [lut]) and azimuth blocks [(line0, line1, px0, px1, lines, lut)]."""
    root = ET.fromstring(xml)
    lines, pix, val = [], [], []
    for v in root.iter("noiseRangeVector"):
        lines.append(int(v.findtext("line")))
        pix.append(_floats(v.find("pixel")))
        val.append(_floats(v.find("noiseRangeLut")))
    blocks = []
    for v in root.iter("noiseAzimuthVector"):
        blocks.append((int(v.findtext("firstAzimuthLine")), int(v.findtext("lastAzimuthLine")),
                       int(v.findtext("firstRangeSample")), int(v.findtext("lastRangeSample")),
                       _floats(v.find("line")), _floats(v.find("noiseAzimuthLut"))))
    return (np.array(lines), pix, val), blocks


def lut_rows(lines: np.ndarray, pix: list, val: list, r0: int, r1: int, width: int) -> np.ndarray:
    """Bilinear interpolation of a sparse (line, pixel) LUT onto rows r0..r1-1 and all `width` columns."""
    cols = np.arange(width, dtype=np.float64)
    full = np.stack([np.interp(cols, p, v) for p, v in zip(pix, val, strict=True)])   # (n_vectors, W)
    rows = np.arange(r0, r1, dtype=np.float64)
    i = np.clip(np.searchsorted(lines, rows, side="right") - 1, 0, len(lines) - 2) if len(lines) > 1 else None
    if i is None:
        return np.repeat(full[:1], len(rows), 0).astype(np.float32)
    w = np.clip((rows - lines[i]) / np.maximum(lines[i + 1] - lines[i], 1), 0, 1)[:, None]
    return ((1 - w) * full[i] + w * full[i + 1]).astype(np.float32)


def noise_azimuth_rows(blocks, r0: int, r1: int, width: int) -> np.ndarray:
    """Azimuth noise scaling for rows r0..r1-1 (1 where no block covers a pixel)."""
    out = np.ones((r1 - r0, width), np.float32)
    rows = np.arange(r0, r1, dtype=np.float64)
    for a0, a1, p0, p1, lines, lut in blocks:
        lo, hi = max(a0, r0), min(a1, r1 - 1)
        if hi < lo or len(lines) == 0:
            continue
        vals = np.interp(rows[lo - r0:hi - r0 + 1], lines, lut) if len(lines) > 1 else np.full(hi - lo + 1, lut[0])
        out[lo - r0:hi - r0 + 1, p0:p1 + 1] = vals[:, None]
    return out


def calibrate_rows(dn: np.ndarray, r0: int, cal, noise_rng, noise_az, floor: float = 1e-5) -> np.ndarray:
    """sigma0 (linear) for a strip of rows starting at r0. DN == 0 (no data) -> NaN."""
    h, w = dn.shape
    a = lut_rows(*cal, r0, r0 + h, w)
    nz = lut_rows(*noise_rng, r0, r0 + h, w) * noise_azimuth_rows(noise_az, r0, r0 + h, w) \
        if noise_rng is not None else 0.0
    d2 = dn.astype(np.float32) ** 2
    s0 = np.maximum((d2 - nz) / (a * a), floor)
    s0[dn == 0] = np.nan
    return s0


@dataclass
class Geo:
    """Pixel <-> lat/lon and per-line time from the product's geolocation grid (GCPs)."""
    line: np.ndarray
    pixel: np.ndarray
    lat: np.ndarray
    lon: np.ndarray
    time: np.ndarray          # datetime64[us] per GCP
    slant_range: np.ndarray | None = None   # metres per GCP (slantRangeTime * c / 2)
    incidence: np.ndarray | None = None     # degrees per GCP
    _fwd: object = field(default=None, repr=False)
    _inv: object = field(default=None, repr=False)

    def _forward(self):
        if self._fwd is None:
            from scipy.interpolate import LinearNDInterpolator
            pts = np.column_stack([self.line, self.pixel])
            self._fwd = LinearNDInterpolator(pts, np.column_stack([self.lat, self.lon]))
        return self._fwd

    def to_latlon(self, rows, cols):
        """Scene pixel (row = line, col = pixel) -> (lat, lon). Linear in the GCP triangulation; points
        outside the GCP hull (a few edge pixels) fall back to the nearest-edge plane fit."""
        rows, cols = np.atleast_1d(rows).astype(float), np.atleast_1d(cols).astype(float)
        ll = self._forward()(rows, cols)
        bad = np.isnan(ll[:, 0])
        if bad.any():
            ll[bad] = self._poly(np.column_stack([self.line, self.pixel]),
                                 np.column_stack([self.lat, self.lon]), np.column_stack([rows[bad], cols[bad]]))
        return ll[:, 0], ll[:, 1]

    def to_rowcol(self, lat, lon):
        """(lat, lon) -> scene (row, col); outside the grid via a 2nd-order polynomial fit (for masks)."""
        lat, lon = np.atleast_1d(lat).astype(float), np.atleast_1d(lon).astype(float)
        if self._inv is None:
            from scipy.interpolate import LinearNDInterpolator
            self._inv = LinearNDInterpolator(np.column_stack([self.lat, self.lon]),
                                             np.column_stack([self.line, self.pixel]))
        rc = self._inv(lat, lon)
        bad = np.isnan(rc[:, 0])
        if bad.any():
            rc[bad] = self._poly(np.column_stack([self.lat, self.lon]), np.column_stack([self.line, self.pixel]),
                                 np.column_stack([lat[bad], lon[bad]]))
        return rc[:, 0], rc[:, 1]

    @staticmethod
    def _poly(src, dst, q):
        def feats(x):
            a, b = x[:, 0], x[:, 1]
            return np.column_stack([np.ones_like(a), a, b, a * a, a * b, b * b])
        coef, *_ = np.linalg.lstsq(feats(src), dst, rcond=None)
        return feats(q) @ coef

    def along_track(self, lat0: float, lon0: float) -> np.ndarray:
        """Unit vector (east, north) of increasing line = the satellite's flight direction, in local metres."""
        from maritime.fusion import to_local_m
        r = float(np.median(self.line))
        c = float(np.median(self.pixel))
        la, lo = self.to_latlon([r, r + 1000], [c, c])
        x, y = to_local_m(la, lo, lat0, lon0)
        v = np.array([x[1] - x[0], y[1] - y[0]])
        return v / np.linalg.norm(v)

    def incidence_at(self, rows, cols) -> np.ndarray:
        """Incidence angle (deg) at scene pixels; 35 if the grid has none."""
        if self.incidence is None:
            return np.full(len(np.atleast_1d(rows)), 35.0)
        from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator
        pts = np.column_stack([self.line, self.pixel])
        q = np.column_stack([np.atleast_1d(rows), np.atleast_1d(cols)]).astype(float)
        v = LinearNDInterpolator(pts, self.incidence)(q)
        bad = np.isnan(v)
        if bad.any():
            v[bad] = NearestNDInterpolator(pts, self.incidence)(q[bad])
        return v

    def times(self, rows) -> np.ndarray:
        """Acquisition time (datetime64[us]) of scene rows, interpolated from the GCP azimuth times."""
        order = np.argsort(self.line, kind="stable")
        ln = self.line[order]
        t = self.time[order].astype("int64")
        uniq, idx = np.unique(ln, return_index=True)
        return np.interp(np.atleast_1d(rows).astype(float), uniq, t[idx]).astype("int64").astype("datetime64[us]")


C_LIGHT = 299_792_458.0


def parse_geolocation(xml: bytes) -> Geo:
    root = ET.fromstring(xml)
    nan = float("nan")
    num = lambda p, k: float(p.findtext(k)) if p.findtext(k) else nan   # noqa: E731
    g = [(int(p.findtext("line")), int(p.findtext("pixel")), float(p.findtext("latitude")),
          float(p.findtext("longitude")), _t(p.findtext("azimuthTime")),
          num(p, "slantRangeTime") * C_LIGHT / 2, num(p, "incidenceAngle"))
         for p in root.iter("geolocationGridPoint")]
    a = list(zip(*g, strict=True))
    sr, inc = np.array(a[5]), np.array(a[6])
    return Geo(np.array(a[0], float), np.array(a[1], float), np.array(a[2]), np.array(a[3]),
               np.array(a[4], dtype="datetime64[us]"), None if np.isnan(sr).all() else sr,
               None if np.isnan(inc).all() else inc)


def platform_speed(xml: bytes) -> float:
    """Satellite speed (m/s) from the first orbit state vector; 7,600 if missing."""
    v = ET.fromstring(xml).find(".//orbit/velocity")
    if v is None:
        return 7600.0
    return float(np.linalg.norm([float(v.findtext(k)) for k in ("x", "y", "z")]))


@dataclass
class GRDScene:
    name: str
    image: np.ndarray            # (2, H, W) uint8: VV, VH on the xView3 dB scale (0 = no data)
    geo: Geo
    start: np.datetime64
    stop: np.datetime64
    platform: str
    orbit_direction: str
    speed_ms: float = 7600.0


def _members(z: zipfile.ZipFile, pol: str) -> dict:
    names = z.namelist()

    def one(pred):
        hits = [n for n in names if pred(n)]
        if len(hits) != 1:
            raise ValueError(f"expected one file for {pol}, found {len(hits)}: {hits[:4]}")
        return hits[0]
    tag = f"-grd-{pol}-"
    parent = lambda n: n.rstrip("/").split("/")[-2] if n.count("/") >= 1 else ""   # noqa: E731
    base = lambda n: n.rsplit("/", 1)[-1]                                          # noqa: E731
    # match files directly inside each folder only: newer products also carry e.g. annotation/rfi/rfi-*.xml
    return {
        "tiff": one(lambda n: parent(n) == "measurement" and tag in n and n.endswith(".tiff")),
        "annotation": one(lambda n: parent(n) == "annotation" and base(n).startswith("s1") and tag in n
                          and n.endswith(".xml")),
        "calibration": one(lambda n: parent(n) == "calibration" and base(n).startswith("calibration-") and tag in n),
        "noise": one(lambda n: parent(n) == "calibration" and base(n).startswith("noise-") and tag in n),
    }


def read_grd(zip_path: Path, strip: int = 1024, log=print) -> GRDScene:
    """Calibrated, noise-removed, quantised VV/VH of an IW GRDH zip, in radar geometry. ~1 GB of RAM."""
    import rasterio

    from maritime.sar.xview3 import quantise

    zip_path = Path(zip_path)
    with zipfile.ZipFile(zip_path) as z:
        m = {pol: _members(z, pol) for pol in ("vv", "vh")}
        ann = z.read(m["vv"]["annotation"])
        cal = {pol: parse_calibration(z.read(m[pol]["calibration"])) for pol in m}
        noi = {pol: parse_noise(z.read(m[pol]["noise"])) for pol in m}
    root = ET.fromstring(ann)
    geo = parse_geolocation(ann)
    start, stop = _t(root.findtext(".//adsHeader/startTime")), _t(root.findtext(".//adsHeader/stopTime"))
    platform = root.findtext(".//adsHeader/missionId") or ""
    direction = root.findtext(".//productInformation/pass") or ""

    img = None
    for b, pol in enumerate(("vv", "vh")):
        with rasterio.open(f"/vsizip/{zip_path.resolve()}/{m[pol]['tiff']}") as src:
            H, W = src.height, src.width
            if img is None:
                img = np.zeros((2, H, W), np.uint8)
            (nr, nb) = noi[pol]
            for r0 in range(0, H, strip):
                h = min(strip, H - r0)
                dn = src.read(1, window=rasterio.windows.Window(0, r0, W, h))
                s0 = calibrate_rows(dn, r0, cal[pol], nr if len(nr[0]) else None, nb)
                with np.errstate(invalid="ignore", divide="ignore"):
                    img[b, r0:r0 + h] = quantise(10 * np.log10(s0))
    log(f"read {zip_path.name}: {img.shape[1]} x {img.shape[2]} px, "
        f"{(img[0] > 0).mean():.0%} valid")
    return GRDScene(zip_path.stem, img, geo, start, stop, platform, direction, platform_speed(ann))
