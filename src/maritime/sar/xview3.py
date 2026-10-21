"""xView3 data preparation.

For each scene in an xView3 link list (the {tiny,validation,train}.txt files from
iuu.xview.us, in aria2 input format: a URL line followed by indented key=value options):
  1. download the scene archive (.tar.gz), verifying its SHA1 if the list gives one;
  2. extract only VV_dB.tif and VH_dB.tif (the two radar polarisations);
  3. cut the scene into 512 x 512 tiles; keep every tile that contains a labelled object
     plus a sample of empty-sea tiles (negatives); quantise dB to uint8;
  4. write one compressed .npz per scene and delete the raw files.
Scenes already done are skipped, so the job can be stopped and restarted, and a new
link list (after links expire) simply continues where the last one stopped.

Tile format (per scene npz):
  images   uint8 (n, 2, 512, 512)  channels VV, VH; 0 = no data, 1..255 = dB in [-50, 10]
  origins  int32 (n, 2)            (row, col) of the tile's top-left pixel in the scene
  objects  structured array        tile index, row/col inside the tile, is_vessel,
                                   is_fishing, vessel_length_m, confidence, shore distance
Pixels are 10 m, so 512 px = 5.12 km.
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import tarfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import requests

TILE = 512
DB_MIN, DB_MAX = -50.0, 10.0
# Ancillary layers shipped with each scene (500 m pixels, co-registered with the SAR bands). Kept
# whole per scene (they are small) so input-layer selection can be tested later.
ANCILLARY = ("bathymetry.tif", "owiWindSpeed.tif", "owiWindDirection.tif", "owiWindQuality.tif", "owiMask.tif")
BANDS = ("VV_dB.tif", "VH_dB.tif")
OBJECT_DTYPE = np.dtype([
    ("tile", "i4"), ("row", "i2"), ("col", "i2"),
    ("is_vessel", "f4"), ("is_fishing", "f4"), ("length_m", "f4"),
    ("confidence", "U6"), ("shore_km", "f4"),
])


# ---------------------------------------------------------------- link lists

@dataclass
class SceneLink:
    url: str
    out: str
    sha1: str | None = None
    options: dict = field(default_factory=dict)

    @property
    def scene_id(self) -> str:
        return re.sub(r"\.tar(\.gz)?$", "", Path(self.out).name)


def parse_links(path: Path) -> list[SceneLink]:
    """Parse an aria2 input file: URL lines, each followed by indented key=value lines."""
    links: list[SceneLink] = []
    for raw in Path(path).read_text().splitlines():
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        if not raw[0].isspace():
            url = raw.strip()
            name = Path(url.split("?", 1)[0]).name
            links.append(SceneLink(url=url, out=name))
        elif links and "=" in raw:
            key, value = raw.strip().split("=", 1)
            links[-1].options[key] = value
            if key == "out":
                links[-1].out = value
            elif key == "checksum" and value.lower().startswith("sha-1="):
                links[-1].sha1 = value.split("=", 1)[1].lower()
    return links


# ---------------------------------------------------------------- download

def download(link: SceneLink, dest_dir: Path, log=print, chunk: int = 1 << 22) -> Path:
    """Stream to disk with resume; verify SHA1 when known. Returns the file path."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    final = dest_dir / Path(link.out).name
    if final.exists():
        return final
    part = final.with_suffix(final.suffix + ".part")
    h = hashlib.sha1()
    have = 0
    if part.exists():                       # resume: re-hash what we already have
        with part.open("rb") as f:
            for block in iter(lambda: f.read(chunk), b""):
                h.update(block)
        have = part.stat().st_size
    headers = {"Range": f"bytes={have}-"} if have else {}
    with requests.get(link.url, stream=True, timeout=60, headers=headers) as r:
        if r.status_code in (401, 403):
            raise PermissionError("download link rejected (expired?): refresh the xView3 page "
                                  "and replace the link list")
        if have and r.status_code == 200:   # server ignored Range: start over
            h, have = hashlib.sha1(), 0
            part.unlink()
        r.raise_for_status()
        total = have + int(r.headers.get("content-length", 0))
        done, last = have, 0
        with part.open("ab") as f:
            for block in r.iter_content(chunk_size=chunk):
                f.write(block)
                h.update(block)
                done += len(block)
                if total and done - last > total / 10:
                    log(f"    {link.scene_id}: {done / 1e9:.1f} / {total / 1e9:.1f} GB")
                    last = done
    if link.sha1 and h.hexdigest() != link.sha1:
        part.unlink()
        raise ValueError(f"{link.out}: SHA1 mismatch; deleted, will re-download next run")
    part.rename(final)
    return final


def extract_bands(archive: Path, dest: Path, bands=BANDS, optional=ANCILLARY) -> dict[str, Path]:
    """Extract the wanted bands (by file name; ignores paths inside the archive).
    Missing `bands` raise; missing `optional` bands are skipped."""
    dest.mkdir(parents=True, exist_ok=True)
    found = {}
    with tarfile.open(archive, "r:*") as tar:
        for m in tar:
            name = Path(m.name).name
            if m.isfile() and (name in bands or name in optional):
                with tar.extractfile(m) as src, (dest / name).open("wb") as out:
                    shutil.copyfileobj(src, out, length=1 << 22)
                found[name] = dest / name
    missing = set(bands) - set(found)
    if missing:
        raise ValueError(f"{archive.name}: missing bands {sorted(missing)}")
    return found


# ---------------------------------------------------------------- tiling

def quantise(db: np.ndarray) -> np.ndarray:
    """dB -> uint8: 0 = no data, 1..255 = linear over [DB_MIN, DB_MAX] (clipped)."""
    valid = np.isfinite(db) & (db > -1000)             # xView3 no-data is a large negative
    q = np.clip((db - DB_MIN) / (DB_MAX - DB_MIN), 0, 1) * 254 + 1
    return np.where(valid, q, 0).astype(np.uint8)


def load_labels(label_dir: Path) -> pd.DataFrame:
    frames = [pd.read_csv(p) for p in sorted(Path(label_dir).glob("*.csv"))
              if "scene" not in p.stem.lower() and not p.stem.startswith(("GRD_", "SLC_"))]
    if not frames:
        raise FileNotFoundError(f"no label CSVs in {label_dir}")
    df = pd.concat(frames, ignore_index=True)
    need = {"scene_id", "detect_scene_row", "detect_scene_column"}
    if not need <= set(df.columns):
        raise ValueError(f"label CSVs lack {sorted(need - set(df.columns))}; columns: {list(df.columns)}")
    return df


def read_ancillary(bands: dict[str, Path]) -> dict[str, np.ndarray]:
    """Whole-scene ancillary layers as float16 arrays named anc_<layer> (NaN = no data)."""
    import rasterio
    out = {}
    for name in ANCILLARY:
        if name not in bands:
            continue
        with rasterio.open(bands[name]) as src:
            a = src.read(1, out_dtype="float32")
            if src.nodata is not None:
                a[a == src.nodata] = np.nan
        a[(a < -1e4) | (a > 1e5)] = np.nan          # xView3 fills no-data with large sentinels
        out["anc_" + name.removesuffix(".tif")] = a.astype(np.float16)
    return out


def tile_scene(bands: dict[str, Path], labels: pd.DataFrame, out_path: Path,
               neg_ratio: float = 1.0, min_valid: float = 0.5, seed: int = 0) -> dict:
    """Write positive tiles (>= 1 object) and sampled negative tiles for one scene."""
    import rasterio
    from rasterio.windows import Window

    rng = np.random.default_rng(seed)
    with rasterio.open(bands["VV_dB.tif"]) as vv, rasterio.open(bands["VH_dB.tif"]) as vh:
        H, W = vv.height, vv.width
        rows = labels["detect_scene_row"].to_numpy().astype(int)
        cols = labels["detect_scene_column"].to_numpy().astype(int)
        tr, tc = rows // TILE, cols // TILE
        positive = sorted(set(zip(tr.tolist(), tc.tolist(), strict=True)))
        n_r, n_c = -(-H // TILE), -(-W // TILE)
        pos_set = set(positive)
        candidates = [(r, c) for r in range(n_r) for c in range(n_c) if (r, c) not in pos_set]
        rng.shuffle(candidates)
        want_neg = int(round(neg_ratio * len(positive)))

        def read(r, c):
            win = Window(c * TILE, r * TILE, TILE, TILE)
            a = vv.read(1, window=win, boundless=True, fill_value=-32768, out_dtype="float32")
            b = vh.read(1, window=win, boundless=True, fill_value=-32768, out_dtype="float32")
            return np.stack([quantise(a), quantise(b)])

        images, origins, keep_tiles = [], [], {}
        for r, c in positive:
            keep_tiles[(r, c)] = len(images)
            images.append(read(r, c))
            origins.append((r * TILE, c * TILE))
        negatives = 0
        for r, c in candidates:
            if negatives >= want_neg:
                break
            img = read(r, c)
            if (img[0] > 0).mean() < min_valid:          # mostly land / no data
                continue
            images.append(img)
            origins.append((r * TILE, c * TILE))
            negatives += 1

    objs = np.zeros(len(labels), dtype=OBJECT_DTYPE)
    objs["tile"] = [keep_tiles[(a, b)] for a, b in zip(tr, tc, strict=True)]
    objs["row"], objs["col"] = rows - tr * TILE, cols - tc * TILE
    for src, dst in (("is_vessel", "is_vessel"), ("is_fishing", "is_fishing"),
                     ("vessel_length_m", "length_m"), ("distance_from_shore_km", "shore_km")):
        objs[dst] = pd.to_numeric(labels[src], errors="coerce") if src in labels else np.nan
    objs["confidence"] = labels["confidence"].fillna("").astype(str) if "confidence" in labels else ""

    anc = read_ancillary(bands)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, images=np.stack(images) if images else np.zeros((0, 2, TILE, TILE), np.uint8),
                        origins=np.array(origins, dtype=np.int32).reshape(-1, 2), objects=objs,
                        scene_shape=np.array([H, W]), **anc)
    tmp.replace(out_path)
    return {"positive_tiles": len(positive), "negative_tiles": negatives, "objects": int(len(objs)),
            "scene_px": [H, W], "ancillary": sorted(k[4:] for k in anc)}


# ---------------------------------------------------------------- driver

def prepare(links_file: Path, label_dir: Path, out_dir: Path, work_dir: Path,
            max_scenes: int | None = None, neg_ratio: float = 1.0, keep_raw: bool = False,
            log=print) -> list[dict]:
    labels = load_labels(label_dir)
    links = parse_links(links_file)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "manifest.jsonl"
    done = {json.loads(line)["scene_id"] for line in manifest_path.read_text().splitlines()} \
        if manifest_path.exists() else set()
    todo = [lk for lk in links if lk.scene_id not in done]
    if max_scenes is not None:
        todo = todo[:max(0, max_scenes - len(done & {lk.scene_id for lk in links}))]
    log(f"{len(links)} scenes in {links_file.name}; {len(done)} already done; doing {len(todo)}")
    results = []
    for i, link in enumerate(todo, 1):
        sid = link.scene_id
        scene_labels = labels[labels["scene_id"] == sid]
        log(f"[{i}/{len(todo)}] {sid}: {len(scene_labels)} labelled objects")
        archive = download(link, work_dir / "downloads", log=log)
        try:
            bands = extract_bands(archive, work_dir / "scenes" / sid)
            info = tile_scene(bands, scene_labels, out_dir / f"{sid}.npz", neg_ratio=neg_ratio)
        finally:
            shutil.rmtree(work_dir / "scenes" / sid, ignore_errors=True)
            if not keep_raw:
                archive.unlink(missing_ok=True)
        info.update(scene_id=sid, source=links_file.stem,
                    npz_mb=round((out_dir / f"{sid}.npz").stat().st_size / 1e6, 1))
        with manifest_path.open("a") as f:
            f.write(json.dumps(info) + "\n")
        log(f"    tiles: {info['positive_tiles']} with objects + {info['negative_tiles']} empty, "
            f"{info['npz_mb']} MB; ancillary: {', '.join(info['ancillary']) or 'NONE'}")
        results.append(info)
    return results


# ---------------------------------------------------------------- whole scenes (evaluation)

def read_scene_u8(bands: dict[str, Path], strip: int = 1024) -> np.ndarray:
    """Whole scene as (2, H, W) uint8 quantised dB, read in row strips so a 29k x 24k scene never
    needs the ~6 GB float32 copy (the server has 6 GB of RAM)."""
    import rasterio
    from rasterio.windows import Window

    with rasterio.open(bands["VV_dB.tif"]) as vv, rasterio.open(bands["VH_dB.tif"]) as vh:
        H, W = vv.height, vv.width
        out = np.empty((2, H, W), np.uint8)
        for r0 in range(0, H, strip):
            win = Window(0, r0, W, min(strip, H - r0))
            out[0, r0:r0 + win.height] = quantise(vv.read(1, window=win, out_dtype="float32"))
            out[1, r0:r0 + win.height] = quantise(vh.read(1, window=win, out_dtype="float32"))
    return out


def prepare_scenes(links_file: Path, scene_ids: list[str], label_dir: Path, shoreline_tar: Path | None,
                   out_dir: Path, work_dir: Path, keep_raw: bool = False, log=print) -> list[dict]:
    """Full-scene evaluation data for a few scenes: <sid>.npz (image (2,H,W) uint8), <sid>_shoreline.npy,
    and labels.csv (all labels of these scenes). Resumable; raw downloads deleted after each scene."""
    out_dir.mkdir(parents=True, exist_ok=True)
    wanted = list(dict.fromkeys(scene_ids))
    links = {lk.scene_id: lk for lk in parse_links(links_file)}
    missing = [s for s in wanted if s not in links]
    if missing:
        raise SystemExit(f"{len(missing)} scenes not in {links_file.name} (fresh links needed?): {missing[:3]}")
    labels = load_labels(label_dir)
    labels[labels["scene_id"].isin(wanted)].to_csv(out_dir / "labels.csv", index=False)
    if shoreline_tar is not None and shoreline_tar.exists():
        with tarfile.open(shoreline_tar, "r:*") as tar:
            for m in tar:
                name = Path(m.name).name
                if m.isfile() and name.removesuffix("_shoreline.npy") in wanted:
                    with tar.extractfile(m) as src, (out_dir / name).open("wb") as dst:
                        shutil.copyfileobj(src, dst)
    else:
        log("no shoreline archive: near-shore scores will be skipped")
    manifest = out_dir / "manifest.jsonl"
    done = {json.loads(x)["scene_id"] for x in manifest.read_text().splitlines()} if manifest.exists() else set()
    todo = [s for s in wanted if s not in done]
    log(f"{len(wanted)} evaluation scenes; {len(wanted) - len(todo)} already done; doing {len(todo)}")
    results = []
    for i, sid in enumerate(todo, 1):
        log(f"[{i}/{len(todo)}] {sid}: {int((labels['scene_id'] == sid).sum())} labelled objects")
        archive = download(links[sid], work_dir / "downloads", log=log)
        try:
            bands = extract_bands(archive, work_dir / "scenes" / sid, optional=())
            img = read_scene_u8(bands)
            tmp = out_dir / f"{sid}.tmp.npz"
            np.savez_compressed(tmp, image=img, scene_shape=np.array(img.shape[1:]))
            tmp.replace(out_dir / f"{sid}.npz")
        finally:
            shutil.rmtree(work_dir / "scenes" / sid, ignore_errors=True)
            if not keep_raw:
                archive.unlink(missing_ok=True)
        info = {"scene_id": sid, "scene_px": list(img.shape[1:]),
                "valid_frac": round(float((img[0] > 0).mean()), 3),
                "npz_mb": round((out_dir / f"{sid}.npz").stat().st_size / 1e6, 1),
                "shoreline": (out_dir / f"{sid}_shoreline.npy").exists()}
        del img
        with manifest.open("a") as f:
            f.write(json.dumps(info) + "\n")
        log(f"    {info['scene_px'][0]} x {info['scene_px'][1]} px, {info['valid_frac']:.0%} valid, "
            f"{info['npz_mb']} MB, shoreline: {'yes' if info['shoreline'] else 'NO'}")
        results.append(info)
    return results
