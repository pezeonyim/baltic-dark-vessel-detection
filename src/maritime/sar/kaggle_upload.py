"""Upload xView3 tiles to Kaggle as a PRIVATE dataset.

Uses the official `kaggle` CLI (>= 2.0). Auth from .env: KAGGLE_API_TOKEN (Kaggle's
current tokens, from kaggle.com > Settings > API) or the legacy KAGGLE_USERNAME + KAGGLE_KEY.
KAGGLE_USERNAME is needed either way, to name the dataset (<username>/<slug>). The dataset stays private: the
xView3 data is shared with registered users and should not be republished.
First call creates the dataset; later calls add a new version.

Besides the npz tiles, the dataset carries what the Kaggle notebook needs:
  maritime_detector.py  copy of maritime/sar/detector.py (model, losses, scoring)
  scenes.csv            scene_id, split (train/validation, from the xView3 label file it is in)
"""

from __future__ import annotations

import csv
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

DATASET_SLUG = "maritime-xview3-tiles"
SCENES_SLUG = "maritime-xview3-scenes"
MINE_SLUG = "maritime-xview3-mine"


def write_scene_splits(tiles_dir: Path, label_dir: Path) -> dict[str, int]:
    """scenes.csv: which official split each tiled scene belongs to (its label CSV's name)."""
    split_of: dict[str, str] = {}
    for p in sorted(Path(label_dir).glob("*.csv")):
        if p.stem not in ("train", "validation", "public"):
            continue
        with p.open(newline="") as f:
            for row in csv.DictReader(f):
                split_of.setdefault(row["scene_id"], p.stem)
    scenes = sorted(q.stem for q in tiles_dir.glob("*.npz"))
    counts: dict[str, int] = {}
    with (tiles_dir / "scenes.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["scene_id", "split"])
        for sid in scenes:
            sp = split_of.get(sid, "unknown")
            counts[sp] = counts.get(sp, 0) + 1
            w.writerow([sid, sp])
    return counts


def bundle(tiles_dir: Path, label_dir: Path, log=print) -> None:
    shutil.copyfile(Path(__file__).with_name("detector.py"), tiles_dir / "maritime_detector.py")
    counts = write_scene_splits(tiles_dir, label_dir)
    log(f"scenes by split: {counts}")


def upload(tiles_dir: Path, message: str = "update", label_dir: Path | None = None, log=print,
           kind: str = "tiles") -> str:
    """kind="tiles": training tiles (dataset maritime-xview3-tiles).
    kind="scenes": whole held-out scenes for evaluation (dataset maritime-xview3-scenes).
    kind="mine": whole training scenes for hard-negative mining (dataset maritime-xview3-mine)."""
    from dotenv import load_dotenv

    from maritime.paths import ROOT
    load_dotenv(ROOT / ".env")
    user = os.environ.get("KAGGLE_USERNAME", "").strip()
    has_token = bool(os.environ.get("KAGGLE_API_TOKEN", "").strip())
    has_key = bool(os.environ.get("KAGGLE_KEY", "").strip())
    if not user or not (has_token or has_key):
        raise SystemExit("set KAGGLE_USERNAME and KAGGLE_API_TOKEN in .env (Kaggle > Settings > API)")
    if not any(tiles_dir.glob("*.npz")):
        raise SystemExit(f"nothing to upload in {tiles_dir}")
    if kind == "tiles":
        bundle(tiles_dir, label_dir or tiles_dir.parent / "labels", log=log)
        slug, title = DATASET_SLUG, "Maritime xView3 tiles (private)"
        subtitle = "512px VV/VH tiles from xView3-SAR; registered-use data, keep private"
    else:
        shutil.copyfile(Path(__file__).with_name("detector.py"), tiles_dir / "maritime_detector.py")
        if kind == "mine":
            slug, title = MINE_SLUG, "Maritime xView3 mining scenes (private)"
            subtitle = "Whole xView3 scenes for hard-negative mining; registered-use, private"
        else:
            slug, title = SCENES_SLUG, "Maritime xView3 eval scenes (private)"
            subtitle = "Whole held-out xView3 validation scenes; registered-use data, keep private"
    ds_id = f"{user}/{slug}"
    meta = {"title": title, "id": ds_id, "licenses": [{"name": "other"}], "subtitle": subtitle}
    (tiles_dir / "dataset-metadata.json").write_text(json.dumps(meta, indent=2))
    marker = tiles_dir / ".kaggle_created"
    kaggle = str(Path(sys.executable).with_name("kaggle"))
    if marker.exists():
        cmd = [kaggle, "datasets", "version", "-p", str(tiles_dir), "-m", message, "-r", "skip"]
    else:
        cmd = [kaggle, "datasets", "create", "-p", str(tiles_dir), "-r", "skip"]   # private by default
    log(" ".join(cmd[1:]))
    subprocess.run(cmd, check=True)
    marker.touch()
    return f"https://www.kaggle.com/datasets/{ds_id}"
