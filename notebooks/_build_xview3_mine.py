"""Builds notebooks/xview3_mine.ipynb. Edit here, then: python3 notebooks/_build_xview3_mine.py"""
import json
from pathlib import Path

cells = []
def md(s): cells.append(("markdown", s.strip("\n")))
def code(s): cells.append(("code", s.strip("\n")))

md("""
# xView3 hard-negative mining

Runs the current models (xview3_train Version 5: VV, VH, 3 seeds) over 20 whole **training** scenes that
have human-checked labels, finds confident detections with no label nearby (false alarms), and cuts
512 px training tiles around them. Retraining with these tiles teaches the model what the sea, coast
and clutter that fooled it look like.

Only manually labelled (xView3 validation) scenes are mined: in AIS-only scenes an "unlabelled
detection" may be a real dark ship, and calling it a negative would teach the model to miss exactly
the ships this project is about.

**Inputs:** `maritime-xview3-mine` dataset + `xview3_train` notebook output (Version 5). GPU on.
**Output:** `hn_<scene>.npz` tile files (same format as the training tiles) for the next training run.
""")

code("DETECTOR_SRC = " + repr(Path(__file__).resolve().parents[1].joinpath("src/maritime/sar/detector.py").read_text())
     + "\nprint(f'embedded detector code: {len(DETECTOR_SRC):,} chars')")

code("""
import glob, json, os, sys, time, zlib
import numpy as np, pandas as pd, torch

CFG = dict(channels=["VV", "VH"], min_score=float(os.environ.get("XV3_MIN_SCORE", 0.3)), radius_px=30.0, max_tiles_per_scene=150, batch=32)
ROOT = os.environ.get("XV3_DATA", "/kaggle/input")
OUT = os.environ.get("XV3_OUT", "/kaggle/working")
scene_files = sorted(p for p in glob.glob(ROOT + "/**/*.npz", recursive=True)
                     if "_shoreline" not in p and ("maritime-xview3-mine" in p or os.environ.get("XV3_ANY_DIR")))
weight_files = sorted(glob.glob(ROOT + "/**/detector_VV+VH@*.pt", recursive=True))
assert scene_files, "add the maritime-xview3-mine dataset as input"
assert weight_files, "add the xview3_train notebook output as input"
DATA = os.path.dirname(scene_files[0])
with open(f"{OUT}/maritime_detector.py", "w") as f:
    f.write(DETECTOR_SRC)
sys.path.insert(0, OUT)
import maritime_detector as D
dev = "cuda" if torch.cuda.is_available() else "cpu"
models = []
for w in weight_files:
    m = D.build_model("resnet18", pretrained=False, in_ch=len(CFG["channels"]))
    m.load_state_dict(torch.load(w, map_location="cpu"))
    models.append(m.to(dev).eval())
print(f"{len(scene_files)} scenes | {len(models)} models | {dev}")
""")

code("""
lab = pd.read_csv(f"{DATA}/labels.csv")
def as_float(s):
    return pd.to_numeric(s.map({True: 1.0, False: 0.0, "True": 1.0, "False": 0.0}).where(s.notna()), errors="coerce")
def gt_for(sid):
    g = lab[lab.scene_id == sid]
    shore = pd.to_numeric(g.distance_from_shore_km, errors="coerce") if "distance_from_shore_km" in g else np.nan
    return pd.DataFrame({"row": g.detect_scene_row.astype(float), "col": g.detect_scene_column.astype(float),
                         "is_vessel": as_float(g.is_vessel), "is_fishing": as_float(g.is_fishing),
                         "length_m": pd.to_numeric(g.vessel_length_m, errors="coerce"),
                         "confidence": g.confidence.fillna("").astype(str), "shore_km": shore}).to_records(index=False)
def shoreline(sid):
    p = f"{DATA}/{sid}_shoreline.npy"
    return list(np.load(p, allow_pickle=True)) if os.path.exists(p) else None
""")

md("""
## Run the models, collect false alarms, cut tiles
A false alarm here = ensemble score >= 0.3 with no label of any confidence within 30 px (300 m). The
radius is looser than the 200 m scoring radius so that slightly misplaced labels are not mistaken for
empty sea. Up to 150 tiles per scene, highest-scoring false alarms first.
""")

code("""
DETS, GT, SHORE, report = {}, {}, {}, []
for p in scene_files:
    sid = os.path.basename(p)[:-4]
    t0 = time.time()
    img = np.load(p)["image"]
    dets, origins = D.predict_scene(models, img, CFG["channels"], batch=CFG["batch"], device=dev,
                                    num_workers=4 if dev == "cuda" else 0)
    ens = dets[-1]
    g = gt_for(sid); GT[sid] = g; SHORE[sid] = shoreline(sid); DETS[sid] = ens
    fa = D.unmatched_detections(ens, np.column_stack([g["row"], g["col"]]) if len(g) else np.zeros((0, 2)),
                                CFG["min_score"], CFG["radius_px"])
    seed = zlib.crc32(sid.encode())
    ims, orgs, objs = D.hard_negative_tiles(img, fa, g, CFG["max_tiles_per_scene"], seed=seed)
    np.savez_compressed(f"{OUT}/hn_{sid}.npz", images=ims, origins=orgs, objects=objs,
                        scene_shape=np.array(img.shape[1:]))
    near = D.shore_distance_px(fa[:, :2], SHORE[sid]) <= 200 if len(fa) else np.zeros(0, bool)
    report.append(dict(scene=sid, labels=len(g), false_alarms_ge03=len(fa),
                       false_alarms_ge04=int((fa[:, 2] >= 0.4).sum()) if len(fa) else 0,
                       near_shore_share=round(float(near.mean()), 2) if len(near) else None,
                       tiles=len(ims), objects_in_tiles=len(objs), seconds=round(time.time() - t0)))
    print(report[-1], flush=True)
    del img
rep = pd.DataFrame(report)
rep.to_csv(f"{OUT}/mining_report.csv", index=False)
print("\\ntotal tiles:", rep.tiles.sum(), "| false alarms >= 0.4:", rep.false_alarms_ge04.sum(),
      "| near-shore share:", round(float(np.nanmean(rep.near_shore_share.astype(float))), 2))
""")

md("""
## Calibration view: the current models on these scenes
These scenes were trained on (as tiles), so the scores are optimistic; the point is the *shape* of the
threshold curve on whole scenes. The real calibration for the retrained model happens in the eval
notebook.
""")

code("""
def to_rec(d, thr):
    d = d[d[:, 2] >= thr]
    return pd.DataFrame({"row": d[:, 0], "col": d[:, 1], "score": d[:, 2], "length_m": d[:, 3],
                         "is_vessel": d[:, 4], "is_fishing": d[:, 5]}).to_records(index=False)
sweep = pd.DataFrame([dict(threshold=t, **D.xview3_metrics({s: to_rec(d, t) for s, d in DETS.items()}, GT, SHORE))
                      for t in [0.3, 0.4, 0.5, 0.6, 0.7]])
print(sweep[["threshold", "precision", "recall", "f1_detect", "f1_shore", "aggregate", "fp"]].round(3).to_string(index=False))
sweep.to_csv(f"{OUT}/calibration_sweep_v5.csv", index=False)
""")

md("## What the false alarms look like (top 24 by score, VH)")

code("""
import matplotlib.pyplot as plt
top = []
for s, d in DETS.items():
    g = GT[s]
    fa = D.unmatched_detections(d, np.column_stack([g["row"], g["col"]]) if len(g) else np.zeros((0, 2)), 0.4, 30)
    top += [(s, r, c, sc) for r, c, sc in fa[:, :3]]
top = sorted(top, key=lambda t: -t[3])[:24]
fig, axes = plt.subplots(3, 8, figsize=(16, 6.6))
cache = {}
for ax, (s, r, c, sc) in zip(axes.flat, top):
    if s not in cache:
        cache = {s: np.load(f"{DATA}/{s}.npz")["image"]}
    r, c = int(r), int(c)
    ax.imshow(cache[s][1, max(r - 32, 0):r + 32, max(c - 32, 0):c + 32], cmap="gray", vmin=100, vmax=230)
    ax.set_title(f"{sc:.2f}", fontsize=8)
for ax in axes.flat:
    ax.axis("off")
plt.tight_layout(); plt.savefig(f"{OUT}/false_alarms.png", dpi=70)
""")

nb = {"nbformat": 4, "nbformat_minor": 5,
      "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
                   "language_info": {"name": "python"}},
      "cells": [{"cell_type": t, "metadata": {}, "source": s.splitlines(keepends=True),
                 **({"outputs": [], "execution_count": None} if t == "code" else {}),
                 "id": f"c{i:02d}"} for i, (t, s) in enumerate(cells)]}
Path(__file__).with_name("xview3_mine.ipynb").write_text(json.dumps(nb, indent=1) + "\n")
