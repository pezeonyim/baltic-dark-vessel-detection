"""Builds notebooks/xview3_scene_eval.ipynb. Edit here, then: python3 notebooks/_build_xview3_scene_eval.py"""
import json
from pathlib import Path

cells = []
def md(s): cells.append(("markdown", s.strip("\n")))
def code(s): cells.append(("code", s.strip("\n")))

md("""
# xView3 whole-scene evaluation

Runs trained detectors over **entire** held-out validation scenes and scores them with the full xView3
metric (detection, close-to-shore, vessel, fishing, length, aggregate). Real scenes are ~95% empty sea,
so every false alarm counts.

**Inputs (right panel > Add Input):** the `maritime-xview3-scenes` dataset (10 held-out scenes), the
output of the `xview3_train` notebook (the run to evaluate), and optionally the `maritime-xview3-mine`
dataset (calibration scenes). GPU on.

**Threshold:** if the mining scenes are attached, the detection threshold is chosen there
(best ensemble detection F1 on whole training scenes) *before* the held-out scenes are scored, so the
headline is not tuned on the test scenes. Without them, the tile-level choice 0.4 is used.
""")

code("DETECTOR_SRC = " + repr(Path(__file__).resolve().parents[1].joinpath("src/maritime/sar/detector.py").read_text())
     + "\nprint(f'embedded detector code: {len(DETECTOR_SRC):,} chars')")

code("""
import glob, json, os, sys, time
import numpy as np, pandas as pd, torch

CFG = dict(
    channels=["VV", "VH"],
    fallback_threshold=0.4,   # tile-level choice (Version 5), used only without calibration scenes
    cfar_z=6.0,
    batch=32,
    run_cfar=os.environ.get("XV3_CFAR", "0") == "1",   # CFAR was scored in the first eval run (F1 0.063)
)
ROOT = os.environ.get("XV3_DATA", "/kaggle/input")
OUT = os.environ.get("XV3_OUT", "/kaggle/working")
def scenes_in(tag):
    return sorted(p for p in glob.glob(ROOT + "/**/*.npz", recursive=True)
                  if tag in p and "_shoreline" not in p and not os.path.basename(p).startswith("hn_"))
held_files = scenes_in(os.environ.get("XV3_HELD_TAG", "maritime-xview3-scenes"))
calib_files = scenes_in(os.environ.get("XV3_CALIB_TAG", "maritime-xview3-mine"))
weight_files = sorted(glob.glob(ROOT + "/**/detector_*.pt", recursive=True))
assert held_files, "add the maritime-xview3-scenes dataset as input"
assert weight_files, "add the xview3_train notebook output as input"
with open(f"{OUT}/maritime_detector.py", "w") as f:
    f.write(DETECTOR_SRC)
sys.path.insert(0, OUT)
import maritime_detector as D
dev = "cuda" if torch.cuda.is_available() else "cpu"
print(f"held-out scenes {len(held_files)} | calibration scenes {len(calib_files)} | {dev}")
print("weights:", [os.path.basename(w) for w in weight_files])
""")

md("## Labels and shorelines")

code("""
def as_float(s):
    return pd.to_numeric(s.map({True: 1.0, False: 0.0, "True": 1.0, "False": 0.0}).where(s.notna()), errors="coerce")
def load_set(files):
    gt, shore = {}, {}
    if not files:
        return gt, shore
    data = os.path.dirname(files[0])
    lab = pd.read_csv(f"{data}/labels.csv")
    for p in files:
        sid = os.path.basename(p)[:-4]
        g = lab[lab.scene_id == sid]
        gt[sid] = pd.DataFrame({"row": g.detect_scene_row.astype(float), "col": g.detect_scene_column.astype(float),
                                "is_vessel": as_float(g.is_vessel), "is_fishing": as_float(g.is_fishing),
                                "length_m": pd.to_numeric(g.vessel_length_m, errors="coerce"),
                                "confidence": g.confidence.fillna("").astype(str)}).to_records(index=False)
        sp = f"{data}/{sid}_shoreline.npy"
        if os.path.exists(sp):
            shore[sid] = list(np.load(sp, allow_pickle=True))
    return gt, shore
GT, SHORE = load_set(held_files)
GT_C, SHORE_C = load_set(calib_files)
DATA = os.path.dirname(held_files[0])
print({s: len(g) for s, g in GT.items()})
print("shorelines for", len(SHORE), "of", len(GT), "held-out scenes;", len(GT_C), "calibration scenes")
""")

md("""
## Sliding-window inference
Each scene is cut into 512 px tiles overlapping by 64 px. Detections within 32 px of a tile edge are
dropped (the neighbouring tile sees them with more context), then duplicates closer than 4 px (40 m)
are merged (overlap duplicates land within 1-2 px; real ships can moor close together). The three seed models are run separately and as an **ensemble** (their outputs averaged),
which usually helps a little for free.
""")

code("""
models = []
for w in weight_files:
    m = D.build_model("resnet18", pretrained=False, in_ch=len(CFG["channels"]))
    m.load_state_dict(torch.load(w, map_location="cpu"))
    models.append(m.to(dev).eval())
names = [os.path.basename(w)[len("detector_"):-3] for w in weight_files] + ["ensemble"]

def run_cfar(img, origins):
    from concurrent.futures import ThreadPoolExecutor
    H, W = img.shape[1:]
    def one(o):
        t = img[:, o[0]:o[0] + 512, o[1]:o[1] + 512]
        c = D.cfar(t)
        if not len(c):
            return None
        c = np.column_stack([c[:, :2], c[:, 2], np.full(len(c), np.nan), np.ones(len(c)), np.zeros(len(c))])
        c = D.keep_core(c, o, (H, W), 512, 32)
        c[:, 0] += o[0]; c[:, 1] += o[1]
        return c
    with ThreadPoolExecutor(4) as ex:
        parts = [p for p in ex.map(one, origins) if p is not None and len(p)]
    return D.merge_close(np.concatenate(parts), 4.0) if parts else np.zeros((0, 6))

def run_set(files, cfar=False):
    dets_by = {n: {} for n in names}; cf = {}; timing = []
    for p in files:
        sid = os.path.basename(p)[:-4]
        t0 = time.time(); img = np.load(p)["image"]; t_load = time.time() - t0
        t0 = time.time()
        dets, origins = D.predict_scene(models, img, CFG["channels"], batch=CFG["batch"], device=dev,
                                        num_workers=4 if dev == "cuda" else 0)
        t_inf = time.time() - t0
        for n, d in zip(names, dets):
            dets_by[n][sid] = d
        if cfar:
            cf[sid] = run_cfar(img, origins)
        timing.append(dict(scene=sid, tiles=len(origins), valid_km2=round(float((img[0] > 0).sum()) * 1e-4),
                           load_s=round(t_load), infer_s=round(t_inf)))
        print(timing[-1], flush=True)
        del img
    return dets_by, cf, timing

def to_rec(d, thr):
    d = d[d[:, 2] >= thr]
    return pd.DataFrame({"row": d[:, 0], "col": d[:, 1], "score": d[:, 2], "length_m": d[:, 3],
                         "is_vessel": d[:, 4], "is_fishing": d[:, 5]}).to_records(index=False)
""")

md("""
## Step 1: choose the threshold on the calibration scenes (if attached)
""")

code("""
GRID = [round(x, 2) for x in np.arange(0.25, 0.751, 0.05)]
if calib_files:
    DETS_C, _, _ = run_set(calib_files)
    cal = pd.DataFrame([dict(threshold=t, **D.xview3_metrics({s: to_rec(d, t) for s, d in DETS_C["ensemble"].items()},
                                                             GT_C, SHORE_C)) for t in GRID])
    print(cal[["threshold", "precision", "recall", "f1_detect", "f1_shore", "aggregate", "fp"]].round(3).to_string(index=False))
    THR = float(cal.loc[cal.f1_detect.idxmax(), "threshold"])
    cal.to_csv(f"{OUT}/calibration_sweep.csv", index=False)
    print(f"\\ncalibrated threshold (best ensemble F1 on calibration scenes): {THR}")
else:
    THR = CFG["fallback_threshold"]
    print(f"no calibration scenes attached: using the tile-level threshold {THR}")
json.dump({"threshold": THR, "calibrated": bool(calib_files)}, open(f"{OUT}/threshold.json", "w"))
""")

md("## Step 2: run the held-out scenes")

code("""
DETS, CFAR, TIMING = run_set(held_files, cfar=CFG["run_cfar"])
pd.DataFrame(TIMING).to_csv(f"{OUT}/timing.csv", index=False)
""")

md("## Scores (full xView3 metric) at the pre-registered thresholds")

code("""
rows = []
for n in names:
    m = D.xview3_metrics({s: to_rec(d, THR) for s, d in DETS[n].items()}, GT, SHORE)
    rows.append(dict(model=n, threshold=THR, **m))
m = D.xview3_metrics({s: to_rec(d, 0.4) for s, d in DETS["ensemble"].items()}, GT, SHORE)
rows.append(dict(model="ensemble @0.4 (previous setting)", threshold=0.4, **m))
if CFAR:
    m = D.xview3_metrics({s: to_rec(d, CFG["cfar_z"]) for s, d in CFAR.items()}, GT, SHORE)
    rows.append(dict(model="CFAR", threshold=CFG["cfar_z"], **m))
res = pd.DataFrame(rows)
cols = ["model", "threshold", "precision", "recall", "f1_detect", "f1_shore", "f1_vessel", "f1_fishing",
        "length_score", "aggregate", "tp", "fp", "fn"]
print(res[cols].round(3).to_string(index=False))
print("\\nprevious runs (ensemble): V5 @0.4 F1 0.717 / shore 0.433 / aggregate 0.547; "
      "V6 +hn @0.45 F1 0.765 / shore 0.451 / aggregate 0.586 / FP 234")
print("xView3 reference (Faster R-CNN baseline, public/holdout scenes): F1 detect 0.43, aggregate 0.19")
res.to_csv(f"{OUT}/scene_scores.csv", index=False)
""")

md("## Per scene (ensemble): where do the false alarms come from?")

code("""
ens = {s: to_rec(d, THR) for s, d in DETS["ensemble"].items()}
area = {t["scene"]: t["valid_km2"] for t in TIMING}
per = []
for s in GT:
    m = D.xview3_metrics({s: ens[s]}, {s: GT[s]}, {s: SHORE[s]} if s in SHORE else None)
    per.append(dict(scene=s, labels=len(GT[s]), tp=m["tp"], fp=m["fp"], fn=m["fn"], f1=m["f1_detect"],
                    f1_shore=m["f1_shore"], fp_per_1000km2=1000 * m["fp"] / max(area[s], 1)))
per = pd.DataFrame(per).sort_values("f1")
print(per.round(3).to_string(index=False))
per.to_csv(f"{OUT}/per_scene.csv", index=False)
""")

md("## Diagnostic only: threshold sweep for the ensemble (not used for the headline numbers)")

code("""
sweep = pd.DataFrame([dict(threshold=t, **D.xview3_metrics({s: to_rec(d, t) for s, d in DETS["ensemble"].items()}, GT, SHORE))
                      for t in GRID])
print(sweep[["threshold", "precision", "recall", "f1_detect", "f1_shore", "aggregate"]].round(3).to_string(index=False))
""")

md("""
## Look at the mistakes
Top-scoring false alarms and missed labels (ensemble, pre-registered threshold), 64 px chips (VH).
These are the raw material for the next training round (hard negatives).
""")

code("""
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment
fps, fns = [], []
for s in GT:
    p, g = ens[s], GT[s]
    pairs = D.match(np.column_stack([p["row"], p["col"]]) if len(p) else np.zeros((0, 2)),
                    np.column_stack([g["row"], g["col"]]) if len(g) else np.zeros((0, 2)))
    mp, mg = {a for a, _ in pairs}, {b for _, b in pairs}
    fps += [(s, p["row"][i], p["col"][i], p["score"][i]) for i in range(len(p)) if i not in mp]
    fns += [(s, g["row"][i], g["col"][i], g["confidence"][i]) for i in range(len(g)) if i not in mg and g["confidence"][i] != "LOW"]
fps = sorted(fps, key=lambda t: -t[3])[:12]; fns = fns[:12]
pd.DataFrame(fps, columns=["scene", "row", "col", "score"]).to_csv(f"{OUT}/top_false_alarms.csv", index=False)
fig, axes = plt.subplots(2, 12, figsize=(24, 4.6))
cache = {}
for k, (items, title) in enumerate([(fps, "false alarm"), (fns, "missed")]):
    for j in range(12):
        ax = axes[k, j]; ax.axis("off")
        if j >= len(items):
            continue
        s, r, c, extra = items[j]
        if s not in cache:
            cache = {s: np.load(f"{DATA}/{s}.npz")["image"]}
        img = cache[s]; r, c = int(r), int(c)
        chip = img[1, max(r - 32, 0):r + 32, max(c - 32, 0):c + 32]
        ax.imshow(chip, cmap="gray", vmin=100, vmax=230)
        ax.set_title(f"{title}\\n{extra:.2f}" if isinstance(extra, float) else f"{title}\\n{extra}", fontsize=8)
plt.tight_layout(); plt.savefig(f"{OUT}/mistakes.png", dpi=70)
""")

md("## Save detections (for AIS fusion and the map)")

code("""
out = []
for s, d in DETS["ensemble"].items():
    out.append(pd.DataFrame({"scene_id": s, "row": d[:, 0], "col": d[:, 1], "score": d[:, 2], "length_m": d[:, 3],
                             "p_vessel": d[:, 4], "p_fishing": d[:, 5]}))
pd.concat(out).to_csv(f"{OUT}/detections_ensemble.csv.gz", index=False)
json.dump({"cfg": CFG, "threshold": THR, "scores": res[cols].to_dict("records")}, open(f"{OUT}/summary.json", "w"), indent=1, default=float)
print("saved", sum(len(d) for d in DETS["ensemble"].values()), "ensemble detections (score >= 0.05)")
""")

nb = {"nbformat": 4, "nbformat_minor": 5,
      "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
                   "language_info": {"name": "python"}},
      "cells": [{"cell_type": t, "metadata": {}, "source": s.splitlines(keepends=True),
                 **({"outputs": [], "execution_count": None} if t == "code" else {}),
                 "id": f"c{i:02d}"} for i, (t, s) in enumerate(cells)]}
Path(__file__).with_name("xview3_scene_eval.ipynb").write_text(json.dumps(nb, indent=1) + "\n")
