"""Builds notebooks/xview3_train.ipynb (keeps the notebook diffable). Edit here, then run:
    python3 notebooks/_build_xview3_train.py
"""
import json
from pathlib import Path

cells = []
def md(s): cells.append(("markdown", s.strip("\n")))
def code(s): cells.append(("code", s.strip("\n")))

md("""
# xView3 vessel detector: training and input-layer comparison

Trains the point detector with the recipe from the label EDA and compares
input-layer sets. Settings > Accelerator > GPU T4 x2; add the `maritime-xview3-tiles`
dataset as input; internet on for ImageNet weights.

**Recipe:** xView3 *validation* scenes carry manual labels (incl. ships without AIS), so they
are the main supervision: a fixed set of them is held out for scoring, the rest are trained on (all
tiles). xView3 *train* scenes only label ships with AIS, so only their tiles that contain labels are
used; their "empty" tiles may hide unlabelled dark vessels.

**Caveat:** scores are on tiles, not whole scenes. Whole-scene scoring comes later.
""")

code("DETECTOR_SRC = " + repr(Path(__file__).resolve().parents[1].joinpath("src/maritime/sar/detector.py").read_text())
     + "\nprint(f'embedded detector code: {len(DETECTOR_SRC):,} chars')")

code("""
import glob, json, os, sys, time
import numpy as np, pandas as pd, torch

# Each run trains a fresh model on the same data. A run is "layers@seed".
# Layer options: VV, VH, VV_local, VH_local, VV_minus_VH, bathymetry, wind_speed, wind_dir_sin,
# wind_dir_cos, land_mask. Default: the input-layer confirmation run, two layer sets x three seeds, interleaved
# so a cut-short session still has balanced results.
# "/manual" = train only on the manual-label scenes (no AIS-only scenes), same number of steps
DEFAULT_RUNS = ";".join(f"VV,VH@{seed}/manual" for seed in (0, 1, 2))
RUNS = []
for r in os.environ.get("XV3_RUNS", DEFAULT_RUNS).split(";"):
    r, _, subset = r.partition("/")
    layers, _, seed = r.partition("@")
    RUNS.append((layers.split(","), int(seed or 0), subset or "all"))
CFG = dict(
    epochs=int(os.environ.get("XV3_EPOCHS", 12)),
    ref_train_tiles=int(os.environ.get("XV3_REF_TILES", 18593)),   # full-data tile count, for equal-step subsets   # fixed length: every run is compared at the same point
    warmup_epochs=1,               # linear warm-up, then cosine decay to ~0 at the last epoch
    lr=2e-4,
    ema_decay=0.999,               # score an exponential moving average of the weights (smooths noise)
    batch=int(os.environ.get("XV3_BATCH", 16)),
    backbone="resnet18",
    pretrained=os.environ.get("XV3_PRETRAINED", "1") == "1",
    threshold=0.3,
    heldout_scenes=10, split_seed=46,
    max_train_tiles=int(os.environ.get("XV3_MAX_TILES", 0)) or None,
)
ROOT = os.environ.get("XV3_DATA", "/kaggle/input")
# the training tiles live next to scenes.csv; other inputs (hard negatives, whole scenes) are elsewhere
csvs = glob.glob(ROOT + "/**/scenes.csv", recursive=True)
npz = glob.glob(os.path.dirname(csvs[0]) + "/*.npz") if csvs else glob.glob(ROOT + "/**/*.npz", recursive=True)
HN = sorted(glob.glob(ROOT + "/**/hn_*.npz", recursive=True))      # hard negatives from xview3_mine
if not npz:
    print("No tiles found under", ROOT)
    for dirpath, dirs, files in os.walk(ROOT, followlinks=True):
        if dirpath.count(os.sep) - ROOT.count(os.sep) < 4:
            print(" ", dirpath, files[:5])
    raise SystemExit("Add the maritime-xview3-tiles dataset: right panel > Input > Add Input.")
DATA = os.path.dirname(npz[0])
OUT = os.environ.get("XV3_OUT", "/kaggle/working")
with open(f"{OUT}/maritime_detector.py", "w") as f:    # always the code this notebook was built with
    f.write(DETECTOR_SRC)
sys.path.insert(0, OUT)
import maritime_detector as D
dev = "cuda" if torch.cuda.is_available() else "cpu"
print("data:", DATA, "| device:", dev, torch.cuda.get_device_name(0) if dev == "cuda" else "")
print("runs:", RUNS)
""")

md("""
## Split by scene (fixed seed, so every run is scored on the same scenes)
""")

code("""
scenes = (pd.read_csv(f"{DATA}/scenes.csv") if os.path.exists(f"{DATA}/scenes.csv") else
          pd.DataFrame({"scene_id": sorted(os.path.basename(p)[:-4] for p in npz), "split": "train"}))
scenes = scenes[scenes.scene_id.isin({os.path.basename(p)[:-4] for p in npz})]
val_scenes = sorted(scenes.scene_id[scenes.split == "validation"])
pool = val_scenes if len(val_scenes) >= 2 else sorted(scenes.scene_id)
n_held = min(CFG["heldout_scenes"], max(1, len(pool) // 2))
heldout = sorted(str(s) for s in np.random.default_rng(CFG["split_seed"]).choice(pool, n_held, replace=False))
manual = set(val_scenes) - set(heldout)                        # trained on, all tiles
ais_only = set(scenes.scene_id) - set(val_scenes) - set(heldout)  # trained on, positive tiles only
json.dump({"heldout": [str(s) for s in heldout], "manual_train": sorted(manual), "ais_only_train": sorted(ais_only)},
          open(f"{OUT}/split.json", "w"), indent=1)
print(f"held out {len(heldout)} | train: {len(manual)} manual-label scenes + {len(ais_only)} AIS-only scenes")
print("held-out scenes:", heldout)

path = lambda s: f"{DATA}/{s}.npz"
def keep_train(sid, counts):
    return counts > 0 if sid in ais_only else np.ones(len(counts), bool)
t0 = time.time()
need_ais = any(sub == "all" for *_, sub in RUNS)   # manual-only runs never load the AIS-only scenes
train = D.load_tiles([path(s) for s in sorted(manual | (ais_only if need_ais else set()))] + HN, keep=keep_train)
n_hn = int(sum(str(s).startswith("hn_") for s in train.scene))
print(f"hard-negative tiles: {n_hn} from {len(HN)} mined scenes" if HN else "no hard negatives attached")
val = D.load_tiles([path(s) for s in heldout])
if CFG["max_train_tiles"]:
    k = CFG["max_train_tiles"]
    train = D.TileSet(train.images[:k], train.origins[:k], train.scene[:k], train.objects[:k],
                      train.ancillary, train.scene_shape)
n_obj = lambda ts: sum(len(o) for o in ts.objects)
print(f"loaded in {time.time()-t0:.0f}s: train {len(train)} tiles / {n_obj(train)} objects "
      f"({train.images.nbytes/1e9:.1f} GB) | held-out {len(val)} tiles / {n_obj(val)} objects")
print("ancillary layers:", sorted({k for a in train.ancillary.values() for k in a}))
""")

md("""
## Data pipeline
Each tile becomes a stack of input layers (`make_input`), then gets random flips/rotations, then
its targets: a Gaussian bump on a 128 x 128 grid at every object centre, plus position, length,
vessel and fishing targets at the centre cell.
""")

code("""
class Tiles(torch.utils.data.Dataset):
    def __init__(self, ts, channels, aug):
        self.ts, self.channels, self.aug = ts, channels, aug
        self.rng = np.random.default_rng(int(time.time()))
    def __len__(self):
        return len(self.ts)
    def __getitem__(self, i):
        sid = self.ts.scene[i]
        x = D.make_input(self.ts.images[i], self.channels, self.ts.ancillary.get(sid),
                         self.ts.origins[i], self.ts.scene_shape.get(sid, (512, 512)))
        objs = self.ts.objects[i]
        if self.aug:
            x, objs = D.augment(x, objs, self.rng)
        heat, centres = D.render_targets(objs)
        return torch.from_numpy(np.ascontiguousarray(x)), torch.from_numpy(heat), centres

def collate(batch):
    xs, heats, cens = zip(*batch)
    return torch.stack(xs), torch.stack(heats), list(cens)

def loader(ts, channels, aug, shuffle):
    return torch.utils.data.DataLoader(Tiles(ts, channels, aug), batch_size=CFG["batch"], shuffle=shuffle,
                                       collate_fn=collate, num_workers=4 if dev == "cuda" else 0,
                                       persistent_workers=dev == "cuda")
""")

md("""
## Scoring (xView3 rules, per held-out scene, then pooled)
""")

code("""
def gts_by_scene(ts):
    out = {}
    for o, org, sid in zip(ts.objects, ts.origins, ts.scene):
        if len(o):
            out.setdefault(sid, []).append(pd.DataFrame({
                "row": o["row"] + org[0], "col": o["col"] + org[1], "is_vessel": o["is_vessel"],
                "is_fishing": o["is_fishing"], "length_m": o["length_m"], "confidence": o["confidence"],
                "shore_km": o["shore_km"]}))
    return {s: pd.concat(v).to_records(index=False) for s, v in out.items()}

def preds_by_scene(dets_per_tile, ts):
    out = {}
    for d, org, sid in zip(dets_per_tile, ts.origins, ts.scene):
        if len(d):
            out.setdefault(sid, []).append(pd.DataFrame({
                "row": d[:, 0] + org[0], "col": d[:, 1] + org[1], "score": d[:, 2],
                "length_m": d[:, 3], "is_vessel": d[:, 4], "is_fishing": d[:, 5]}))
    return {s: pd.concat(v).to_records(index=False) for s, v in out.items()}

def at(preds, t):
    return {s: p[p["score"] >= t] for s, p in preds.items()}

def near_shore_recall(preds, gts, km=2.0):
    # share of non-LOW labels within `km` of shore that some prediction lands within 200 m of
    hit = tot = 0
    for s, g in gts.items():
        m = (g["shore_km"] <= km) & (g["confidence"] != "LOW")
        if not m.any():
            continue
        p = preds.get(s)
        tot += int(m.sum())
        if p is None or not len(p):
            continue
        d = np.hypot(g["row"][m][:, None] - p["row"][None], g["col"][m][:, None] - p["col"][None]).min(1)
        hit += int((d <= 20).sum())
    return hit / tot if tot else float("nan")

GT = gts_by_scene(val)
print({s: len(g) for s, g in GT.items()})
""")

md("## Baseline: CFAR (no learning)")

code("""
cfar_dets = []
for img in val.images:
    c = D.cfar(img)
    cfar_dets.append(np.column_stack([c[:, :2], c[:, 2], np.full(len(c), np.nan), np.ones(len(c)),
                                      np.zeros(len(c))]) if len(c) else np.zeros((0, 6)))
cfar_preds = preds_by_scene(cfar_dets, val)
cfar_best = max((D.score(at(cfar_preds, z), GT) | {"z": z} for z in [5, 6, 8, 10, 12, 15]),
                key=lambda s: s["f1_detect"])
print(f"CFAR: best z={cfar_best['z']}  P={cfar_best['precision']:.3f} R={cfar_best['recall']:.3f} "
      f"F1={cfar_best['f1_detect']:.3f}")
""")

md("""
## Train each run
Same recipe for every run: fixed number of epochs, AdamW, linear warm-up then cosine decay, mixed
precision. Scores come from an exponential moving average (EMA) of the weights, which averages out
the step-to-step jitter that made single-epoch scores swing in the ablation.
""")

code("""
import math
from torch.optim.swa_utils import AveragedModel

def build(in_ch):
    try:
        return D.build_model(CFG["backbone"], pretrained=CFG["pretrained"], in_ch=in_ch)
    except Exception as e:
        print("pretrained weights unavailable, training from scratch:", type(e).__name__)
        return D.build_model(CFG["backbone"], pretrained=False, in_ch=in_ch)

@torch.no_grad()
def predict(model, dl):
    model.eval(); dets, losses = [], []
    for x, h, c in dl:
        out = {k: v.float() for k, v in model(x.to(dev)).items()}
        losses.append(D.compute_loss(out, h.to(dev), c)[0].item())
        dets += D.decode(out, threshold=0.05)
    return dets, float(np.mean(losses))

def lr_at(step, total, warm):
    if step < warm:
        return (step + 1) / warm
    return 0.5 * (1 + math.cos(math.pi * (step - warm) / max(1, total - warm)))

def subset_tiles(ts, subset):
    # 'all' = every training tile; 'manual' = drop the AIS-only scenes. Their labels only cover
    # ships with AIS, so unlit boats in them are trained as background; this run tests whether that hurts.
    if subset == "all":
        return ts
    keep = np.array([str(s) not in ais_only for s in ts.scene])
    if keep.all():                                   # already loaded without them: no copy
        return ts
    return D.TileSet(ts.images[keep], ts.origins[keep], ts.scene[keep],
                     [o for o, k in zip(ts.objects, keep) if k], ts.ancillary, ts.scene_shape)

def train_run(channels, seed, subset="all"):
    tag = "+".join(channels) + ("+hn" if HN else "") + ("" if subset == "all" else f"+{subset}")
    name = tag + f"@{seed}"
    run_train = subset_tiles(train, subset)
    # same number of optimizer steps as a full-data run (V6: 18,593 tiles x 12 epochs), so only the data differs
    epochs = CFG["epochs"] if subset == "all" else max(1, round(CFG["epochs"] * CFG["ref_train_tiles"] / len(run_train)))
    print(f"[{name}] {len(run_train)} of {len(train)} training tiles, {epochs} epochs")
    torch.manual_seed(seed); np.random.seed(seed)
    model = build(len(channels)).to(dev)
    # EMA with a warm-up: decay grows from ~0 towards CFG["ema_decay"], so the average does not drag the
    # untrained starting weights along for the first epochs (timm-style; found in the input-layer runs).
    ema_step = [0]
    @torch.no_grad()
    def ema_fn(ema_p, cur_p, num_averaged):
        ema_step[0] += 1
        d = min(CFG["ema_decay"], (1 + ema_step[0] / len(ema_p)) / (10 + ema_step[0] / len(ema_p)))
        for e, c in zip(ema_p, cur_p):
            if e.is_floating_point():
                e.lerp_(c, 1 - d)
            else:                                     # integer buffers (BatchNorm batch counters)
                e.copy_(c)
    ema = AveragedModel(model, multi_avg_fn=ema_fn, use_buffers=True)
    tdl, vdl = loader(run_train, channels, True, True), loader(val, channels, False, False)
    total, warm = epochs * len(tdl), CFG["warmup_epochs"] * len(tdl)
    opt = torch.optim.AdamW(model.parameters(), lr=CFG["lr"], weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda st: lr_at(st, total, warm))
    scaler = torch.amp.GradScaler(enabled=dev == "cuda")
    hist = []
    for ep in range(epochs):
        model.train(); t0 = time.time(); run = []
        for x, h, c in tdl:
            with torch.autocast(dev, enabled=dev == "cuda"):
                out = model(x.to(dev))
            out = {k: v.float() for k, v in out.items()}
            loss, _ = D.compute_loss(out, h.to(dev), c)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sched.step()
            ema.update_parameters(model)
            run.append(loss.item())
        dets, vloss = predict(ema.module, vdl)
        s = D.score(at(preds_by_scene(dets, val), CFG["threshold"]), GT)
        hist.append(dict(run=name, epoch=ep + 1, train_loss=np.mean(run), val_loss=vloss, **s))
        print(f"[{name}] ep {ep+1:2d} {time.time()-t0:4.0f}s lr {sched.get_last_lr()[0]:.1e} "
              f"loss {np.mean(run):.3f}/{vloss:.3f} P {s['precision']:.3f} R {s['recall']:.3f} "
              f"F1 {s['f1_detect']:.3f} vessel {s['f1_vessel']:.3f} fishing {s['f1_fishing']:.3f} "
              f"length {s['length_score']:.3f}", flush=True)
    torch.save(ema.module.state_dict(), f"{OUT}/detector_{name}.pt")
    preds = preds_by_scene(dets, val)                # final-epoch EMA model
    sweep = pd.DataFrame([D.score(at(preds, t), GT) | {"threshold": t} for t in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]])
    b = sweep.loc[sweep.f1_detect.idxmax()]
    per_scene = [D.score(at({s_: preds[s_]}, b.threshold) if s_ in preds else {}, {s_: GT[s_]})["f1_detect"]
                 for s_ in GT]
    f1_curve = [h_["f1_detect"] for h_ in hist]
    res = dict(layers=tag, seed=seed, threshold=b.threshold, precision=b.precision,
               recall=b.recall, f1_detect=b.f1_detect, f1_vessel=b.f1_vessel, f1_fishing=b.f1_fishing,
               length_score=b.length_score, near_shore_recall=near_shore_recall(at(preds, b.threshold), GT),
               scene_f1_min=float(np.min(per_scene)), scene_f1_max=float(np.max(per_scene)),
               last3_f1_spread=float(np.ptp(f1_curve[-3:])))
    return res, hist, dets

results, histories, first_dets = [], [], None
for channels, seed, subset in RUNS:
    res, hist, dets = train_run(channels, seed, subset)
    results.append(res); histories += hist
    first_dets = first_dets if first_dets is not None else (channels, dets, res["threshold"])
    pd.DataFrame(results).to_csv(f"{OUT}/runs.csv", index=False)
    pd.DataFrame(histories).to_csv(f"{OUT}/history.csv", index=False)
""")

md("## Compare runs: every seed, then mean and spread per layer set")

code("""
table = pd.DataFrame(results)
print(table.round(3).to_string(index=False))
metrics = ["f1_detect", "f1_vessel", "f1_fishing", "length_score", "near_shore_recall"]
summary = table.groupby("layers")[metrics].agg(["mean", "std", "count"]).round(3)
print("\\n", summary.to_string())
summary.to_csv(f"{OUT}/summary.csv")
print(f"\\nCFAR baseline F1 {cfar_best['f1_detect']:.3f}")
""")

md("## Look at held-out tiles (green = label, orange = detector, x = CFAR)")

code("""
import matplotlib.pyplot as plt
channels, dets, thr = first_dets
idx = [i for i, o in enumerate(val.objects) if len(o)][:4] or [0]
fig, axes = plt.subplots(1, len(idx), figsize=(5 * len(idx), 5), squeeze=False)
for ax, i in zip(axes[0], idx):
    ax.imshow(val.images[i][1], cmap="gray", vmin=100, vmax=230)
    o = val.objects[i]; d = dets[i]; d = d[d[:, 2] >= thr]; c = cfar_dets[i]
    ax.scatter(o["col"], o["row"], s=120, facecolors="none", edgecolors="lime", label="label")
    ax.scatter(d[:, 1], d[:, 0], s=40, facecolors="none", edgecolors="orange", label="detector")
    if len(c):
        c = c[c[:, 2] >= cfar_best["z"]]; ax.scatter(c[:, 1], c[:, 0], s=25, marker="x", c="red", label="CFAR")
    ax.set_title(f"{val.scene[i][:10]} tile {i}"); ax.axis("off")
axes[0][0].legend(loc="lower left")
plt.tight_layout(); plt.savefig(f"{OUT}/examples.png", dpi=80)
""")

nb = {"nbformat": 4, "nbformat_minor": 5,
      "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
                   "language_info": {"name": "python"}},
      "cells": [{"cell_type": t, "metadata": {}, "source": s.splitlines(keepends=True),
                 **({"outputs": [], "execution_count": None} if t == "code" else {}),
                 "id": f"c{i:02d}"} for i, (t, s) in enumerate(cells)]}
Path(__file__).with_name("xview3_train.ipynb").write_text(json.dumps(nb, indent=1) + "\n")
