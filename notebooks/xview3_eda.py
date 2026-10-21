"""xView3 image EDA (run on Kaggle with the tiles dataset attached, CPU is enough).

Questions:
  Q1 inventory: tiles, objects, no-data per scene and split
  Q2 sea clutter: background dB levels per scene and channel (do scenes need normalising?)
  Q3 object contrast: how much brighter are labelled objects than the sea, by type/size/confidence;
     which channel (VV or VH) separates objects better
  Q4 label placement: distance from each label to the brightest nearby pixel
  Q5 unlabelled objects: bright CFAR blobs in "empty" tiles and unmatched blobs in positive tiles,
     train scenes (AIS-only labels) vs validation scenes (AIS + manual labels)
Writes figures and eda_summary.json to OUT; prints a compact text summary.
"""
import glob
import json
import os

import numpy as np
import pandas as pd
from scipy import ndimage

ROOT = os.environ.get("XV3_DATA", "/kaggle/input")
OUT = os.environ.get("XV3_OUT", "/kaggle/working")
npz_paths = sorted(glob.glob(ROOT + "/**/*.npz", recursive=True))
assert npz_paths, "add the maritime-xview3-tiles dataset as input"
DATA = os.path.dirname(npz_paths[0])
split_of = (pd.read_csv(f"{DATA}/scenes.csv").set_index("scene_id").split.to_dict()
            if os.path.exists(f"{DATA}/scenes.csv") else {})
DB_MIN, DB_MAX = -50.0, 10.0


def to_db(u8):
    return np.where(u8 > 0, (u8.astype(np.float32) - 1) / 254 * (DB_MAX - DB_MIN) + DB_MIN, np.nan)


def cfar(db, guard=4, train=16, k=5.0):
    valid = np.isfinite(db)
    x = np.where(valid, db, np.nanmedian(db))
    big, small = 2 * (guard + train) + 1, 2 * guard + 1
    def box(a, s):
        return ndimage.uniform_filter(a, s, mode="nearest") * s * s
    n = big * big - small * small
    mean = (box(x, big) - box(x, small)) / n
    var = (box(x * x, big) - box(x * x, small)) / n - mean ** 2
    det = valid & (x > mean + k * np.sqrt(np.maximum(var, 1e-6)))
    lab, nl = ndimage.label(det)
    if nl == 0:
        return np.zeros((0, 2))
    idx = np.arange(1, nl + 1)
    keep = ndimage.sum(det, lab, idx) >= 2
    return np.array(ndimage.center_of_mass(det, lab, idx)).reshape(-1, 2)[keep]


rows_scene, rows_obj, rows_tile = [], [], []
sea_hist = {"VV": np.zeros(120), "VH": np.zeros(120)}
bins = np.linspace(DB_MIN, DB_MAX, 121)
rng = np.random.default_rng(0)
gallery = []

for p in npz_paths:
    sid = os.path.basename(p)[:-4]
    split = split_of.get(sid, "unknown")
    z = np.load(p, allow_pickle=False)
    imgs, objs = z["images"], z["objects"]
    for t in range(len(imgs)):
        vv, vh = to_db(imgs[t, 0]), to_db(imgs[t, 1])
        o = objs[objs["tile"] == t]
        nodata = float(np.isnan(vv).mean())
        # sea background: valid pixels at least 10 px from any label
        mask = np.isfinite(vv)
        if len(o):
            m = np.zeros(vv.shape, bool)
            m[o["row"].clip(0, 511), o["col"].clip(0, 511)] = True
            mask &= ~ndimage.binary_dilation(m, iterations=10)
        bg_vv, bg_vh = np.nanmedian(np.where(mask, vv, np.nan)), np.nanmedian(np.where(mask, vh, np.nan))
        if t % 4 == 0:
            sea_hist["VV"] += np.histogram(vv[mask], bins)[0]
            sea_hist["VH"] += np.histogram(vh[mask], bins)[0]
        blobs = cfar(vv)
        if len(blobs) and len(o):
            d = np.hypot(blobs[:, None, 0] - o["row"][None], blobs[:, None, 1] - o["col"][None]).min(1)
            unmatched = int((d > 20).sum())
        else:
            unmatched = len(blobs)
        rows_tile.append(dict(scene=sid, split=split, tile=t, n_obj=len(o), nodata=nodata,
                              bg_vv=bg_vv, bg_vh=bg_vh, cfar_blobs=len(blobs), cfar_unmatched=unmatched))
        for ob in o:
            r, c = int(ob["row"]), int(ob["col"])
            r0, r1, c0, c1 = max(r - 5, 0), min(r + 6, 512), max(c - 5, 0), min(c + 6, 512)
            win = vv[r0:r1, c0:c1]
            if np.all(np.isnan(win)):
                rows_obj.append(dict(scene=sid, split=split, in_nodata=True)); continue
            pr, pc = np.unravel_index(np.nanargmax(win), win.shape)
            rows_obj.append(dict(
                scene=sid, split=split, in_nodata=False,
                peak_vv=float(np.nanmax(win)), peak_vh=float(np.nanmax(vh[r0:r1, c0:c1])),
                con_vv=float(np.nanmax(win) - bg_vv), con_vh=float(np.nanmax(vh[r0:r1, c0:c1]) - bg_vh),
                offset_px=float(np.hypot(r0 + pr - r, c0 + pc - c)),
                is_vessel=float(ob["is_vessel"]), is_fishing=float(ob["is_fishing"]),
                length_m=float(ob["length_m"]), confidence=str(ob["confidence"]),
                shore_km=float(ob["shore_km"])))
            if len(gallery) < 400 and 16 <= r < 496 and 16 <= c < 496:
                gallery.append((rows_obj[-1], imgs[t, :, r - 16:r + 16, c - 16:c + 16]))
    rows_scene.append(dict(scene=sid, split=split, tiles=len(imgs), objects=len(objs),
                           pos_tiles=int(len(np.unique(objs["tile"])))))

S, O, T = pd.DataFrame(rows_scene), pd.DataFrame(rows_obj), pd.DataFrame(rows_tile)
O.loc[O.get("shore_km", pd.Series(dtype=float)) >= 9999, "shore_km"] = np.nan
Ov = O[~O.in_nodata].copy()
Ov["type"] = np.select([Ov.is_vessel == 1, Ov.is_vessel == 0], ["vessel", "infrastructure"], "unknown")
Ov["size"] = pd.cut(Ov.length_m, [0, 25, 60, 150, 1000], labels=["<25m", "25-60m", "60-150m", ">150m"])
summary = {}

print("Q1 INVENTORY")
print(S.groupby("split")[["tiles", "pos_tiles", "objects"]].sum().to_string())
print(f"objects inside no-data: {int(O.in_nodata.sum())} of {len(O)}")
print("tile no-data fraction: median", round(T.nodata.median(), 3), " tiles >25% no-data:", int((T.nodata > .25).sum()))

print("\nQ2 SEA CLUTTER (median background dB per scene)")
q2 = T.groupby(["split", "scene"])[["bg_vv", "bg_vh"]].median().round(1)
print(q2.to_string())
summary["bg_spread_vv_db"] = float(q2.bg_vv.max() - q2.bg_vv.min())
summary["bg_spread_vh_db"] = float(q2.bg_vh.max() - q2.bg_vh.min())

print("\nQ3 OBJECT CONTRAST (peak minus local sea, dB; median [p10])")
def med(g):
    return pd.Series({"n": len(g), "VV": f"{g.con_vv.median():.1f} [{g.con_vv.quantile(.1):.1f}]",
                      "VH": f"{g.con_vh.median():.1f} [{g.con_vh.quantile(.1):.1f}]",
                      "weak<3dB VV %": round((g.con_vv < 3).mean() * 100, 1)})
for col in ["split", "type", "size", "confidence"]:
    print(Ov.groupby(col, observed=True).apply(med, include_groups=False).to_string(), "\n")
# separability: contrast of objects vs random sea pixels' deviation, via simple AUC on peak-minus-bg
summary["contrast_median_vv"] = float(Ov.con_vv.median())
summary["contrast_median_vh"] = float(Ov.con_vh.median())
summary["weak_objects_pct_vv"] = float((Ov.con_vv < 3).mean() * 100)
summary["weak_objects_pct_vh"] = float((Ov.con_vh < 3).mean() * 100)

print("Q4 LABEL PLACEMENT (px from label to brightest pixel within 5 px; 1 px = 10 m)")
print(Ov.offset_px.describe(percentiles=[.5, .75, .9]).round(2).to_string())
summary["offset_median_px"] = float(Ov.offset_px.median())

print("\nQ5 UNLABELLED BRIGHT OBJECTS (CFAR blobs on VV, k=5)")
T["kind"] = np.where(T.n_obj > 0, "positive tile", "empty tile")
q5 = T.groupby(["split", "kind"]).agg(tiles=("tile", "size"), blobs_per_tile=("cfar_blobs", "mean"),
                                       unmatched_per_tile=("cfar_unmatched", "mean"),
                                       labels_per_tile=("n_obj", "mean")).round(2)
print(q5.to_string())
summary["q5"] = {f"{a}|{b}": v for (a, b), v in q5["unmatched_per_tile"].items()}

# figures
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig, ax = plt.subplots(1, 3, figsize=(16, 4))
c = (bins[:-1] + bins[1:]) / 2
for ch in ["VV", "VH"]:
    ax[0].plot(c, sea_hist[ch] / sea_hist[ch].sum(), label=f"sea {ch}")
ax[0].hist(Ov.peak_vv, bins=bins, density=True, histtype="step", label="object peak VV")
ax[0].hist(Ov.peak_vh, bins=bins, density=True, histtype="step", label="object peak VH")
ax[0].set_xlabel("dB"); ax[0].legend(); ax[0].set_title("Sea vs object brightness")
for t_ in ["vessel", "infrastructure", "unknown"]:
    g = Ov[Ov.type == t_]
    ax[1].scatter(g.con_vv, g.con_vh, s=4, alpha=.4, label=t_)
ax[1].axline((0, 0), slope=1, c="grey", lw=.5); ax[1].set_xlabel("VV contrast dB"); ax[1].set_ylabel("VH contrast dB")
ax[1].legend(); ax[1].set_title("Contrast per object")
g = Ov.dropna(subset=["length_m"])
ax[2].scatter(g.length_m, g.con_vv, s=4, alpha=.4); ax[2].set_xscale("log")
ax[2].set_xlabel("length m"); ax[2].set_ylabel("VV contrast dB"); ax[2].set_title("Bigger ships are brighter?")
plt.tight_layout(); plt.savefig(f"{OUT}/eda_contrast.png", dpi=80)

cats = [("vessel <25m", lambda r: r["is_vessel"] == 1 and r["length_m"] < 25),
        ("vessel >100m", lambda r: r["is_vessel"] == 1 and r["length_m"] > 100),
        ("fishing", lambda r: r["is_fishing"] == 1),
        ("infrastructure", lambda r: r["is_vessel"] == 0),
        ("weak (<3 dB)", lambda r: r["con_vv"] < 3)]
fig, ax = plt.subplots(len(cats), 8, figsize=(12, 1.6 * len(cats)))
for i, (name, f) in enumerate(cats):
    picks = [g for g in gallery if f(g[0])][:8]
    for j in range(8):
        a = ax[i, j]; a.axis("off")
        if j < len(picks):
            a.imshow(picks[j][1][1], cmap="gray", vmin=100, vmax=230)
            a.set_title(f"{picks[j][0]['con_vv']:.0f}dB", fontsize=7)
    ax[i, 0].text(-40, 16, name, fontsize=8, ha="right", va="center")
plt.tight_layout(); plt.savefig(f"{OUT}/eda_gallery.png", dpi=80)

with open(f"{OUT}/eda_summary.json", "w") as f:
    json.dump(summary, f, indent=1, default=float)
print("\nSUMMARY", json.dumps(summary, default=float))
