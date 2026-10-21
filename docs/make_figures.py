"""Figures for the README, rebuilt from the project's own data:

    python docs/make_figures.py length     # length head: raw vs fine-tuned, cross-validated by ship
    python docs/make_figures.py coverage   # second AIS feed: radar detections matched to AIS, before / after
    python docs/make_figures.py chips      # radar cut-outs of each kind of target
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

OUT = Path(__file__).parent
SURFACE, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8984", "#e6e5e1"
S1, S2, S3 = "#2a78d6", "#eb6834", "#1baf7a"   # reference palette slots 1-3 (validated all-pairs; S3 < 3:1
                                                 # contrast, so charts using it carry value labels)
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": MUTED,
                     "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
                     "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE})


def _clean(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def length():
    from maritime.sar import length as L
    from maritime.sar.run import DEFAULT_MODELS, MODEL_DIR
    from maritime.sar.scene_pipeline import load_models
    X, d = L.build_dataset("baltic", log=lambda *a: None)
    model = load_models([MODEL_DIR / DEFAULT_MODELS[0]])[0]
    F = L.chip_features(model, X)
    rng = np.random.default_rng(0)
    ships = d["mmsi"].unique()
    rng.shuffle(ships)
    fold = d["mmsi"].map({s: i % 5 for i, s in enumerate(ships)}).to_numpy()
    est = np.zeros(len(d))
    y = d["ais_length"].to_numpy()
    for k in range(5):
        tr, te = fold != k, fold == k
        est[te] = L.predict(L.train_head(model, F[tr], y[tr]), F[te])
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.2), sharex=True, sharey=True)
    for ax, (title, v) in zip(axes, [("Before: the detector's own length output", d["length_m"].to_numpy()),
                                     ("After: retrained length head (ships held out)", est)], strict=True):
        _clean(ax)
        ax.scatter(y, np.clip(v, 5, None), s=9, color=S1, alpha=0.35, linewidths=0)
        ax.plot([10, 420], [10, 420], color=INK2, linewidth=1.2, linestyle="--")
        ax.text(330, 250, "perfect", color=INK2, fontsize=9, rotation=38)
        med = np.median(np.abs(v - y) / y) * 100
        ax.set_title(f"{title}\nmedian error {med:.0f}%", fontsize=10, loc="left", color=INK)
        ax.set_xscale("log"), ax.set_yscale("log")
        ax.set_xlim(10, 420), ax.set_ylim(5, 420)
        ax.set_xlabel("length reported in AIS (m)")
    axes[0].set_ylabel("length estimated from radar (m)")
    fig.suptitle(f"{len(d):,} radar cut-outs of {d['mmsi'].nunique():,} ships matched to AIS",
                 x=0.01, ha="left", fontsize=11, color=INK)
    fig.tight_layout()
    fig.savefig(OUT / "fig_length.png", dpi=160)
    print("fig_length.png", len(d))


def coverage():
    # radar detections of ship size matched to AIS (vs "no coverage"), before / after the Digitraffic feed
    # (scenes before 2026-10-15 18:00 UTC, when the second feed started, vs after; measured 2026-10-18)
    areas = ["Gulf of Finland", "Gulf of Bothnia", "Gotland to Aland", "South Baltic\n(control, no new feed)"]
    before, after = [49.0, 17.9, 33.3, 73.6], [76.7, 67.3, 61.0, 69.7]
    fig, ax = plt.subplots(figsize=(8.5, 3.6))
    _clean(ax)
    ax.grid(axis="y", visible=False)
    yy = np.arange(len(areas))[::-1]
    h = 0.36
    ax.barh(yy + h / 2, before, height=h - 0.04, color=S1, label="one free AIS feed")
    ax.barh(yy - h / 2, after, height=h - 0.04, color=S2, label="after adding Digitraffic")
    for y_, b, a in zip(yy, before, after, strict=True):
        ax.text(b + 1, y_ + h / 2, f"{b:.0f}%", va="center", fontsize=9, color=INK2)
        ax.text(a + 1, y_ - h / 2, f"{a:.0f}%", va="center", fontsize=9, color=INK)
    ax.set_yticks(yy, areas)
    ax.set_xlim(0, 100)
    ax.set_xlabel("ship-sized radar detections matched to an AIS ship (%)")
    ax.set_title("A second AIS feed: radar ships that could finally be checked", loc="left", fontsize=11, pad=26)
    ax.legend(frameon=False, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2, fontsize=9, borderaxespad=0.2)
    fig.tight_layout()
    fig.savefig(OUT / "fig_coverage.png", dpi=160)
    print("fig_coverage.png")


def chips():
    from maritime.sar import scene_pipeline as P
    from maritime.sar.run import sar_dir
    from maritime.web.app import apply_evidence
    base = sar_dir("baltic")
    frames = []
    for f in sorted(base.glob("detections/date=*/*.parquet")):
        d = pd.read_parquet(f)
        if "chip" in d and (base / "chips" / f.parent.name / f"{f.stem}.npz").exists():
            frames.append(d.assign(_file=str(base / "chips" / f.parent.name / f"{f.stem}.npz")))
    det = pd.concat(frames, ignore_index=True)
    det = P.refine_status(det, P.static_mask(det))
    det = apply_evidence(det, base / "evidence" / "latest.parquet")
    kinds = [("ais", "Matched to its AIS"), ("ais_gap", "AIS briefly missed\n(explained)"),
             ("dark", "Dark candidate\n(no AIS fits)"), ("static", "Fixed structure\n(same spot, two passes)"),
             ("small", "Small craft\n(no AIS required)")]
    rng = np.random.default_rng(3)
    fig, axes = plt.subplots(3, len(kinds), figsize=(10, 6.6))
    for j, (k, label) in enumerate(kinds):
        pool = det[det["status"] == k]
        pool = pool[pool["score"] >= 0.6]
        if k == "ais":
            pool = pool[pool["length_rough_m"] >= 100]
        pick = pool.iloc[rng.choice(len(pool), size=min(3, len(pool)), replace=False)] if len(pool) else pool
        for i in range(3):
            ax = axes[i, j]
            ax.set_xticks([]), ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(False)
            if i >= len(pick):
                continue
            r = pick.iloc[i]
            c = np.load(r["_file"])["chips"][int(r["chip"])][0].astype(float)    # VV
            ax.imshow(c[16:80, 16:80], cmap="gray", vmin=np.percentile(c, 2), vmax=np.percentile(c, 99.7))
            if k != "static" and r.get("length_rough_m") == r.get("length_rough_m") and r.get("length_rough_m"):
                ax.text(2, 61, f"~{int(round(r['length_rough_m'], -1))} m", color="#ffffff", fontsize=8)
        axes[0, j].set_title(label, fontsize=9, color=INK)
    fig.suptitle("What the detector sees: 640 m x 640 m radar cut-outs (Sentinel-1 VV), Baltic, October 2026",
                 x=0.01, ha="left", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / "fig_chips.png", dpi=160)
    print("fig_chips.png", det["status"].value_counts().to_dict())


def _xview3_sweep():
    from maritime.sar import detector as D
    root = Path(__file__).parents[1] / "data" / "xview3"
    held = open(root / "heldout_scenes.txt").read().split()
    det = pd.read_csv(root / "scene_eval" / "detections_ensemble.csv.gz")
    lab = pd.read_csv(root / "labels" / "validation.csv")
    lab = lab[lab["scene_id"].isin(held)]
    gts = {}
    for sid, g in lab.groupby("scene_id"):
        a = np.zeros(len(g), dtype=[("row", "f4"), ("col", "f4"), ("confidence", "U6"), ("is_vessel", "f4"),
                                    ("is_fishing", "f4"), ("length_m", "f4")])
        a["row"], a["col"] = g["detect_scene_row"], g["detect_scene_column"]
        a["confidence"] = g["confidence"]
        a["is_vessel"] = g["is_vessel"].map({True: 1.0, False: 0.0}).astype(float)
        a["is_fishing"] = g["is_fishing"].map({True: 1.0, False: 0.0}).astype(float)
        a["length_m"] = g["vessel_length_m"]
        gts[sid] = a
    rows = []
    for t in np.round(np.arange(0.2, 0.91, 0.025), 3):
        preds = {}
        for sid, g in det[det["score"] >= t].groupby("scene_id"):
            a = np.zeros(len(g), dtype=[("row", "f4"), ("col", "f4"), ("is_vessel", "f4"), ("is_fishing", "f4"),
                                        ("length_m", "f4")])
            a["row"], a["col"] = g["row"], g["col"]
            a["is_vessel"], a["is_fishing"], a["length_m"] = g["p_vessel"], g["p_fishing"], g["length_m"]
            preds[sid] = a
        r = D.score(preds, gts)
        rows.append({"threshold": t, **{k: r[k] for k in ("precision", "recall", "f1_detect", "tp", "fp", "fn")}})
    return pd.DataFrame(rows)


def xview3():
    sw = _xview3_sweep()
    sw.to_csv(OUT / "xview3_threshold_sweep.csv", index=False)
    op = sw.loc[(sw["threshold"] - 0.45).abs().idxmin()]
    print(sw.loc[sw["threshold"].isin([0.4, 0.45, 0.5])].round(3).to_string(index=False))
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4.3))
    _clean(a1)
    rr = np.linspace(0.01, 1, 200)
    for f, lab in ((0.43, "reference, F1 0.43"), (0.77, "1st place, F1 0.77")):
        pp = f * rr / (2 * rr - f)
        ok = (pp > 0) & (pp <= 1)
        a1.plot(rr[ok], pp[ok], color=MUTED, linewidth=1, linestyle=":")
        r_lab = 0.94
        a1.text(r_lab, f * r_lab / (2 * r_lab - f) + 0.015, lab, color=INK2, fontsize=8, ha="right", va="bottom")
    a1.plot(sw["recall"], sw["precision"], color=S1, linewidth=2)
    a1.scatter([op["recall"]], [op["precision"]], color=S1, s=60, zorder=3, edgecolors=SURFACE, linewidths=1.5)
    a1.annotate(f"chosen threshold 0.45\nF1 {op['f1_detect']:.3f} (P {op['precision']:.2f}, R {op['recall']:.2f})",
                (op["recall"], op["precision"]), xytext=(-150, -60), textcoords="offset points", fontsize=8,
                color=INK, arrowprops={"arrowstyle": "-", "color": INK2, "lw": 0.8})
    a1.set_xlim(0.4, 0.95), a1.set_ylim(0.2, 1.08)
    a1.set_xlabel("recall"), a1.set_ylabel("precision")
    a1.set_title("Ship detection on 10 held-out xView3 scenes\n(threshold swept 0.2 to 0.9)", loc="left",
                 fontsize=10)
    fu = pd.read_csv(Path(__file__).parents[1] / "data" / "xview3" / "fusion_compare_v6_v7.csv")
    fu = fu[fu["model"] == "V7"]
    _clean(a2)
    a2.plot(fu["thr"], fu["P"], color=S1, linewidth=2, marker="o", markersize=5, label="precision of dark flags")
    a2.plot(fu["thr"], fu["R"], color=S2, linewidth=2, marker="o", markersize=5, label="share of dark vessels found")
    a2.axvspan(0.6, 0.86, color=S1, alpha=0.07)
    a2.text(0.605, 0.04, "map tier \"high\"", color=INK2, fontsize=8)
    a2.axvspan(0.45, 0.6, color=MUTED, alpha=0.08)
    a2.text(0.455, 0.04, "\"possible\"", color=INK2, fontsize=8)
    a2.set_ylim(0, 1.02), a2.set_xlim(0.4, 0.86)
    a2.set_xlabel("detector score threshold")
    a2.set_title("Dark-vessel flags (AIS hidden, then matched)\nsame held-out scenes", loc="left", fontsize=10)
    a2.legend(frameon=False, fontsize=8, loc="center right")
    fig.tight_layout()
    fig.savefig(OUT / "fig_xview3_pr.png", dpi=160)
    print("fig_xview3_pr.png")


def compare():
    metrics = ["Detection F1", "Close-to-shore F1", "Vessel F1", "Fishing F1", "Length score", "Aggregate"]
    rows = {  # our 10 held-out validation scenes; xView3 paper (Table 1, holdout set)
        "This project": [0.773, 0.446, 0.933, 0.773, 0.665, 0.590],
        "xView3 1st place (BloodAxe)": [0.7702, 0.5310, 0.9392, 0.8425, 0.6971, 0.6177],
        "xView3 reference model": [0.4302, 0.1293, 0.6891, 0.3946, 0.0, 0.1904],
    }
    cols = [S1, S2, S3]
    fig, ax = plt.subplots(figsize=(10, 3.9))
    _clean(ax)
    ax.grid(axis="x", visible=False)
    x = np.arange(len(metrics))
    w = 0.26
    for i, ((name, v), c) in enumerate(zip(rows.items(), cols, strict=True)):
        b = ax.bar(x + (i - 1) * w, v, width=w - 0.03, color=c, label=name)
        for xi, vi in zip(x + (i - 1) * w, v, strict=True):
            ax.text(xi, vi + 0.015, f"{vi:.2f}", ha="center", fontsize=7.5, color=INK2)
        del b
    ax.set_xticks(x, metrics)
    ax.set_ylim(0, 1.05)
    ax.set_title("xView3 metrics: published reference, competition winner and this project", loc="left",
                 fontsize=11, pad=24)
    ax.legend(frameon=False, ncol=3, fontsize=9, loc="lower left", bbox_to_anchor=(0, 1.0), borderaxespad=0.2)
    fig.tight_layout()
    fig.savefig(OUT / "fig_xview3_compare.png", dpi=160)
    print("fig_xview3_compare.png")


def baltic():
    """Detector recall against AIS (first 73 Baltic scenes) and evidence verdicts for dark candidates (72 h)."""
    rep = Path(__file__).parents[1] / "data" / "processed" / "baltic" / "sar" / "report"
    r = pd.read_csv(rep / "recall_20261014_1919.csv")
    r = r[r["transceiver_class"] == "A"]
    bands = ["<30 m", "30-60 m", "60-120 m", ">=120 m"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.5, 4.0), gridspec_kw={"width_ratios": [1.15, 1]})
    _clean(a1)
    a1.grid(axis="x", visible=False)
    x = np.arange(len(bands))
    w = 0.38
    for i, (mv, lab, c) in enumerate(((True, "moving", S1), (False, "stationary (anchored, moored)", S2))):
        vals, ns = [], []
        for b in bands:
            g = r[(r["length_band"] == b) & (r["moving"] == mv)]
            vals.append(g["found"].mean() * 100 if len(g) else 0)
            ns.append(len(g))
        a1.bar(x + (i - 0.5) * w, vals, width=w - 0.03, color=c, label=lab)
        for xi, v, n in zip(x + (i - 0.5) * w, vals, ns, strict=True):
            a1.text(xi, v + 1.5, f"{v:.0f}%", ha="center", fontsize=8, color=INK)
            a1.text(xi, -7, f"{n:,}", ha="center", fontsize=6.5, color=MUTED)
    a1.set_xticks(x, bands)
    a1.set_ylim(-10, 100)
    a1.set_yticks([0, 20, 40, 60, 80, 100])
    a1.set_ylabel("AIS ships found by the detector (%)")
    a1.set_xlabel("length reported in AIS (class A); grey numbers = AIS ships in the images")
    a1.legend(frameon=False, fontsize=8, loc="upper left")
    a1.set_title("Recall against AIS, 73 Baltic scenes", loc="left", fontsize=10)
    e = pd.read_csv(rep / "evidence_20261019_2046.csv")
    order = [("explained", "explained: ordinary AIS reception gap"), ("usual_gap", "usual gap for that area"),
             ("low_reception", "low reception at the time"), ("weak_link", "weak link (not evidence)"),
             ("unexplained", "unexplained: no AIS ship fits"), ("went_silent", "went silent (all checks passed)")]
    counts = e["verdict"].value_counts()
    _clean(a2)
    a2.grid(axis="y", visible=False)
    labels = [lab for _, lab in order]
    vals = [int(counts.get(k, 0)) for k, _ in order]
    yy = np.arange(len(order))[::-1]
    a2.barh(yy, vals, color=S1, height=0.6)
    for y_, v in zip(yy, vals, strict=True):
        a2.text(v + 0.6, y_, str(v), va="center", fontsize=9, color=INK)
    a2.set_yticks(yy, labels, fontsize=8.5)
    a2.set_xlabel("dark candidates, 72 h to 2026-10-19")
    a2.set_title(f"Evidence verdicts, {len(e)} dark candidates", loc="left", fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / "fig_baltic.png", dpi=160)
    print("fig_baltic.png")


def risk():
    """Risk model: share of listed vessels among the top k, cross-validated, vs two baselines."""
    from sklearn.ensemble import IsolationForest

    from maritime import risk as RK
    k = pd.read_parquet(Path(__file__).parents[1] / "data" / "processed" / "baltic" / "risk" / "latest.parquet")
    y = k["y"].to_numpy()
    rule = (k["tanker"] * 2 + (k["russian_port_h"] > 0) * 2 + np.log1p(k["russian_port_h"])).to_numpy()
    iso = -IsolationForest(random_state=0).fit(RK.design(k)).score_samples(RK.design(k))
    ks = np.arange(10, 301, 5)
    fig, ax = plt.subplots(figsize=(8.5, 3.8))
    _clean(ax)
    for name, sc_, c in (("logistic regression (cross-validated)", k["risk_cv"].to_numpy(), S1),
                         ("rule: tanker that called at a Russian port", rule, S2),
                         ("Isolation Forest (no labels)", iso, S3)):
        o = np.argsort(-sc_)
        ax.plot(ks, [y[o[:kk]].mean() * 100 for kk in ks], color=c, linewidth=2, label=name)
    ax.axhline(y.mean() * 100, color=INK2, linewidth=1, linestyle="--")
    ax.text(300, y.mean() * 100 + 1.5, f"base rate {y.mean() * 100:.1f}%", ha="right", fontsize=8, color=INK2)
    ax.set_xlabel("top k vessels by score")
    ax.set_ylabel("listed vessels in the top k (%)")
    ax.set_ylim(0, 80)
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    ax.set_title(f"Risk ranking: {len(k):,} vessels over 7 days, {int(y.sum())} on sanctions / shadow-fleet lists",
                 loc="left", fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / "fig_risk.png", dpi=160)
    print("fig_risk.png")


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parents[1] / "src"))
    {"length": length, "coverage": coverage, "chips": chips, "xview3": xview3, "compare": compare,
     "baltic": baltic, "risk": risk}[sys.argv[1]]()
