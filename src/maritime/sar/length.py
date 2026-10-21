"""Baltic length head: the detector's length output is ~4x too short on Baltic Sentinel-1 scenes.
Fix: keep the whole detector frozen and retrain only its small length head on the saved 96 x 96 cut-outs
of AIS-matched ships, with the AIS-reported length as the label (free labels that grow every day).

Train and predict on the same cut-outs (the model sees less context on a cut-out than on a 512 px tile, so the raw
outputs differ by ~1.3x; doing both on cut-outs keeps them consistent).

    maritime sar length-train        (writes data/models/length_head_baltic.pt)
    maritime sar relength --hours N  (recompute length_est_m for scenes already processed, from their cut-outs)
"""

from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pandas as pd

HEAD_FILE = "length_head_baltic.pt"
PEAK_WIN = (10, 15)            # look for the heat peak in output cells 10..14 (chip centre = cell 12)


def chip_features(model, chips: np.ndarray, batch: int = 128):
    """Frozen decoder features the length head sees (n, 64, 3, 3) around the heat peak nearest the chip centre."""
    import torch

    from maritime.sar import detector as D
    feats, hook = [], {}
    h_ = model.u1.register_forward_hook(lambda mod, i, o: hook.__setitem__("d", o))
    a, b = PEAK_WIN
    try:
        with torch.no_grad():
            for i in range(0, len(chips), batch):
                x = torch.from_numpy(np.stack([D.make_input(c, ("VV", "VH")) for c in chips[i:i + batch]]))
                out = model(x.nan_to_num(0))
                h = out["heat"].sigmoid()[:, 0, a:b, a:b].reshape(len(x), -1)
                k = h.argmax(1)
                cy, cx = a + k // (b - a), a + k % (b - a)
                d = hook["d"]
                feats.append(torch.stack([d[j, :, cy[j] - 1:cy[j] + 2, cx[j] - 1:cx[j] + 2] for j in range(len(x))]))
    finally:
        h_.remove()
    return torch.cat(feats) if feats else torch.zeros((0, 64, 3, 3))


def train_head(model, feats, lengths_m, epochs: int = 300, seed: int = 0):
    """L1 on log(1 + m), starting from the detector's own length head."""
    import torch
    import torch.nn as nn
    torch.manual_seed(seed)
    head = copy.deepcopy(model.length)
    y = torch.log1p(torch.as_tensor(np.asarray(lengths_m, np.float32)))
    opt = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=1e-3)
    head.train()
    for _ in range(epochs):
        perm = torch.randperm(len(feats))
        for i in range(0, len(perm), 256):
            b = perm[i:i + 256]
            loss = nn.functional.l1_loss(head(feats[b])[:, 0, 1, 1], y[b])
            opt.zero_grad()
            loss.backward()
            opt.step()
    return head.eval()


def predict(head, feats) -> np.ndarray:
    import torch
    with torch.no_grad():
        return torch.expm1(head(feats)[:, 0, 1, 1]).clamp(min=0).numpy()


def load_head(model, path: Path):
    import torch
    head = copy.deepcopy(model.length)
    blob = torch.load(path, map_location="cpu", weights_only=True)
    head.load_state_dict(blob["state"])
    return head.eval()


def estimate(model, head, chips: np.ndarray) -> np.ndarray:
    return predict(head, chip_features(model, chips)) if len(chips) else np.zeros(0)


def metrics(est, true) -> dict:
    est, true = np.asarray(est, float), np.asarray(true, float)
    e = (est - true) / true
    big, small = true >= 45, true < 30
    from maritime.sar.scene_pipeline import DARK_MIN_LENGTH_M
    return {"n": int(len(true)), "median_abs_err_pct": round(float(np.median(np.abs(e)) * 100), 1),
            "within_25pct": round(float(np.mean(np.abs(e) <= 0.25) * 100), 1),
            "median_bias_pct": round(float(np.median(e) * 100), 1),
            "log_corr": round(float(np.corrcoef(np.log1p(est), np.log1p(true))[0, 1]), 3),
            f"ships_ge45m_below_{DARK_MIN_LENGTH_M:g}m_pct": round(float(np.mean(est[big] < DARK_MIN_LENGTH_M) * 100), 1),
            f"boats_lt30m_below_{DARK_MIN_LENGTH_M:g}m_pct":
                round(float(np.mean(est[small] < DARK_MIN_LENGTH_M) * 100), 1) if small.any() else None,
            "n_boats_lt30m": int(small.sum())}


def build_dataset(region_name: str, log=print) -> tuple[np.ndarray, pd.DataFrame]:
    """Cut-outs of AIS-matched detections with an AIS-reported length of 10-400 m."""
    import duckdb

    from maritime.paths import PROCESSED
    from maritime.sar.run import sar_dir
    base = sar_dir(region_name)
    rows, chips = [], []
    for f in sorted(base.glob("detections/date=*/*.parquet")):
        d = pd.read_parquet(f)
        cf = base / "chips" / f.parent.name / f"{f.stem}.npz"
        if "chip" not in d or not cf.exists():
            continue
        m = d[(d["status"] == "ais") & d["chip"].notna()]
        if m.empty:
            continue
        rows.append(m.assign(scene=f.stem))
        chips.append(np.load(cf)["chips"][m["chip"].astype(int).to_numpy()])
    if not rows:
        return np.zeros((0, 2, 96, 96), np.uint8), pd.DataFrame()
    det, X = pd.concat(rows, ignore_index=True), np.concatenate(chips)
    files = [str(p) for p in sorted((PROCESSED / region_name / "ais").glob("date=*/*.parquet"))]
    st = duckdb.sql(f"SELECT mmsi, arg_max(length, ts) FILTER (WHERE length IS NOT NULL) AS ais_length "
                    f"FROM read_parquet({files!r}, union_by_name=true) GROUP BY 1").df()
    det = det.merge(st, on="mmsi", how="left")
    ok = det["ais_length"].between(10, 400).to_numpy()
    log(f"{int(ok.sum())} AIS-matched cut-outs with an AIS length ({det.loc[ok, 'scene'].nunique()} scenes, "
        f"{det.loc[ok, 'mmsi'].nunique()} ships)")
    return X[ok], det[ok].reset_index(drop=True)


def train_and_save(region_name: str, folds: int = 5, log=print) -> dict:
    """Cross-validated by ship (no MMSI in both train and test), then fit on everything and save."""
    import datetime as dt

    import torch

    from maritime.sar.run import DEFAULT_MODELS, MODEL_DIR
    from maritime.sar.scene_pipeline import load_models
    X, d = build_dataset(region_name, log=log)
    if len(d) < 200:
        raise SystemExit(f"only {len(d)} labelled cut-outs; collect more scenes first")
    model = load_models([MODEL_DIR / DEFAULT_MODELS[0]])[0]
    F = chip_features(model, X)
    rng = np.random.default_rng(0)
    ships = d["mmsi"].unique()
    rng.shuffle(ships)
    fold = d["mmsi"].map({s: i % folds for i, s in enumerate(ships)}).to_numpy()
    est = np.zeros(len(d))
    for k in range(folds):
        tr, te = fold != k, fold == k
        est[te] = predict(train_head(model, F[tr], d["ais_length"].to_numpy()[tr]), F[te])
    cv = metrics(est, d["ais_length"])
    log(f"cross-validated by ship: {cv}")
    head = train_head(model, F, d["ais_length"].to_numpy())
    out = MODEL_DIR / HEAD_FILE
    torch.save({"state": head.state_dict(), "detector": DEFAULT_MODELS[0], "n": int(len(d)),
                "trained": dt.date.today().isoformat()}, out)
    log(f"saved {out}")
    return cv


def relength(region_name: str, hours: float = 24 * 30, force: bool = False, log=print) -> int:
    """Add/refresh length_est_m in detection files that have cut-outs. Returns the number of files updated."""
    from maritime.sar.run import DEFAULT_MODELS, MODEL_DIR, sar_dir
    from maritime.sar.scene_pipeline import load_models
    hp = MODEL_DIR / HEAD_FILE
    if not hp.exists():
        raise SystemExit(f"{hp} missing: run `maritime sar length-train`")
    model = load_models([MODEL_DIR / DEFAULT_MODELS[0]])[0]
    head = load_head(model, hp)
    base = sar_dir(region_name)
    since = (pd.Timestamp.now("UTC").tz_localize(None) - pd.Timedelta(hours=hours)).date()
    n = 0
    for f in sorted(base.glob("detections/date=*/*.parquet")):
        if f.parent.name[5:] < str(since):
            continue
        d = pd.read_parquet(f)
        cf = base / "chips" / f.parent.name / f"{f.stem}.npz"
        if d.empty or "chip" not in d or not cf.exists() or ("length_est_m" in d and not force):
            continue
        chips = np.load(cf)["chips"]
        d["length_est_m"] = np.round(estimate(model, head, chips[d["chip"].astype(int).to_numpy()]), 1)
        d.to_parquet(f, index=False)
        n += 1
    log(f"{n} scenes updated with length_est_m")
    return n
