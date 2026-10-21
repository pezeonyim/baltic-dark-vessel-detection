"""Test SAR+AIS fusion on xView3, where we know the answer.

xView3 validation labels record where each object came from: `ais` / `ais/manual` objects had an AIS
broadcast at the image time; `manual` objects were found by analysts in the image with no AIS match.
So we can play the product end to end on held-out scenes:
  - the AIS feed = the positions of the AIS-sourced labels (optionally jittered, to mimic the error of
    estimating an AIS position at the image time);
  - the detector output = the ensemble detections saved by the scene-eval notebook;
  - fusion flags dark-vessel candidates (fusion.dark_candidates);
  - truth: a "dark vessel" is a manual, non-LOW label with is_vessel True. Manual non-vessels are
    unlit infrastructure; manual LOW labels are ignored (as in the xView3 metric).
This is an upper bound for the real thing: real AIS needs time interpolation and has gaps and spoofing.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from maritime.fusion import dark_candidates
from maritime.sar.detector import match

PIX_M = 10.0
MATCH_PX = 20.0   # 200 m, the xView3 matching distance


def _xy(rows, cols):
    return np.column_stack([np.asarray(cols, float) * PIX_M, -np.asarray(rows, float) * PIX_M])


def label_kind(lab: pd.DataFrame) -> np.ndarray:
    """ais | dark | infra | low for each label."""
    src = lab["source"].astype(str)
    is_ais = src.str.contains("ais")
    low = lab["confidence"].astype(str) == "LOW"
    vessel = lab["is_vessel"].astype(str).str.lower() == "true"
    return np.select([is_ais, low, vessel], ["ais", "low", "dark"], default="infra")


def simulate_scene(dets: pd.DataFrame, lab: pd.DataFrame, score_min=0.45, p_vessel_min=0.5,
                   gate_m=200.0, ais_jitter_m=0.0, rng=None):
    """dets: row, col, score, p_vessel for one scene. lab: validation labels for the same scene.
    Returns (candidates frame with an `outcome` column, labels frame with a `fate` column)."""
    rng = rng or np.random.default_rng(0)
    lab = lab.reset_index(drop=True).copy()
    lab["kind"] = label_kind(lab)
    ais_lab = lab[lab["kind"] == "ais"]
    ais_xy = _xy(ais_lab["detect_scene_row"], ais_lab["detect_scene_column"])
    if ais_jitter_m > 0 and len(ais_xy):
        ais_xy = ais_xy + rng.normal(0, ais_jitter_m, ais_xy.shape)
    ais_pos = pd.DataFrame({"x": ais_xy[:, 0], "y": ais_xy[:, 1], "unc_m": 0.0})

    d = dets.copy()
    xy = _xy(d["row"], d["col"])
    d["x"], d["y"] = xy[:, 0], xy[:, 1]
    d = dark_candidates(d, ais_pos, score_min=score_min, p_vessel_min=p_vessel_min, base_gate_m=gate_m)

    # which label (if any) each detection really is: the xView3 matching, all labels at once
    pairs = match(np.column_stack([d["row"], d["col"]]) if len(d) else np.zeros((0, 2)),
                  np.column_stack([lab["detect_scene_row"], lab["detect_scene_column"]]) if len(lab)
                  else np.zeros((0, 2)), MATCH_PX)
    det_lab = np.full(len(d), -1)
    for a, b in pairs:
        det_lab[a] = b
    truth = np.where(det_lab >= 0, lab["kind"].to_numpy()[np.maximum(det_lab, 0)], "none")
    d["truth"] = truth
    d["shore_km"] = np.where(det_lab >= 0, lab["distance_from_shore_km"].to_numpy()[np.maximum(det_lab, 0)],
                             np.nan)

    # candidate outcomes
    outcome = {"dark": "tp", "low": "ignored", "infra": "fp_infra", "ais": "fp_ais_missed", "none": "fp_clutter"}
    d["outcome"] = np.where(d["status"] == "dark", pd.Series(truth).map(outcome).to_numpy(), "")

    # fate of every true dark vessel
    lab["fate"] = ""
    status_by_lab = {b: d["status"].iat[a] for a, b in pairs}
    fate_of = {"dark": "found", "ais": "absorbed_by_ais", "non_vessel": "filtered_as_non_vessel"}
    for i in np.where(lab["kind"] == "dark")[0]:
        lab.at[i, "fate"] = fate_of[status_by_lab[i]] if i in status_by_lab else "not_detected"
    return d, lab


def summarise(cands: pd.DataFrame, labs: pd.DataFrame) -> dict:
    c = cands[cands["status"] == "dark"]
    oc = c["outcome"].value_counts()
    tp = int(oc.get("tp", 0))
    fp = int(sum(oc.get(k, 0) for k in ("fp_infra", "fp_ais_missed", "fp_clutter")))
    dark = labs[labs["kind"] == "dark"]
    near = dark["distance_from_shore_km"] < 2.0
    found = dark["fate"] == "found"
    fates = dark["fate"].value_counts()
    return {
        "candidates": len(c), "tp": tp, "fp": fp, "ignored": int(oc.get("ignored", 0)),
        "precision": tp / max(tp + fp, 1),
        "dark_vessels": len(dark), "recall": float(found.mean()) if len(dark) else float("nan"),
        "recall_near_shore": float(found[near].mean()) if near.any() else float("nan"),
        "recall_offshore": float(found[~near].mean()) if (~near).any() else float("nan"),
        "fp_infra": int(oc.get("fp_infra", 0)), "fp_ais_missed": int(oc.get("fp_ais_missed", 0)),
        "fp_clutter": int(oc.get("fp_clutter", 0)),
        "miss_not_detected": int(fates.get("not_detected", 0)),
        "miss_absorbed_by_ais": int(fates.get("absorbed_by_ais", 0)),
        "miss_filtered": int(fates.get("filtered_as_non_vessel", 0)),
        "ais_matched": int((cands["status"] == "ais").sum()),
    }


def run(dets: pd.DataFrame, labels: pd.DataFrame, **kw):
    """All scenes in dets. Returns (summary dict, per-scene DataFrame, candidates, labels-with-fate)."""
    rng = np.random.default_rng(kw.pop("seed", 0))
    cs, ls, per = [], [], []
    for sid in sorted(dets["scene_id"].unique()):
        c, lab = simulate_scene(dets[dets["scene_id"] == sid], labels[labels["scene_id"] == sid], rng=rng, **kw)
        c["scene_id"], lab["scene_id"] = sid, sid
        cs.append(c)
        ls.append(lab)
        per.append({"scene_id": sid, **summarise(c, lab)})
    C, L = pd.concat(cs, ignore_index=True), pd.concat(ls, ignore_index=True)
    return summarise(C, L), pd.DataFrame(per), C, L
