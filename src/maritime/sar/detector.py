"""SAR vessel detector: data, model, losses, decoding, xView3-style scoring.

Self-contained (numpy, scipy, torch, torchvision only) so the same file runs locally,
on the server, and inside a Kaggle notebook (uploaded next to the tiles).

Model: a CenterNet-style point detector. A ResNet encoder (input channels chosen from CHANNELS:
VV/VH in dB, locally normalised variants, ancillary layers; see make_input)
with a small upsampling decoder predicts, at 1/4 resolution (128 x 128 for a 512 tile):
  heatmap   probability that an object is centred in this cell (focal loss)
  offset    sub-cell position of the centre (L1)
  length    object length in metres, as log(1 + m) (L1, where labelled)
  vessel    logit: vessel vs fixed infrastructure (BCE, where labelled)
  fishing   logit: fishing vs not, for vessels (BCE, where labelled)
Why points, not boxes: xView3 labels are points; a heatmap peak is the natural output
and is what strong xView3 solutions used.

Baseline: CFAR (constant false alarm rate), the classic radar ship detector: a pixel is a
detection if it is much brighter than the sea clutter around it.

Scoring follows the xView3 reference metric: per scene, Hungarian matching of predictions
to labels within 200 m (20 px at 10 m/px); predictions matched to LOW-confidence labels are
dropped (neither TP nor FP); unmatched LOW labels are not counted as misses.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

PIX_M = 10.0
MATCH_M = 200.0
STRIDE = 4


# ---------------------------------------------------------------- data

@dataclass
class TileSet:
    images: np.ndarray        # (n, 2, 512, 512) uint8: VV, VH quantised dB (0 = no data)
    origins: np.ndarray       # (n, 2) tile origin in scene pixels
    scene: np.ndarray         # (n,) scene id per tile
    objects: list             # per tile: structured array of objects inside it
    ancillary: dict = field(default_factory=dict)    # scene -> {layer: 2-D float array}
    scene_shape: dict = field(default_factory=dict)  # scene -> (H, W) in SAR pixels

    def __len__(self):
        return len(self.images)


def load_tiles(paths, keep=None) -> TileSet:
    """Load tiles from per-scene npz files into ONE preallocated array (no stacking copy).
    keep(scene_id, n_objects_per_tile) -> bool mask of tiles to keep (default: all)."""
    paths = sorted(paths)
    plans = []
    for p in paths:                                   # pass 1: small arrays only, to size the result
        z = np.load(p, allow_pickle=False)
        sid = str(p).rsplit("/", 1)[-1].removesuffix(".npz")
        org, o = z["origins"], z["objects"]
        counts = np.bincount(o["tile"], minlength=len(org)) if len(o) else np.zeros(len(org), int)
        mask = np.ones(len(org), bool) if keep is None else np.asarray(keep(sid, counts), bool)
        plans.append((p, sid, mask))
    n = int(sum(m.sum() for _, _, m in plans))
    images = np.empty((n, 2, 512, 512), np.uint8)
    origins, scenes, objs, anc, shapes = [], [], [], {}, {}
    i = 0
    for p, sid, mask in plans:                        # pass 2: decompress each array once
        z = np.load(p, allow_pickle=False)
        im, org, o = z["images"], z["origins"], z["objects"]
        shapes[sid] = tuple(int(v) for v in z["scene_shape"])
        anc[sid] = {k[4:]: z[k].astype(np.float32) for k in z.files if k.startswith("anc_")}
        for t in np.flatnonzero(mask):
            images[i] = im[t]
            origins.append(org[t])
            scenes.append(sid)
            objs.append(o[o["tile"] == t])
            i += 1
        del im
    return TileSet(images, np.array(origins).reshape(-1, 2), np.array(scenes), objs, anc, shapes)


# ---------------------------------------------------------------- input channels
# Each channel is a float32 512x512 map, roughly in [-3, 3]. Which ones the model sees is a
# config choice (input-layer selection by ablation).
CHANNELS = ("VV", "VH", "VV_local", "VH_local", "VV_minus_VH", "bathymetry", "wind_speed",
            "wind_dir_sin", "wind_dir_cos", "land_mask")


def _db(u8):
    return np.where(u8 > 0, (u8.astype(np.float32) - 1) / 254 * 60.0 - 50.0, np.nan)


def _anc_crop(layer, origin, scene_shape, size=512):
    """Crop a coarse scene-wide ancillary layer to one tile and upsample it to size x size."""
    from scipy import ndimage
    H, W = scene_shape
    sy, sx = layer.shape[0] / H, layer.shape[1] / W           # ancillary px per SAR px
    r0, c0 = origin[0] * sy, origin[1] * sx
    ys = np.clip(r0 + (np.arange(size) + 0.5) * sy - 0.5, 0, layer.shape[0] - 1)
    xs = np.clip(c0 + (np.arange(size) + 0.5) * sx - 0.5, 0, layer.shape[1] - 1)
    filled = np.where(np.isnan(layer), np.nanmedian(layer) if np.isfinite(layer).any() else 0, layer)
    yy, xx = np.meshgrid(ys, xs, indexing="ij")
    return ndimage.map_coordinates(filled, [yy, xx], order=1, mode="nearest").astype(np.float32)


def make_input(img_u8, channels, anc=None, origin=(0, 0), scene_shape=(512, 512)):
    """uint8 (2, H, W) tile (+ scene ancillary layers) -> float32 (len(channels), H, W)."""
    vv, vh = _db(img_u8[0]), _db(img_u8[1])
    anc = anc or {}
    out = []
    for ch in channels:
        if ch == "VV":
            a = (vv + 20.0) / 8.0                       # global scaling (sea ~ -20 dB)
        elif ch == "VH":
            a = (vh + 28.0) / 8.0
        elif ch == "VV_local":                          # subtract this tile's sea level (EDA Q2)
            a = (vv - np.nanmedian(vv)) / 8.0 if np.isfinite(vv).any() else vv
        elif ch == "VH_local":
            a = (vh - np.nanmedian(vh)) / 8.0 if np.isfinite(vh).any() else vh
        elif ch == "VV_minus_VH":
            a = (vv - vh - 8.0) / 6.0
        elif ch in ("bathymetry", "wind_speed", "wind_dir_sin", "wind_dir_cos", "land_mask"):
            key = {"bathymetry": "bathymetry", "wind_speed": "owiWindSpeed", "land_mask": "owiMask",
                   "wind_dir_sin": "owiWindDirection", "wind_dir_cos": "owiWindDirection"}[ch]
            if key not in anc:
                a = np.zeros(vv.shape, np.float32)      # layer absent for this scene
            else:
                m = _anc_crop(anc[key], origin, scene_shape, vv.shape[-1])
                if ch == "bathymetry":                  # metres, negative = below sea level
                    a = -np.sign(m) * np.log1p(np.abs(m)) / 4.0
                elif ch == "wind_speed":
                    a = (m - 7.0) / 4.0
                elif ch == "wind_dir_sin":
                    a = np.sin(np.deg2rad(m))
                elif ch == "wind_dir_cos":
                    a = np.cos(np.deg2rad(m))
                else:
                    a = (m > 0).astype(np.float32)      # owiMask: 0 = ocean
        else:
            raise ValueError(f"unknown channel {ch}; choose from {CHANNELS}")
        out.append(np.nan_to_num(a, nan=0.0))
    return np.stack(out).astype(np.float32)


def render_targets(objs, size=512, stride=STRIDE, sigma=1.5):
    """Heatmap + per-centre regression targets at 1/stride resolution."""
    h = size // stride
    heat = np.zeros((h, h), np.float32)
    ys, xs = np.mgrid[0:h, 0:h]
    centres = []
    for o in objs:
        cy, cx = o["row"] / stride, o["col"] / stride
        iy, ix = int(cy), int(cx)
        if not (0 <= iy < h and 0 <= ix < h):
            continue
        heat = np.maximum(heat, np.exp(-((ys - cy) ** 2 + (xs - cx) ** 2) / (2 * sigma**2)))
        heat[iy, ix] = 1.0
        centres.append((iy, ix, cy - iy, cx - ix, o["length_m"], o["is_vessel"], o["is_fishing"]))
    return heat, np.array(centres, np.float32).reshape(-1, 7)


def augment(img, objs, rng):
    """Random flips and 90-degree rotations (radar statistics are unchanged by these).
    img: (C, H, W) array of any dtype; objs: structured array with row/col."""
    size = img.shape[-1]
    rows, cols = objs["row"].astype(np.float64), objs["col"].astype(np.float64)
    if rng.random() < 0.5:
        img, cols = img[:, :, ::-1], size - 1 - cols
    if rng.random() < 0.5:
        img, rows = img[:, ::-1, :], size - 1 - rows
    k = int(rng.integers(4))
    for _ in range(k):                       # rotate 90 deg counter-clockwise
        img = np.rot90(img, 1, axes=(1, 2))
        rows, cols = size - 1 - cols, rows
    out = objs.copy()
    out["row"], out["col"] = rows, cols
    return np.ascontiguousarray(img), out


# ---------------------------------------------------------------- model

def build_model(backbone: str = "resnet18", pretrained: bool = False, in_ch: int = 2):
    import torch
    import torch.nn as nn
    import torchvision

    weights = "DEFAULT" if pretrained else None
    enc = getattr(torchvision.models, backbone)(weights=weights)
    w = enc.conv1.weight.data                      # (64, 3, 7, 7): adapt RGB -> in_ch channels
    enc.conv1 = nn.Conv2d(in_ch, 64, 7, 2, 3, bias=False)
    enc.conv1.weight.data = w.mean(1, keepdim=True).repeat(1, in_ch, 1, 1) * (3.0 / in_ch)
    ch = [64, 64, 128, 256, 512] if backbone in ("resnet18", "resnet34") else [64, 256, 512, 1024, 2048]

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.stem = nn.Sequential(enc.conv1, enc.bn1, enc.relu)        # 1/2
            self.pool = enc.maxpool                                        # 1/4
            self.l1, self.l2, self.l3, self.l4 = enc.layer1, enc.layer2, enc.layer3, enc.layer4
            def up(cin, cout):
                return nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU())
            self.u3 = up(ch[4] + ch[3], 256)
            self.u2 = up(256 + ch[2], 128)
            self.u1 = up(128 + ch[1], 64)
            def head(cout, bias=0.0):
                h = nn.Sequential(nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(), nn.Conv2d(64, cout, 1))
                nn.init.constant_(h[-1].bias, bias)
                return h
            self.heat = head(1, -2.19)        # prior p ~ 0.1, standard CenterNet init
            self.offset, self.length = head(2), head(1)
            self.vessel, self.fishing = head(1), head(1)

        def forward(self, x):
            f1 = self.l1(self.pool(self.stem(x)))          # 1/4
            f2 = self.l2(f1)                                # 1/8
            f3 = self.l3(f2)                                # 1/16
            f4 = self.l4(f3)                                # 1/32
            def cat_up(a, b):
                return torch.cat([nn.functional.interpolate(a, size=b.shape[-2:], mode="nearest"), b], 1)
            d = self.u3(cat_up(f4, f3))
            d = self.u2(cat_up(d, f2))
            d = self.u1(cat_up(d, f1))                      # back to 1/4
            return {"heat": self.heat(d), "offset": self.offset(d), "length": self.length(d),
                    "vessel": self.vessel(d), "fishing": self.fishing(d)}

    return Net()


def focal_loss(logits, target, alpha=2.0, beta=4.0):
    """CenterNet focal loss: down-weights easy negatives and negatives near a centre."""
    import torch
    p = torch.sigmoid(logits).clamp(1e-4, 1 - 1e-4)
    pos = target.eq(1).float()
    neg = 1 - pos
    pos_loss = -torch.log(p) * (1 - p) ** alpha * pos
    neg_loss = -torch.log(1 - p) * p**alpha * (1 - target) ** beta * neg
    n = pos.sum().clamp(min=1)
    return (pos_loss.sum() + neg_loss.sum()) / n


def compute_loss(out, heat_t, centres_list):
    """centres_list: per image (k, 7) array [iy, ix, dy, dx, length, vessel, fishing]."""
    import torch
    import torch.nn.functional as F
    loss_heat = focal_loss(out["heat"][:, 0], heat_t)
    off_l, len_l, ves_l, fis_l = [], [], [], []
    for b, c in enumerate(centres_list):
        if len(c) == 0:
            continue
        c = torch.as_tensor(c, device=heat_t.device)
        iy, ix = c[:, 0].long(), c[:, 1].long()
        off_l.append(F.l1_loss(out["offset"][b, :, iy, ix].T, c[:, 2:4], reduction="sum"))
        m = ~torch.isnan(c[:, 4])
        if m.any():
            len_l.append(F.l1_loss(out["length"][b, 0, iy, ix][m], torch.log1p(c[m, 4]), reduction="sum"))
        m = ~torch.isnan(c[:, 5])
        if m.any():
            ves_l.append(F.binary_cross_entropy_with_logits(out["vessel"][b, 0, iy, ix][m], c[m, 5], reduction="sum"))
        m = ~torch.isnan(c[:, 6]) & (c[:, 5] == 1)
        if m.any():
            fis_l.append(F.binary_cross_entropy_with_logits(out["fishing"][b, 0, iy, ix][m], c[m, 6], reduction="sum"))
    n = max(sum(len(c) for c in centres_list), 1)
    def total(lst):
        return torch.stack(lst).sum() / n if lst else heat_t.new_zeros(())
    parts = {"heat": loss_heat, "offset": total(off_l), "length": total(len_l),
             "vessel": total(ves_l), "fishing": total(fis_l)}
    loss = parts["heat"] + parts["offset"] + 0.5 * parts["length"] + 0.5 * parts["vessel"] + 0.5 * parts["fishing"]
    return loss, {k: float(v.detach()) for k, v in parts.items()}


def decode(out, threshold=0.3, max_det=100):
    """Heatmap peaks (3x3 local maxima above threshold) -> detections per image:
    (row, col, score, length_m, p_vessel, p_fishing) in tile pixel coordinates."""
    import torch
    import torch.nn.functional as F
    heat = torch.sigmoid(out["heat"][:, 0])
    peaks = heat * (heat == F.max_pool2d(heat[:, None], 3, 1, 1)[:, 0])
    results = []
    for b in range(heat.shape[0]):
        scores, idx = peaks[b].flatten().topk(min(max_det, peaks[b].numel()))
        keep = scores > threshold
        scores, idx = scores[keep], idx[keep]
        w = heat.shape[-1]
        iy, ix = idx // w, idx % w
        off = out["offset"][b, :, iy, ix]
        rows = (iy.float() + off[0]) * STRIDE
        cols = (ix.float() + off[1]) * STRIDE
        length = torch.expm1(out["length"][b, 0, iy, ix]).clamp(min=0)
        pv = torch.sigmoid(out["vessel"][b, 0, iy, ix])
        pf = torch.sigmoid(out["fishing"][b, 0, iy, ix])
        results.append(torch.stack([rows, cols, scores, length, pv, pf], 1).detach().cpu().numpy())
    return results


# ---------------------------------------------------------------- CFAR baseline

def cfar(img_u8, guard=4, train=16, k=5.0, min_px=2):
    """Cell-averaging CFAR on VV (dB): detect pixels > local clutter mean + k * std,
    with clutter measured in a ring outside a guard window; returns (row, col, score)."""
    from scipy import ndimage
    db = img_u8[0].astype(np.float32)
    valid = db > 0
    big, small = 2 * (guard + train) + 1, 2 * guard + 1
    def box(a, s):
        return ndimage.uniform_filter(a, s, mode="nearest") * s * s
    n = big * big - small * small
    s1 = box(db, big) - box(db, small)
    s2 = box(db**2, big) - box(db**2, small)
    mean = s1 / n
    std = np.sqrt(np.maximum(s2 / n - mean**2, 1e-6))
    det = valid & (db > mean + k * std)
    lab, nlab = ndimage.label(det)
    if nlab == 0:
        return np.zeros((0, 3))
    idx = np.arange(1, nlab + 1)
    sizes = ndimage.sum(det, lab, idx)
    com = np.array(ndimage.center_of_mass(det, lab, idx)).reshape(-1, 2)
    peak = ndimage.maximum((db - mean) / std, lab, idx)
    keep = sizes >= min_px
    return np.column_stack([com[keep], peak[keep]])


# ---------------------------------------------------------------- xView3-style scoring

def match(pred_rc, gt_rc, max_px=MATCH_M / PIX_M):
    """Hungarian matching within max_px; returns list of (pred_idx, gt_idx)."""
    from scipy.optimize import linear_sum_assignment
    from scipy.spatial import distance_matrix
    if len(pred_rc) == 0 or len(gt_rc) == 0:
        return []
    d = distance_matrix(pred_rc, gt_rc)
    big = 1e9
    rows, cols = linear_sum_assignment(np.where(d <= max_px, d, big))
    return [(r, c) for r, c in zip(rows, cols, strict=True) if d[r, c] <= max_px]


def score(preds: dict, gts: dict) -> dict:
    """preds/gts: scene_id -> structured arrays.
    preds: rows/cols (scene px), is_vessel, is_fishing, length_m (vessel/fishing as 0/1).
    gts: same + confidence. Returns detection P/R/F1, vessel F1, fishing F1, length error."""
    tp = fp = fn = 0
    vessel, fishing, length_err = [], [], []
    for sid in set(preds) | set(gts):
        p = preds.get(sid, np.zeros(0, dtype=[("row", "f4"), ("col", "f4")]))
        g = gts.get(sid, np.zeros(0, dtype=[("row", "f4"), ("col", "f4"), ("confidence", "U6")]))
        pairs = match(np.column_stack([p["row"], p["col"]]) if len(p) else np.zeros((0, 2)),
                      np.column_stack([g["row"], g["col"]]) if len(g) else np.zeros((0, 2)))
        low = (g["confidence"] == "LOW") if len(g) else np.zeros(0, bool)
        matched_p = {a for a, _ in pairs}
        matched_g = {b for _, b in pairs}
        good = [(a, b) for a, b in pairs if not low[b]]
        tp += len(good)
        fp += len(p) - len(matched_p)                     # unmatched predictions
        fn += int(sum(1 for b in range(len(g)) if b not in matched_g and not low[b]))
        for a, b in good:
            if "is_vessel" in g.dtype.names and not np.isnan(g["is_vessel"][b]):
                vessel.append((p["is_vessel"][a], g["is_vessel"][b]))
                if g["is_vessel"][b] == 1 and not np.isnan(g["is_fishing"][b]):
                    fishing.append((p["is_fishing"][a], g["is_fishing"][b]))
            if "length_m" in g.dtype.names and not np.isnan(g["length_m"][b]):
                length_err.append(abs(p["length_m"][a] - g["length_m"][b]) / max(g["length_m"][b], 1e-6))

    def f1(tp_, fp_, fn_):
        prec = tp_ / (tp_ + fp_) if tp_ + fp_ else 0.0
        rec = tp_ / (tp_ + fn_) if tp_ + fn_ else 0.0
        return prec, rec, (2 * prec * rec / (prec + rec) if prec + rec else 0.0)

    def class_f1(pairs):
        if not pairs:
            return float("nan")
        a = np.array(pairs, dtype=float)
        t = ((a[:, 0] >= 0.5) & (a[:, 1] == 1)).sum()
        f_p = ((a[:, 0] >= 0.5) & (a[:, 1] == 0)).sum()
        f_n = ((a[:, 0] < 0.5) & (a[:, 1] == 1)).sum()
        return f1(t, f_p, f_n)[2]

    prec, rec, f1d = f1(tp, fp, fn)
    pel = 1.0 - min(float(np.mean(length_err)), 1.0) if length_err else float("nan")
    return {"tp": tp, "fp": fp, "fn": fn, "precision": prec, "recall": rec, "f1_detect": f1d,
            "f1_vessel": class_f1(vessel), "f1_fishing": class_f1(fishing), "length_score": pel}


# ---------------------------------------------------------------- whole-scene inference + full xView3 metric

def scene_tiles(H, W, size=512, overlap=64):
    """Tile origins covering an H x W scene with `overlap` px between neighbours (last row/col flush
    with the scene edge)."""
    def starts(n):
        if n <= size:
            return [0]
        s = list(range(0, n - size, size - overlap)) + [n - size]
        return sorted(set(s))
    return [(r, c) for r in starts(H) for c in starts(W)]


def keep_core(dets, origin, scene_shape, size=512, margin=32):
    """Drop detections within `margin` px of a tile edge, unless that edge is the scene edge
    (a neighbouring tile sees those objects with more context)."""
    if len(dets) == 0:
        return dets
    r0, c0 = origin
    H, W = scene_shape
    r, c = dets[:, 0], dets[:, 1]
    lo_r = 0 if r0 == 0 else margin
    lo_c = 0 if c0 == 0 else margin
    hi_r = size if r0 + size >= H else size - margin
    hi_c = size if c0 + size >= W else size - margin
    return dets[(r >= lo_r) & (r < hi_r) & (c >= lo_c) & (c < hi_c)]


def merge_close(dets, radius_px=10.0):
    """Greedy de-duplication: keep the highest-score detection, drop others within radius_px.
    dets: (n, k) with columns row, col, score, ..."""
    if len(dets) < 2:
        return dets
    from scipy.spatial import cKDTree
    order = np.argsort(-dets[:, 2])
    d = dets[order]
    tree = cKDTree(d[:, :2])
    alive = np.ones(len(d), bool)
    for i in range(len(d)):
        if alive[i]:
            nb = tree.query_ball_point(d[i, :2], radius_px)
            alive[[j for j in nb if j > i]] = False
    return d[alive]


def shore_distance_px(points_rc, contours):
    """Distance (px) from each point to the nearest shoreline vertex; inf when there is no shoreline."""
    pts = np.asarray(points_rc, float).reshape(-1, 2)
    segs = [np.asarray(c, float).reshape(-1, 2) for c in (contours if contours is not None else [])]
    segs = [s for s in segs if len(s)]
    if not segs or not len(pts):
        return np.full(len(pts), np.inf)
    from scipy.spatial import cKDTree
    return cKDTree(np.concatenate(segs)).query(pts)[0]


def xview3_metrics(preds: dict, gts: dict, shorelines: dict | None = None, shore_km=2.0) -> dict:
    """Full xView3 scoring: score() plus close-to-shore detection F1 (labels and predictions within
    shore_km of the shoreline, matched among themselves) and the aggregate
    F1_D * (1 + F1_shore + F1_vessel + F1_fishing + length_score) / 5 (NaN parts count as 0)."""
    out = score(preds, gts)
    if shorelines:
        lim = shore_km * 1000 / PIX_M
        def near(d, sid):
            if d is None or not len(d):
                return d
            dist = shore_distance_px(np.column_stack([d["row"], d["col"]]), shorelines.get(sid))
            return d[dist <= lim]
        sids = [s for s in set(preds) | set(gts) if s in shorelines]
        ps = {s: near(preds.get(s), s) for s in sids if preds.get(s) is not None}
        gs = {s: near(gts.get(s), s) for s in sids if gts.get(s) is not None}
        sh = score(ps, gs)
        out.update(f1_shore=sh["f1_detect"], shore_tp=sh["tp"], shore_fp=sh["fp"], shore_fn=sh["fn"])
    else:
        out.update(f1_shore=float("nan"))
    def z(v):
        return 0.0 if v is None or (isinstance(v, float) and np.isnan(v)) else float(v)
    out["aggregate"] = out["f1_detect"] * (1 + z(out["f1_shore"]) + z(out["f1_vessel"])
                                           + z(out["f1_fishing"]) + z(out["length_score"])) / 5
    return out


# ---------------------------------------------------------------- shared scene pipeline (eval, mining, calibration)

# Same layout as maritime.sar.xview3.OBJECT_DTYPE (kept here so this file stays self-contained on Kaggle).
OBJECT_DTYPE = np.dtype([
    ("tile", "i4"), ("row", "i2"), ("col", "i2"),
    ("is_vessel", "f4"), ("is_fishing", "f4"), ("length_m", "f4"),
    ("confidence", "U6"), ("shore_km", "f4"),
])


def predict_scene(models, img, channels=("VV", "VH"), size=512, overlap=64, batch=32, min_valid=0.05,
                  device="cpu", num_workers=0, merge_px=4.0, decode_threshold=0.05, max_det=200, keep_tile=None):
    """Slide `models` over a whole (2, H, W) uint8 scene. Returns (dets, origins): dets is a list with one
    (n, 6) array per model plus a final one for the output-averaged ensemble; columns are
    row, col, score, length_m, p_vessel, p_fishing in scene pixels. keep_tile(origin) -> bool can skip more
    tiles (e.g. all land) on top of the no-data rule."""
    import torch
    H, W = img.shape[1:]
    origins = [o for o in scene_tiles(H, W, size, overlap)
               if (img[0, o[0]:o[0] + size:8, o[1]:o[1] + size:8] > 0).mean() >= min_valid
               and (keep_tile is None or keep_tile(o))]

    class _Tiles(torch.utils.data.Dataset):
        def __len__(self):
            return len(origins)

        def __getitem__(self, i):
            r, c = origins[i]
            t = img[:, r:r + size, c:c + size]
            if t.shape[1:] != (size, size):
                t = np.pad(t, ((0, 0), (0, size - t.shape[1]), (0, size - t.shape[2])))
            return torch.from_numpy(make_input(t, list(channels))), i

    dl = torch.utils.data.DataLoader(_Tiles(), batch_size=batch, num_workers=num_workers)
    per = [[] for _ in range(len(models) + 1)]
    with torch.no_grad():
        for x, idx in dl:
            # channels_last is ~25-30% faster for convolutions on ARM CPUs; models must match
            x = x.to(device, memory_format=torch.channels_last) if str(device) == "cpu" else x.to(device)
            with torch.autocast("cuda" if str(device).startswith("cuda") else "cpu",
                                enabled=str(device).startswith("cuda")):
                outs = [{k: v.float() for k, v in m(x).items()} for m in models]
            outs.append({k: torch.stack([o[k] for o in outs]).mean(0) for k in outs[0]})
            for j, o in enumerate(outs):
                for d, i in zip(decode(o, threshold=decode_threshold, max_det=max_det), idx.tolist(), strict=True):
                    d = keep_core(d, origins[i], (H, W), size, overlap // 2)
                    if len(d):
                        d = d.copy()
                        d[:, 0] += origins[i][0]
                        d[:, 1] += origins[i][1]
                        per[j].append(d)
    dets = [merge_close(np.concatenate(p), merge_px) if p else np.zeros((0, 6)) for p in per]
    return dets, origins


def unmatched_detections(dets, label_rc, min_score=0.3, radius_px=30.0):
    """Detections scoring >= min_score with no label of ANY confidence within radius_px:
    candidate false alarms (a lenient radius so slightly misplaced labels don't count as negatives)."""
    d = dets[dets[:, 2] >= min_score]
    if not len(d) or not len(label_rc):
        return d
    from scipy.spatial import cKDTree
    dist = cKDTree(np.asarray(label_rc, float).reshape(-1, 2)).query(d[:, :2])[0]
    return d[dist > radius_px]


def hard_negative_tiles(img, false_alarms, labels, max_tiles=150, size=512, margin=64, seed=0):
    """Training tiles around false alarms: each false alarm lands at a random spot at least `margin` px
    inside its tile; false alarms already inside an earlier tile are skipped (no near-duplicate tiles). Any real labels inside a
    tile are kept as objects, so the tile teaches "this is sea" without erasing true ships.
    labels: structured array with row, col, is_vessel, is_fishing, length_m, confidence, shore_km (scene px).
    Returns images (n, 2, size, size) uint8, origins (n, 2), objects (OBJECT_DTYPE)."""
    rng = np.random.default_rng(seed)
    H, W = img.shape[1:]
    fa = false_alarms[np.argsort(-false_alarms[:, 2])] if len(false_alarms) else false_alarms
    origins = []
    for r, c in fa[:, :2] if len(fa) else []:
        if len(origins) >= max_tiles:
            break
        if any(o[0] + 16 <= r < o[0] + size - 16 and o[1] + 16 <= c < o[1] + size - 16
               for o in origins):                  # already inside a tile: avoid near-duplicate tiles
            continue
        r0 = int(np.clip(r - rng.integers(margin, size - margin), 0, max(H - size, 0)))
        c0 = int(np.clip(c - rng.integers(margin, size - margin), 0, max(W - size, 0)))
        origins.append((r0, c0))
    images = np.zeros((len(origins), 2, size, size), np.uint8)
    objs = []
    for t, (r0, c0) in enumerate(origins):
        tile = img[:, r0:r0 + size, c0:c0 + size]
        images[t, :, :tile.shape[1], :tile.shape[2]] = tile
        if len(labels):
            m = (labels["row"] >= r0) & (labels["row"] < r0 + size) & (labels["col"] >= c0) & (labels["col"] < c0 + size)
            for lb in labels[m]:
                objs.append((t, int(lb["row"]) - r0, int(lb["col"]) - c0, lb["is_vessel"], lb["is_fishing"],
                             lb["length_m"], lb["confidence"], lb["shore_km"]))
    return images, np.array(origins, np.int32).reshape(-1, 2), np.array(objs, OBJECT_DTYPE)
