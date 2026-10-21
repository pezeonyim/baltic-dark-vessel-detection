"""SAR + AIS fusion: which radar detections have no AIS broadcast behind them?

A SAR image is a snapshot at one instant. AIS is a stream of position reports at irregular times. To
compare them we (1) estimate where every AIS transmitter was at the image time, with an uncertainty
that grows with the time since its nearest report, (2) match detections to those positions one-to-one
(Hungarian), allowing each AIS position a gate of base_gate_m + its uncertainty, and (3) call the
unmatched detections that the detector thinks are vessels "dark-vessel candidates".

Everything here works in local metres (x east, y north). Use `to_local_m` for lat/lon input; xView3
scene pixels are 10 m, so pixel coordinates times 10 work directly.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

KNOT_MS = 0.514444
EARTH_R = 6_371_000.0


def to_local_m(lat, lon, lat0: float, lon0: float):
    """Equirectangular projection around (lat0, lon0). Accurate to well under 1% over a 250 km scene."""
    lat, lon = np.asarray(lat, float), np.asarray(lon, float)
    x = np.radians(lon - lon0) * EARTH_R * np.cos(np.radians(lat0))
    y = np.radians(lat - lat0) * EARTH_R
    return x, y


def ais_at_time(ais: pd.DataFrame, t, max_age_s: float = 1800.0, pos_err_m: float = 50.0,
                drift_ms: float = 1.0) -> pd.DataFrame:
    """Estimate each MMSI's position at time t.

    ais: columns mmsi, ts (datetime64 UTC), x, y (local metres), sog (knots), cog (degrees).
    If reports bracket t, interpolate linearly between them; otherwise dead-reckon from the nearest
    report with its sog/cog. Transmitters whose nearest report is more than max_age_s away are dropped.
    Returns mmsi, x, y, dt_s (time to nearest report) and unc_m: pos_err_m + drift_ms * dt_s (for
    interpolation, dt_s is the shorter side of the bracket), i.e. how far off the estimate could be.
    """
    t = pd.Timestamp(t)
    out = []
    for mmsi, g in ais.sort_values("ts").groupby("mmsi", sort=False):
        ts = g["ts"].to_numpy()
        dts = (ts - np.datetime64(t.to_datetime64())) / np.timedelta64(1, "s")
        before, after = np.where(dts <= 0)[0], np.where(dts > 0)[0]
        near = int(np.argmin(np.abs(dts)))
        if abs(dts[near]) > max_age_s:
            continue
        if len(before) and len(after) and (dts[after[0]] - dts[before[-1]]) <= 2 * max_age_s:
            a, b = before[-1], after[0]
            w = -dts[a] / (dts[b] - dts[a])
            x = g["x"].iat[a] + w * (g["x"].iat[b] - g["x"].iat[a])
            y = g["y"].iat[a] + w * (g["y"].iat[b] - g["y"].iat[a])
            dt = min(-dts[a], dts[b])
            r = g.iloc[a if -dts[a] <= dts[b] else b]
        else:
            r = g.iloc[near]
            sog = 0.0 if pd.isna(r.get("sog", np.nan)) else float(r["sog"])
            cog = 0.0 if pd.isna(r.get("cog", np.nan)) else float(r["cog"])
            v = sog * KNOT_MS
            dt_signed = -dts[near]        # seconds from the report to t (negative = report after t)
            x = r["x"] + v * np.sin(np.radians(cog)) * dt_signed
            y = r["y"] + v * np.cos(np.radians(cog)) * dt_signed
            dt = abs(dts[near])
        sog = r.get("sog", np.nan)
        cog = r.get("cog", np.nan)
        out.append((mmsi, float(x), float(y), float(dt), pos_err_m + drift_ms * float(dt),
                    float(sog) if pd.notna(sog) else np.nan, float(cog) if pd.notna(cog) else np.nan))
    return pd.DataFrame(out, columns=["mmsi", "x", "y", "dt_s", "unc_m", "sog", "cog"])


def doppler_shift_m(sog_kn, cog_deg, along: np.ndarray, incidence_deg, range_over_speed_s: float) -> np.ndarray:
    """Expected along-track displacement (m, unsigned) of a moving ship in a SAR image:
    (slant range / satellite speed) x radial velocity, radial velocity = ground speed across the track x
    sin(incidence). `along` is the unit vector of the satellite track in (east, north)."""
    sog = np.nan_to_num(np.asarray(sog_kn, float), nan=0.0) * KNOT_MS
    cog = np.radians(np.nan_to_num(np.asarray(cog_deg, float), nan=0.0))
    v = np.column_stack([sog * np.sin(cog), sog * np.cos(cog)])       # east, north
    across = np.array([-along[1], along[0]])
    v_r = np.abs(v @ across) * np.sin(np.radians(np.asarray(incidence_deg, float)))
    return range_over_speed_s * v_r


def match_to_ais(det_xy: np.ndarray, ais_xy: np.ndarray, base_gate_m: float = 200.0,
                 ais_unc_m: np.ndarray | None = None, along: np.ndarray | None = None,
                 along_extra_m: np.ndarray | None = None) -> np.ndarray:
    """One-to-one matching of detections to AIS positions (Hungarian on distance).
    A pair is allowed if distance <= base_gate_m + that AIS position's uncertainty. With `along` (unit vector of
    the satellite track) and `along_extra_m` (per AIS position), the gate is stretched along the track by that
    much: SAR shows a ship that moves towards or away from the radar shifted along the track (Doppler), by
    (slant range / satellite speed) x radial speed, often several hundred metres.
    Returns an int array, one entry per detection: index of the matched AIS position, or -1."""
    from scipy.optimize import linear_sum_assignment

    det_xy, ais_xy = np.asarray(det_xy, float).reshape(-1, 2), np.asarray(ais_xy, float).reshape(-1, 2)
    res = np.full(len(det_xy), -1, int)
    if len(det_xy) == 0 or len(ais_xy) == 0:
        return res
    gate = base_gate_m + (np.zeros(len(ais_xy)) if ais_unc_m is None else np.asarray(ais_unc_m, float))
    diff = det_xy[:, None, :] - ais_xy[None, :, :]
    if along is not None and along_extra_m is not None:
        u = np.asarray(along, float) / np.linalg.norm(along)
        da = diff @ u
        dc = diff @ np.array([-u[1], u[0]])
        g_along = gate + np.asarray(along_extra_m, float)
        d = np.sqrt((da / g_along[None, :]) ** 2 + (dc / gate[None, :]) ** 2) * gate[None, :]   # in "gate metres"
    else:
        d = np.sqrt((diff ** 2).sum(-1))
    ok = d <= gate[None, :]
    # only solve for rows/cols that have at least one admissible pair (keeps big scenes cheap)
    ri, ci = np.where(ok.any(1))[0], np.where(ok.any(0))[0]
    if len(ri) == 0:
        return res
    sub = np.where(ok[np.ix_(ri, ci)], d[np.ix_(ri, ci)], 1e12)
    r, c = linear_sum_assignment(sub)
    keep = sub[r, c] < 1e12
    res[ri[r[keep]]] = ci[c[keep]]
    return res


def dark_candidates(dets: pd.DataFrame, ais_pos: pd.DataFrame, score_min: float = 0.45,
                    p_vessel_min: float = 0.5, base_gate_m: float = 200.0,
                    along: np.ndarray | None = None) -> pd.DataFrame:
    """dets: x, y, score, p_vessel (+ anything else, carried through). ais_pos: x, y, unc_m (+ mmsi, and
    optionally along_extra_m with `along`, the satellite track direction: see match_to_ais).
    Returns the detections with score >= score_min, each labelled:
      status = "ais"       matched to an AIS position (column ais_idx = row of ais_pos)
               "dark"      unmatched and p_vessel >= p_vessel_min  -> dark-vessel candidate
               "non_vessel" unmatched but the detector thinks it is not a vessel (platform, turbine...)
    """
    d = dets[dets["score"] >= score_min].copy().reset_index(drop=True)
    unc = ais_pos["unc_m"].to_numpy() if "unc_m" in ais_pos else None
    extra = ais_pos["along_extra_m"].to_numpy() if "along_extra_m" in ais_pos else None
    idx = match_to_ais(d[["x", "y"]].to_numpy(), ais_pos[["x", "y"]].to_numpy(), base_gate_m, unc,
                       along=along if extra is not None else None, along_extra_m=extra)
    d["ais_idx"] = idx
    d["status"] = np.where(idx >= 0, "ais", np.where(d["p_vessel"] >= p_vessel_min, "dark", "non_vessel"))
    return d
