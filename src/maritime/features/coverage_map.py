"""Learned coverage map and gap surprise score.

Question: given where a ship was, how unusual is a silence this long?

Silences are normalized by the expected report interval: r = silence / expected.
For a silence starting in H3 cell c:

  P(r > x | c) = p_c * S(x - k)        for x >= k (the gap threshold)

- p_c: probability that a silence starting in c exceeds k at all. This is the
  coverage part; badly covered cells have high p_c. Estimated per cell with
  empirical Bayes shrinkage toward the parent hexagon:
      p_child = (exceed + alpha * p_parent) / (n + alpha)
  A cell with lots of data keeps its own rate; a sparse cell leans on its parent's.
  alpha = 200 means the prior is worth 200 observed silences.

- S: how far past k long silences run, shared by all cells. Generalized Pareto
  survival function, fitted by maximum likelihood (scipy). This is the
  peaks-over-threshold method from extreme value theory: excesses over a high
  threshold follow a GPD. Gulf week 1 fit: shape xi ~ 1.09 (very heavy tail: coverage
  dropouts mixed with ships switched off in port for days), scale ~ 36.

Why S is global: versions with a per-cell tail failed calibration. An exponential
per-cell tail was U-shaped (real tail far heavier, scores up to 10^-72). A per-cell
GPD scale derived from the mean excess broke because with xi near 1 the mean is
dominated by a few extreme silences. A single global GPD calibrates well: on training
excesses the conditional tail probabilities are close to uniform (each decile 9-12%).

Training data: every consecutive class A ping pair, excluding pairs that overlap a
region-wide outage (those say nothing about local coverage).

Gap surprise = -log10 P(r > r_gap), using the larger p of the start and end cells
(if either end is badly covered, the silence is less surprising).
Surprise 2 means "1 in 100 silences here run this long", 4 means 1 in 10,000.
Surprise measures coverage only: a ship switched off in port is surprising but
innocent. Context (in port, did not move) belongs to risk scoring, not here.
"""

from __future__ import annotations

import h3
import numpy as np
import pandas as pd
from scipy.stats import genpareto

from maritime.features.gaps import SOURCE_FLOOR_SECONDS, SPEC_INTERVAL_SQL

LEVELS = (5, 4, 3)  # finest first; res 5 cells are ~250 km^2


def interval_stats(con, source: str, source_name: str, outage_intervals: pd.DataFrame,
                   k: float = 30.0) -> tuple[pd.DataFrame, np.ndarray]:
    """Per rounded (~1 km) start position: number of silences and how many exceeded k.
    Also returns every excess (r - k), for fitting the tail."""
    floor = SOURCE_FLOOR_SECONDS[source_name]
    out = outage_intervals if len(outage_intervals) else pd.DataFrame(
        {"start": pd.Series(dtype="datetime64[us]"), "end": pd.Series(dtype="datetime64[us]")})
    con.register("cov_outages", out[["start", "end"]])
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE cov_r AS
        WITH p AS (
            SELECT ts, lag(ts) OVER w AS p_ts, lag(lat) OVER w AS p_lat, lag(lon) OVER w AS p_lon,
                   lag(sog) OVER w AS p_sog, lag(nav_status) OVER w AS p_nav,
                   lag(transceiver_class) OVER w AS p_cls
            FROM {source} WHERE transceiver_class = 'A'
            WINDOW w AS (PARTITION BY mmsi ORDER BY ts)
        ),
        r AS (
            SELECT round(p_lat, 2) AS lat, round(p_lon, 2) AS lon, p_ts, ts,
                   epoch(ts - p_ts) / greatest({SPEC_INTERVAL_SQL}, {floor}) AS r
            FROM p WHERE p_ts IS NOT NULL
        )
        SELECT lat, lon, r FROM r
        WHERE NOT EXISTS (SELECT 1 FROM cov_outages o WHERE o.start < r.ts AND o."end" > r.p_ts)
    """)
    stats = con.sql(f"""
        SELECT lat, lon, count(*) AS n, count(*) FILTER (WHERE r > {k}) AS exceed
        FROM cov_r GROUP BY lat, lon
    """).df()
    excess = con.sql(f"SELECT r - {k} AS e FROM cov_r WHERE r > {k}").df()["e"].to_numpy()
    con.execute("DROP TABLE cov_r")
    return stats, excess


def fit(stats: pd.DataFrame, excess: np.ndarray, alpha: float = 200.0) -> dict:
    """Returns {"cells": {res: DataFrame[n, exceed, p]}, "p0", "xi", "sigma"}."""
    xi, _, sigma = genpareto.fit(excess, floc=0)

    finest = LEVELS[0]
    cells = [h3.latlng_to_cell(a, b, finest) for a, b in zip(stats.lat, stats.lon, strict=True)]
    agg = {finest: stats.assign(cell=cells).groupby("cell")[["n", "exceed"]].sum()}
    for res in LEVELS[1:]:
        parents = [h3.cell_to_parent(c, res) for c in agg[finest].index]
        agg[res] = agg[finest].groupby(parents).sum()

    n, x = agg[finest].sum()
    p0 = x / n
    fitted: dict[int, pd.DataFrame] = {}
    parent_p, parent_res = None, None
    for res in reversed(LEVELS):  # coarse to fine
        d = agg[res].copy()
        if parent_p is None:
            prior = pd.Series(p0, index=d.index)
        else:
            parents = [h3.cell_to_parent(c, parent_res) for c in d.index]
            prior = pd.Series(parent_p.reindex(parents).to_numpy(), index=d.index)
        d["p"] = (d.exceed + alpha * prior) / (d.n + alpha)
        fitted[res] = d
        parent_p, parent_res = d["p"], res
    return {"cells": fitted, "p0": float(p0), "xi": float(xi), "sigma": float(sigma)}


def cell_p(model: dict, lat: float, lon: float) -> tuple[float, int]:
    """Long-silence probability for a point, backing off to coarser cells if unseen."""
    for res in LEVELS:
        c = h3.latlng_to_cell(lat, lon, res)
        if c in model["cells"][res].index:
            return float(model["cells"][res].at[c, "p"]), res
    return model["p0"], 0


def tail_probability(model: dict, p: float, r: float, k: float = 30.0) -> float:
    """P(silence ratio > r) in a cell with long-silence probability p."""
    if r <= k:
        return 1.0
    return p * float(genpareto.sf(r - k, model["xi"], 0, model["sigma"]))


def score_gaps(gaps: pd.DataFrame, model: dict, k: float = 30.0) -> pd.DataFrame:
    rows = []
    for g in gaps.itertuples():
        ps, rs = cell_p(model, g.start_lat, g.start_lon)
        pe, re_ = cell_p(model, g.end_lat, g.end_lon)
        prob = tail_probability(model, max(ps, pe), g.ratio_outside_outage, k)
        rows.append((ps, pe, prob, -np.log10(max(prob, 1e-300)), min(rs, re_)))
    cols = ["p_long_start", "p_long_end", "tail_prob", "surprise", "coverage_res"]
    return gaps.join(pd.DataFrame(rows, columns=cols, index=gaps.index))


def cells_table(model: dict, res: int = LEVELS[0]) -> pd.DataFrame:
    """Coverage cells with centroids, for maps and inspection."""
    d = model["cells"][res].copy()
    centers = [h3.cell_to_latlng(c) for c in d.index]
    d["lat"], d["lon"] = [c[0] for c in centers], [c[1] for c in centers]
    return d.rename_axis("cell").reset_index()
