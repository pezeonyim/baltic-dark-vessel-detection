"""Coverage map: shrinkage and tail scoring on synthetic counts."""

import numpy as np
import pandas as pd

from maritime.features import coverage_map as cm


def model():
    # well-observed good cell (Louisiana), well-observed bad cell (Mexican waters),
    # and a barely observed cell next to the good one
    stats = pd.DataFrame({
        "lat": [29.0, 21.0, 29.0],
        "lon": [-90.0, -93.0, -89.95],
        "n": [100_000, 100_000, 5],
        "exceed": [10, 3_000, 5],
    })
    rng = np.random.default_rng(0)
    excess = rng.pareto(1.0, 5_000) * 30
    return cm.fit(stats, excess)


def test_poor_coverage_cell_has_higher_long_silence_rate():
    m = model()
    p_good, _ = cm.cell_p(m, 29.0, -90.0)
    p_bad, _ = cm.cell_p(m, 21.0, -93.0)
    assert p_bad > 100 * p_good


def test_sparse_cell_is_shrunk_toward_parent():
    m = model()
    p_sparse, _ = cm.cell_p(m, 29.0, -89.95)
    assert p_sparse < 0.5            # raw rate would be 5/5 = 1.0


def test_longer_silence_is_more_surprising():
    m = model()
    p, _ = cm.cell_p(m, 29.0, -90.0)
    assert cm.tail_probability(m, p, 40) > cm.tail_probability(m, p, 400) > 0
    assert cm.tail_probability(m, p, 10) == 1.0   # below threshold: not a gap
