"""Spatial helpers. H3 hexagons let any region be treated as a set of cells, so local
baselines ("what is normal in this cell for this vessel type") need no hardcoded
geography."""

from __future__ import annotations

import h3
import numpy as np
import pandas as pd


def add_h3(df: pd.DataFrame, resolution: int, col: str = "h3") -> pd.DataFrame:
    """Return a copy of df with an H3 cell id column computed from lat/lon."""
    out = df.copy()
    out[col] = [
        h3.latlng_to_cell(lat, lon, resolution) if pd.notna(lat) and pd.notna(lon) else None
        for lat, lon in zip(out["lat"].to_numpy(), out["lon"].to_numpy(), strict=True)
    ]
    return out


def haversine_km(lat1, lon1, lat2, lon2):
    """Great-circle distance in km. Works on scalars or numpy arrays."""
    r = 6371.0088
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * r * np.arcsin(np.sqrt(a))
