"""Vessel labels from facts, not models.

Flag state comes from the MMSI: for ship stations (first digit 2-7) the first three
digits are the Maritime Identification Digits (MID), assigned by the ITU to a country.
Table: reference/mid.csv, from github.com/punio/mid (ITU list), with fixes: France
'FX' -> 'FR', missing codes for Azores/Madeira (PT), Alaska (US) and French southern
territories (TF), plus a `sovereign` column mapping territories to their state.

Flag groups use the sovereign state, so Gibraltar or Bermuda count as NATO-flagged,
the Faroe Islands as Danish, and Hong Kong / Macao as Chinese.

Caveats shown to users: flag is not ownership or allegiance. Shadow fleet tankers mostly
fly flags of convenience (Panama, Liberia, Gabon, Cameroon...) and some broadcast a false
MMSI, so "Russian-flagged" misses most Russian-controlled ships.
"""

from __future__ import annotations

import csv
from functools import lru_cache
from importlib import resources

import pandas as pd

NATO = {
    "AL", "BE", "BG", "CA", "HR", "CZ", "DK", "EE", "FI", "FR", "DE", "GR", "HU", "IS",
    "IT", "LV", "LT", "LU", "ME", "NL", "MK", "NO", "PL", "PT", "RO", "SK", "SI", "ES",
    "SE", "TR", "GB", "US",
}  # 32 members as of 2026
GROUPS = ("nato", "russia", "china", "other", "unknown")

MILITARY_TYPE = 35         # AIS ship type "military ops"
LAW_ENFORCEMENT_TYPE = 55


@lru_cache(maxsize=1)
def mid_table() -> dict[int, dict[str, str]]:
    path = resources.files("maritime.reference").joinpath("mid.csv")
    with path.open() as f:
        return {int(r["mid"]): r for r in csv.DictReader(f)}


def mid_of(mmsi: int | None) -> int | None:
    """MID for ship-station MMSIs (9 digits, first digit 2-7); None otherwise."""
    if mmsi is None or not (200_000_000 <= int(mmsi) <= 799_999_999):
        return None
    return int(mmsi) // 1_000_000


def flag_group(sovereign: str | None) -> str:
    if not sovereign:
        return "unknown"
    if sovereign == "RU":
        return "russia"
    if sovereign == "CN":
        return "china"
    if sovereign in NATO:
        return "nato"
    return "other"


def flag_info(mmsi: int | None) -> dict[str, str | None]:
    row = mid_table().get(mid_of(mmsi) or -1)
    if not row:
        return {"flag_country": None, "flag_iso2": None, "flag_group": "unknown"}
    return {"flag_country": row["country"], "flag_iso2": row["iso2"],
            "flag_group": flag_group(row["sovereign"])}


def add_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Add flag_country, flag_iso2, flag_group, is_military, is_law_enforcement."""
    out = df.copy()
    info = pd.DataFrame([flag_info(m) for m in out["mmsi"]], index=out.index)
    out = out.join(info)
    vt = out["vessel_type"] if "vessel_type" in out else pd.Series(index=out.index, dtype=float)
    out["is_military"] = vt.eq(MILITARY_TYPE).fillna(False)
    out["is_law_enforcement"] = vt.eq(LAW_ENFORCEMENT_TYPE).fillna(False)
    return out
