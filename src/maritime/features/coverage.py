"""Receiver coverage: region-wide outages.

An outage is a time bin in which far fewer vessels are heard than usual. We count
distinct class A vessels per bin and compare with the rolling median over the
surrounding +/- window_hours. Bins below `threshold` x median are outages.

Why vessels and not pings: one ship in a busy port can produce many pings; distinct
vessels measure how much of the fleet the receiver network can hear.

Why the median: it ignores the outage bins themselves, so a 4-hour outage does not
drag the baseline down (a mean would).

On Gulf week 1 the ratio is sharply bimodal: 967 of 1,008 ten-minute bins sit at
>= 0.9 of median and 34 sit below 0.2, so the threshold choice is not delicate.
"""

from __future__ import annotations

import duckdb
import pandas as pd


def vessel_volume(con: duckdb.DuckDBPyConnection, source: str, bin_minutes: int = 10,
                  window_hours: int = 12) -> pd.DataFrame:
    """Distinct class A vessels per bin, with rolling median and ratio."""
    return con.sql(f"""
        WITH b AS (
            SELECT time_bucket(INTERVAL {bin_minutes} MINUTE, ts) AS bin_start,
                   count(DISTINCT mmsi) AS vessels
            FROM {source}
            WHERE transceiver_class = 'A'
            GROUP BY 1
        )
        SELECT bin_start,
               bin_start + INTERVAL {bin_minutes} MINUTE AS bin_end,
               vessels,
               median(vessels) OVER w AS baseline,
               vessels / median(vessels) OVER w AS ratio
        FROM b
        WINDOW w AS (ORDER BY bin_start RANGE BETWEEN INTERVAL {window_hours} HOUR PRECEDING
                                              AND INTERVAL {window_hours} HOUR FOLLOWING)
        ORDER BY bin_start
    """).df()


def outages(volume: pd.DataFrame, threshold: float = 0.5) -> pd.DataFrame:
    """Merge consecutive low-volume bins into outage intervals."""
    low = volume[volume["ratio"] < threshold].sort_values("bin_start")
    if low.empty:
        return pd.DataFrame(columns=["start", "end", "minutes", "min_ratio"])
    new_run = low["bin_start"].ne(low["bin_end"].shift())
    run_id = new_run.cumsum()
    out = low.groupby(run_id).agg(start=("bin_start", "min"), end=("bin_end", "max"),
                                  min_ratio=("ratio", "min"))
    out["minutes"] = (out["end"] - out["start"]).dt.total_seconds() / 60
    return out[["start", "end", "minutes", "min_ratio"]].reset_index(drop=True)
