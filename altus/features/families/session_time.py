"""Family 1: Session + time-of-day features.

Why this matters: the same OHLCV setup behaves differently at 10:00 UTC (NY open
volatility) vs 13:00 UTC (lunch chop) vs 02:00 UTC (Asia thin). Layer 1 currently
has no way to read the clock. These features give it that context.

All features are deterministic from the timestamp. No rolling computation.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def compute(df_1m: pd.DataFrame) -> pd.DataFrame:
    """Compute session + time features aligned to df_1m.index.

    Returns 6 columns: time_hour_sin, time_hour_cos, time_dow_sin, time_dow_cos,
    session_is_ny, session_is_overnight.
    """
    idx = df_1m.index
    hour_utc = idx.hour + idx.minute / 60.0
    dow = idx.dayofweek.astype(np.float32)

    # Cyclic encoding — avoids the 23->0 discontinuity a raw hour feature would have.
    hour_sin = np.sin(2 * np.pi * hour_utc / 24.0)
    hour_cos = np.cos(2 * np.pi * hour_utc / 24.0)
    dow_sin = np.sin(2 * np.pi * dow / 7.0)
    dow_cos = np.cos(2 * np.pi * dow / 7.0)

    # Session indicators (UTC).
    # NY RTH:    13:30-20:00 UTC (covers EDT 09:30-16:00; close enough for EST too).
    # Overnight: 22:00-08:00 UTC (post-NY close + Asia + early London).
    # London/Europe open is the implicit "other" — neither NY nor overnight.
    h = hour_utc
    session_is_ny = ((h >= 13.5) & (h < 20.0)).astype(np.float32)
    session_is_overnight = ((h >= 22.0) | (h < 8.0)).astype(np.float32)

    return pd.DataFrame(
        {
            "time_hour_sin": hour_sin,
            "time_hour_cos": hour_cos,
            "time_dow_sin": dow_sin,
            "time_dow_cos": dow_cos,
            "session_is_ny": session_is_ny,
            "session_is_overnight": session_is_overnight,
        },
        index=idx,
    )


FEATURE_COLUMNS = (
    "time_hour_sin",
    "time_hour_cos",
    "time_dow_sin",
    "time_dow_cos",
    "session_is_ny",
    "session_is_overnight",
)
