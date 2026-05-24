"""Family E7 (Phase E): Session anatomy. Answers Q24 (where in the session's rhythm).

Why this matters: markets have predictable intraday rhythm. Cash open (NY
13:30-14:30 UTC) volatility ≠ midday lull (16:00-19:00 UTC) ≠ closing hour
(19:00-20:00 UTC) ≠ overnight (22:00-08:00 UTC). The session_time family
captures *what time it is*; this one captures *where in the session's natural
arc we are* — which phase of the day's auction process.

Features (5 total):
  • sa_mins_from_cash_open    minutes since 13:30 UTC NY open (clipped to ±390)
  • sa_mins_to_cash_close     minutes until 20:00 UTC NY close (clipped to ±390)
  • sa_is_opening_hour        1.0 during 13:30-14:30 UTC (first hour of cash session)
  • sa_is_closing_hour        1.0 during 19:00-20:00 UTC (last hour of cash session)
  • sa_dow_friday_close       1.0 during Friday last 2 hours (15:00-20:00 UTC)
                              — captures end-of-week unwind behavior

CAUSALITY: deterministic from timestamp; no leakage possible.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def compute(df_1m: pd.DataFrame) -> pd.DataFrame:
    idx = df_1m.index
    hour_utc = idx.hour + idx.minute / 60.0
    dow = idx.dayofweek

    # NY cash session: 13:30 → 20:00 UTC (390 minutes)
    mins_in_session = (hour_utc - 13.5) * 60.0
    mins_in_session = np.clip(mins_in_session, -390.0, 390.0)
    mins_to_close = (20.0 - hour_utc) * 60.0
    mins_to_close = np.clip(mins_to_close, -390.0, 390.0)

    is_opening = ((hour_utc >= 13.5) & (hour_utc < 14.5)).astype(np.float32)
    is_closing = ((hour_utc >= 19.0) & (hour_utc < 20.0)).astype(np.float32)

    # Friday afternoon (NY): dow == 4 AND hour 15:00-20:00 UTC
    friday_close = ((dow == 4) & (hour_utc >= 15.0) & (hour_utc < 20.0)).astype(np.float32)

    return pd.DataFrame({
        "sa_mins_from_cash_open": mins_in_session.astype(np.float32) / 60.0,  # in hours, range ±6.5
        "sa_mins_to_cash_close": mins_to_close.astype(np.float32) / 60.0,
        "sa_is_opening_hour": is_opening,
        "sa_is_closing_hour": is_closing,
        "sa_dow_friday_close": friday_close,
    }, index=idx)


FEATURE_COLUMNS = (
    "sa_mins_from_cash_open",
    "sa_mins_to_cash_close",
    "sa_is_opening_hour",
    "sa_is_closing_hour",
    "sa_dow_friday_close",
)
