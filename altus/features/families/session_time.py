"""Family 1: Session + time-of-day features.

Why this matters: the same OHLCV setup behaves differently at 10:00 UTC (NY open
volatility) vs 13:00 UTC (lunch chop) vs 02:00 UTC (Asia thin). Layer 1 currently
has no way to read the clock. These features give it that context.

The user trades all three majors (Asia / London / NY) and wants L1+L2 to
*learn* that Asia and London are typically lower quality (i.e., require a
higher conviction bar) — soft penalty via richer time encoding rather than
a hard session gate. The surfer principle: regime is context, not a gate.

All features are deterministic from the timestamp. No rolling computation.

Sessions used (UTC, approximate, ignores DST shifts of ±1h):
  - Asia        :  00:00 - 07:00 UTC  (Tokyo + Sydney; thin, mean-reverting)
  - London/Eur  :  07:00 - 13:00 UTC  (London open + pre-NY)
  - NY pre-mkt  :  12:00 - 13:30 UTC  (US futures-only liquidity ramp)
  - NY RTH      :  13:30 - 20:00 UTC  (US cash session; covers EDT 09:30-16:00)
  - Post-close  :  20:00 - 22:00 UTC  (US after-hours quieting)
  - Overnight   :  22:00 - 00:00 UTC + 00:00-08:00 (existing flag, kept)

DST note: timestamps are UTC and we deliberately do NOT shift session
boundaries with EDT/EST. Most published US-session bias is anchored to
9:30-16:00 ET, which drifts ±1h vs UTC across the year. Treating that drift
as part of the seasonality the model has to learn is fine for a feature
(L1 can pick it up via hour_sin/cos × dow_sin/cos interactions); we only
DST-correct for L3 hard rules (EoD flatten), where being 1h off matters.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


# Session boundaries (UTC). Keep here as constants so other modules can reuse.
ASIA_START_UTC = 0.0
ASIA_END_UTC = 7.0
LONDON_START_UTC = 7.0
LONDON_END_UTC = 13.0
NY_PREMKT_START_UTC = 12.0
NY_RTH_START_UTC = 13.5     # 09:30 ET (EDT)
NY_RTH_END_UTC = 20.0       # 16:00 ET (EDT)
POST_CLOSE_END_UTC = 22.0


def compute(df_1m: pd.DataFrame) -> pd.DataFrame:
    """Compute session + time features aligned to df_1m.index.

    Returns 12 columns. The first 6 are the original v1 set (preserved for
    backward compatibility with any old checkpoints / configs); the new 6
    encode session bucket + distance-to-NY-close so the model can learn
    "be more selective in Asia/London" and "tighten up near close".
    """
    idx = df_1m.index
    hour_utc = idx.hour + idx.minute / 60.0
    dow = idx.dayofweek.astype(np.float32)

    # ---- v1 columns (kept) ----
    hour_sin = np.sin(2 * np.pi * hour_utc / 24.0)
    hour_cos = np.cos(2 * np.pi * hour_utc / 24.0)
    dow_sin = np.sin(2 * np.pi * dow / 7.0)
    dow_cos = np.cos(2 * np.pi * dow / 7.0)
    h = hour_utc
    session_is_ny = ((h >= NY_RTH_START_UTC) & (h < NY_RTH_END_UTC)).astype(np.float32)
    session_is_overnight = ((h >= POST_CLOSE_END_UTC) | (h < ASIA_END_UTC + 1.0)).astype(np.float32)

    # ---- v2 additions ----
    session_is_asia = ((h >= ASIA_START_UTC) & (h < ASIA_END_UTC)).astype(np.float32)
    session_is_london = ((h >= LONDON_START_UTC) & (h < LONDON_END_UTC)).astype(np.float32)
    # NY pre-market: futures liquidity ramp before cash open.
    session_is_ny_premkt = ((h >= NY_PREMKT_START_UTC) & (h < NY_RTH_START_UTC)).astype(np.float32)

    # Distance-to-NY-close in hours, signed:
    #   - positive when before close (8.5h at Asia start → 0 at close)
    #   - clamped at 0 after close (model treats post-close as "expired window")
    # This is the highest-signal-per-feature time encoding for the EoD pattern:
    # most US-session edge concentrates in the final hour, and the L3 layer will
    # also flatten 20min before close, so L1/L2 has a reason to bias toward
    # signals earlier in the session.
    hours_to_close = np.clip(NY_RTH_END_UTC - h, 0.0, 24.0).astype(np.float32)
    hours_since_ny_open = np.clip(h - NY_RTH_START_UTC, 0.0, 24.0).astype(np.float32)
    # Within-NY-session fractional position (NaN-safe — 0 outside session):
    rth_dur = NY_RTH_END_UTC - NY_RTH_START_UTC
    ny_frac = np.where(
        session_is_ny.astype(bool),
        (h - NY_RTH_START_UTC) / rth_dur,
        0.0,
    ).astype(np.float32)

    return pd.DataFrame(
        {
            "time_hour_sin": hour_sin,
            "time_hour_cos": hour_cos,
            "time_dow_sin": dow_sin,
            "time_dow_cos": dow_cos,
            "session_is_ny": session_is_ny,
            "session_is_overnight": session_is_overnight,
            # v2 additions:
            "session_is_asia": session_is_asia,
            "session_is_london": session_is_london,
            "session_is_ny_premkt": session_is_ny_premkt,
            "hours_to_ny_close": hours_to_close,
            "hours_since_ny_open": hours_since_ny_open,
            "ny_session_fraction": ny_frac,
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
    "session_is_asia",
    "session_is_london",
    "session_is_ny_premkt",
    "hours_to_ny_close",
    "hours_since_ny_open",
    "ny_session_fraction",
)
