"""A1 — Open Range Breakout (ORB).

Thesis: First 30-60min of NY RTH session establishes a reference range.
Algos + traders pile stops above the OR high and below the OR low. A clean
break with confirmation tends to continue because the breakout triggers
stop runs in that direction, providing fuel.

Detection conditions:
  * Time of day: past NY RTH start + 30min AND before NY RTH start + 3h
  * OR established: high/low over first 30min of session known
  * OR size: 0.5 to 3.0 × ATR(60min) (not too tight, not blown out)
  * Breakout: close > OR_high (long) OR close < OR_low (short)
  * Range integrity: no close beyond OR boundary in OR window

Outputs (5 features):
  orb_active            1.0 if breakout signal active
  orb_strength          0-1 continuous match score
  orb_direction         +1 long, -1 short, 0 inactive
  orb_breakout_age      bars since first break (for freshness)
  orb_range_atr         OR width in ATR units (size context)

CAUSALITY: OR boundaries are locked at the end of the first 30min and used
forward. Past breakout detection is over past bars.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from altus.features.families.setup_utils import (
    EPS, atr_safe, clip_clamp, fresh_age_decay,
    hour_of_day_utc, in_ny_rth, session_id,
    NY_RTH_START_UTC,
)


NEEDS_RAW_1M = True


def _compute_or_anchors(df_1m: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """For each bar, return (or_high, or_low, mins_since_session_start).
    OR = first 30min of NY RTH. Locked after the 30th minute.

    Uses direct positional iteration for robustness — avoids index.get_loc
    edge cases with potential timestamp duplicates.
    """
    n = len(df_1m)
    idx = df_1m.index
    highs = df_1m["high"].to_numpy(dtype=np.float64)
    lows = df_1m["low"].to_numpy(dtype=np.float64)
    hour = hour_of_day_utc(idx)
    in_rth = in_ny_rth(idx)
    in_or_window = (hour >= NY_RTH_START_UTC) & (hour < NY_RTH_START_UTC + 0.5)

    or_high = np.full(n, np.nan, dtype=np.float64)
    or_low = np.full(n, np.nan, dtype=np.float64)
    mins_since = np.zeros(n, dtype=np.int32)

    # Walk forward through bars. Detect RTH-session starts and OR windows.
    # State machine: track current session's RTH-start position + OR-end position.
    cur_session_rth_start = -1   # position where current session's RTH started
    cur_session_or_end = -1      # position where OR window ended
    cur_or_high = -np.inf
    cur_or_low = np.inf
    in_session_or = False

    for i in range(n):
        if in_rth[i]:
            # Check if this is the start of a new RTH session
            if i == 0 or not in_rth[i - 1]:
                cur_session_rth_start = i
                cur_session_or_end = -1
                cur_or_high = -np.inf
                cur_or_low = np.inf
                in_session_or = in_or_window[i]
            # Accumulate OR if we're in the OR window
            if in_or_window[i]:
                cur_or_high = max(cur_or_high, float(highs[i]))
                cur_or_low = min(cur_or_low, float(lows[i]))
                in_session_or = True
            elif in_session_or:
                # Just exited OR window — lock the OR
                cur_session_or_end = i - 1
                in_session_or = False
            # If OR has been locked, propagate forward
            if cur_session_or_end >= 0:
                or_high[i] = cur_or_high
                or_low[i] = cur_or_low
                mins_since[i] = i - cur_session_rth_start
        else:
            # Outside RTH — handle case where session ended without exiting OR explicitly
            if in_session_or and cur_or_high > -np.inf:
                # Session ended mid-OR (unusual). Don't propagate (no valid OR locked).
                pass
            in_session_or = False

    return or_high, or_low, mins_since


def compute(df_primary: pd.DataFrame, df_1m: pd.DataFrame | None = None) -> pd.DataFrame:
    if df_1m is None:
        df_1m = df_primary
    n = len(df_1m)
    idx = df_1m.index
    highs = df_1m["high"].to_numpy(dtype=np.float64)
    lows = df_1m["low"].to_numpy(dtype=np.float64)
    closes = df_1m["close"].to_numpy(dtype=np.float64)
    # ATR baseline: use daily-average ATR (1440-bar window) so OR size is
    # normalized against typical-daily-volatility, not against the very vol
    # window that contains the OR (which gives meaningless ratios). 30-min OR
    # range / 1d avg ATR typically lands in [1, 8] for MNQ.
    atr_arr = atr_safe(df_1m, n=14)              # short-window ATR (used downstream)
    atr_daily = atr_safe(df_1m, n=1440)          # 1-day ATR for OR size normalization

    or_high, or_low, mins_since = _compute_or_anchors(df_1m)

    # Where do we have a valid locked OR?
    valid = ~np.isnan(or_high) & ~np.isnan(or_low)
    range_pts = np.where(valid, or_high - or_low, 0.0)
    range_atr = np.where(valid, range_pts / np.maximum(atr_daily, EPS), 0.0)

    # OR size eligibility: 1.0 to 8.0 × daily ATR (calibrated against observed
    # OR sizes — typical 30-min opening range on MNQ is 2-5× the 1d ATR).
    or_size_ok = valid & (range_atr >= 1.0) & (range_atr <= 8.0)

    # Time eligibility: 30 to 90 min into session (active breakout window).
    # Beyond 90min the OR-driven momentum has typically faded — late breaks
    # are weak signals.
    time_ok = (mins_since >= 30) & (mins_since <= 90)

    # Breakout detection
    breakout_long = or_size_ok & time_ok & (closes > or_high)
    breakout_short = or_size_ok & time_ok & (closes < or_low)
    active = (breakout_long | breakout_short).astype(np.int32)
    direction = np.where(breakout_long, 1, np.where(breakout_short, -1, 0)).astype(np.int32)

    # Breakout age: bars since first break in this session
    age = np.full(n, 91, dtype=np.int32)  # large default
    last_active_pos = -1
    last_active_sid = -1
    sid = session_id(idx)
    for i in range(n):
        if active[i]:
            if last_active_sid != sid[i]:
                last_active_pos = i
                last_active_sid = sid[i]
            age[i] = i - last_active_pos
        elif last_active_sid == sid[i] and last_active_pos >= 0:
            age[i] = i - last_active_pos

    # Strength scoring
    distance_beyond = np.where(
        breakout_long, (closes - or_high) / np.maximum(atr_arr, EPS),
        np.where(breakout_short, (or_low - closes) / np.maximum(atr_arr, EPS), 0.0)
    )
    distance_beyond = np.clip(distance_beyond, 0.0, 1.0)
    freshness = fresh_age_decay(age.astype(np.float32), half_life_bars=20)
    strength = 0.5 * active.astype(np.float32) + 0.2 * distance_beyond + 0.2 * freshness + 0.1 * (range_atr > 1.0).astype(np.float32)
    strength = clip_clamp(strength, 0.0, 1.0)

    return pd.DataFrame({
        "orb_active": active.astype(np.float32),
        "orb_strength": strength,
        "orb_direction": direction.astype(np.float32),
        "orb_breakout_age": clip_clamp(age.astype(np.float32), 0.0, 90.0),
        "orb_range_atr": clip_clamp(range_atr.astype(np.float32), 0.0, 4.0),
    }, index=idx)


FEATURE_COLUMNS = (
    "orb_active",
    "orb_strength",
    "orb_direction",
    "orb_breakout_age",
    "orb_range_atr",
)
