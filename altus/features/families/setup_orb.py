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
    """
    n = len(df_1m)
    idx = df_1m.index
    highs = df_1m["high"].to_numpy(dtype=np.float64)
    lows = df_1m["low"].to_numpy(dtype=np.float64)
    hour = hour_of_day_utc(idx)
    in_rth = in_ny_rth(idx)
    sid = session_id(idx)

    or_high = np.full(n, np.nan, dtype=np.float64)
    or_low = np.full(n, np.nan, dtype=np.float64)
    mins_since = np.zeros(n, dtype=np.int32)

    # For each session, find the first 30min window and lock OR.
    in_or = (hour >= NY_RTH_START_UTC) & (hour < NY_RTH_START_UTC + 0.5)
    df_tmp = pd.DataFrame({"sid": sid, "in_or": in_or.astype(int),
                            "in_rth": in_rth.astype(int)}, index=idx)

    # Per-session OR high/low from the first 30min
    for s, grp in df_tmp.groupby("sid"):
        or_mask = grp["in_or"].to_numpy().astype(bool)
        rth_mask = grp["in_rth"].to_numpy().astype(bool)
        if not or_mask.any():
            continue
        # Find positions in df_1m where this session's OR is built
        grp_pos = grp.index.map(lambda t: df_1m.index.get_loc(t)).to_numpy()
        or_pos = grp_pos[or_mask]
        if len(or_pos) == 0:
            continue
        sess_or_high = float(np.max(highs[or_pos]))
        sess_or_low = float(np.min(lows[or_pos]))
        # Apply forward to all RTH bars in this session AFTER the OR is locked
        # (i.e., positions past the 30min mark)
        rth_pos = grp_pos[rth_mask]
        if len(rth_pos) == 0:
            continue
        first_or_end_pos = or_pos[-1] + 1 if or_pos[-1] + 1 < n else n
        for p in rth_pos:
            if p > or_pos[-1]:
                or_high[p] = sess_or_high
                or_low[p] = sess_or_low
                mins_since[p] = int(p - rth_pos[0])

    return or_high, or_low, mins_since


def compute(df_primary: pd.DataFrame, df_1m: pd.DataFrame | None = None) -> pd.DataFrame:
    if df_1m is None:
        df_1m = df_primary
    n = len(df_1m)
    idx = df_1m.index
    highs = df_1m["high"].to_numpy(dtype=np.float64)
    lows = df_1m["low"].to_numpy(dtype=np.float64)
    closes = df_1m["close"].to_numpy(dtype=np.float64)
    atr_arr = atr_safe(df_1m, n=60)  # 60-bar ATR for "range vs hourly vol" context

    or_high, or_low, mins_since = _compute_or_anchors(df_1m)

    # Where do we have a valid locked OR?
    valid = ~np.isnan(or_high) & ~np.isnan(or_low)
    range_pts = np.where(valid, or_high - or_low, 0.0)
    range_atr = np.where(valid, range_pts / np.maximum(atr_arr, EPS), 0.0)

    # OR size eligibility: 0.5 to 3.0 ATR
    or_size_ok = valid & (range_atr >= 0.5) & (range_atr <= 3.0)

    # Time eligibility: 30 to 180 min into session
    time_ok = (mins_since >= 30) & (mins_since <= 180)

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
