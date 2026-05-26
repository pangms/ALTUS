"""Shared helpers for the setup-detection families (A1-A8).

Setup families have a lot of common machinery: ATR computation, fractal swing
detection, session-window checks, level-proximity tests, rolling stats. Putting
them here keeps the 8 setup_*.py files DRY and consistent.

All helpers are CAUSAL by default — features at row T use only data <T.
Where a helper needs a window of future bars to "confirm" something (e.g.,
fractal swing detection), the function returns the confirmed value at the
LATER bar where confirmation is complete (the standard "lookback-with-delay"
pattern).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


EPS = 1e-9


# Session boundaries in UTC. Anchored to EDT-equivalent NY RTH; ±1h drift
# across DST is acceptable for setup-time-window gating (see ARCHITECTURE.md).
NY_RTH_START_UTC = 13.5    # 09:30 ET
NY_RTH_END_UTC = 20.0      # 16:00 ET


def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    """Average true range over n bars. Standard implementation."""
    prev_close = df["close"].shift(1)
    tr = pd.concat([
        (df["high"] - df["low"]).abs(),
        (df["high"] - prev_close).abs(),
        (df["low"] - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(n, min_periods=2).mean()


def atr_safe(df: pd.DataFrame, n: int = 14, floor: float = 0.5) -> np.ndarray:
    """ATR with a floor to prevent divide-by-zero in distance features."""
    return atr(df, n).replace(0, np.nan).ffill().fillna(1.0).clip(lower=floor).to_numpy(dtype=np.float64)


def hour_of_day_utc(index: pd.DatetimeIndex) -> np.ndarray:
    """Fractional hour-of-day in UTC for each timestamp."""
    return (index.hour + index.minute / 60.0).to_numpy()


def in_ny_rth(index: pd.DatetimeIndex) -> np.ndarray:
    """Boolean array: True when the timestamp is inside NY RTH."""
    h = hour_of_day_utc(index)
    return (h >= NY_RTH_START_UTC) & (h < NY_RTH_END_UTC)


def in_ny_rth_first_hour(index: pd.DatetimeIndex) -> np.ndarray:
    """First 30 minutes of NY RTH (13:30 to 14:00 UTC EDT)."""
    h = hour_of_day_utc(index)
    return (h >= NY_RTH_START_UTC) & (h < NY_RTH_START_UTC + 0.5)


def in_ny_rth_last_30min(index: pd.DatetimeIndex) -> np.ndarray:
    """Last 30 minutes of NY RTH (19:30 to 20:00 UTC EDT)."""
    h = hour_of_day_utc(index)
    return (h >= NY_RTH_END_UTC - 0.5) & (h < NY_RTH_END_UTC)


def causal_rolling_max(s: pd.Series, window: int, min_periods: int | None = None) -> pd.Series:
    """Rolling max strictly over past `window` bars (causal via shift(1))."""
    return s.rolling(window, min_periods=min_periods or 2).max().shift(1)


def causal_rolling_min(s: pd.Series, window: int, min_periods: int | None = None) -> pd.Series:
    """Rolling min strictly over past `window` bars (causal via shift(1))."""
    return s.rolling(window, min_periods=min_periods or 2).min().shift(1)


def causal_rolling_mean(s: pd.Series, window: int, min_periods: int | None = None) -> pd.Series:
    """Rolling mean strictly over past `window` bars (causal via shift(1))."""
    return s.rolling(window, min_periods=min_periods or 2).mean().shift(1)


def detect_fractal_swings(
    highs: np.ndarray, lows: np.ndarray, lookback: int = 5
) -> tuple[np.ndarray, np.ndarray]:
    """Fractal swing detection. Returns (swing_high_idx_for_each_bar,
    swing_low_idx_for_each_bar). A bar i is a swing high if its high is the
    maximum across [i-lookback, i+lookback]. Confirmation needs +lookback
    future bars, so the "confirmed at bar j" output places the swing-source
    bar's INDEX at position j = source_idx + lookback.

    For causal usage: at bar T, the most recently CONFIRMED swing is found in
    the output array at position T (the array slot stores the source-bar
    index of the most recent confirmed swing).
    """
    n = len(highs)
    last_confirmed_swing_high = np.full(n, -1, dtype=np.int64)
    last_confirmed_swing_low = np.full(n, -1, dtype=np.int64)

    cur_h = -1
    cur_l = -1
    for i in range(lookback, n - lookback):
        window_h = highs[i - lookback : i + lookback + 1]
        window_l = lows[i - lookback : i + lookback + 1]
        if highs[i] >= window_h.max():
            cur_h = i
        elif lows[i] <= window_l.min():
            cur_l = i
        # Confirmed at i + lookback
        conf_pos = i + lookback
        if conf_pos < n:
            last_confirmed_swing_high[conf_pos] = cur_h
            last_confirmed_swing_low[conf_pos] = cur_l

    # Forward-fill so every bar has the most-recently-confirmed swing
    # available. Numpy doesn't have a built-in ffill — use a loop variant.
    for i in range(1, n):
        if last_confirmed_swing_high[i] == -1:
            last_confirmed_swing_high[i] = last_confirmed_swing_high[i - 1]
        if last_confirmed_swing_low[i] == -1:
            last_confirmed_swing_low[i] = last_confirmed_swing_low[i - 1]
    return last_confirmed_swing_high, last_confirmed_swing_low


def session_id(index: pd.DatetimeIndex) -> np.ndarray:
    """Integer session counter. Increments on each NY RTH start. Useful for
    grouping operations within a session (e.g., session-anchored cumulative
    stats).
    """
    in_rth = in_ny_rth(index)
    starts = np.zeros_like(in_rth, dtype=bool)
    starts[1:] = in_rth[1:] & ~in_rth[:-1]
    return np.cumsum(starts).astype(np.int64)


def bars_since_session_start(index: pd.DatetimeIndex) -> np.ndarray:
    """Number of bars elapsed since the start of the current NY RTH session.
    Returns 0 when not in RTH or at first RTH bar.
    """
    in_rth = in_ny_rth(index)
    sid = session_id(index)
    out = np.zeros(len(index), dtype=np.int64)
    if not in_rth.any():
        return out
    # Compute within each session: cumulative count of in_rth bars
    df_tmp = pd.DataFrame({"sid": sid, "in": in_rth.astype(int)})
    out = df_tmp.groupby("sid").cumcount().to_numpy()
    # Zero out bars not in RTH
    out = np.where(in_rth, out, 0)
    return out


def fresh_age_decay(age_bars: np.ndarray, half_life_bars: int = 30) -> np.ndarray:
    """Exponential freshness decay: 1.0 at age 0, 0.5 at half_life, 0.25 at
    2× half_life. Used by setup-strength scoring to prefer fresh signals."""
    return np.power(0.5, age_bars / max(half_life_bars, 1))


def nearest_level_distance(
    close: np.ndarray, levels: np.ndarray, atr_safe_arr: np.ndarray
) -> np.ndarray:
    """For each bar, signed distance to the nearest level in ATR units.
    Negative = level is above; positive = level is below. NaN-safe.
    """
    if len(levels) == 0:
        return np.zeros(len(close), dtype=np.float32)
    # Broadcast: |close - level| for each level, pick the smallest
    diffs = levels[None, :] - close[:, None]  # (n_bars, n_levels)
    abs_diffs = np.abs(diffs)
    nearest_idx = np.argmin(abs_diffs, axis=1)
    nearest_diffs = diffs[np.arange(len(close)), nearest_idx]
    return (nearest_diffs / atr_safe_arr).astype(np.float32)


def clip_clamp(x: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """Clamp values + replace NaN/inf with 0. Standard sanitizer."""
    return np.nan_to_num(np.clip(x, lo, hi), nan=0.0, posinf=hi, neginf=lo).astype(np.float32)
