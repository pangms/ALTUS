"""Family 10 (Phase B-3): Sweep + trap detection.

Why this matters: institutional players hunt liquidity by sweeping recent
highs/lows to trigger retail stops, then reversing. A "sweep" is a wick that
pokes above a recent high and CLOSES BACK BELOW — a stop hunt. A "failed
breakout" is when price closes above a recent high but returns below within
N bars — a trap for breakout traders. Both are classic patterns we'd want
the model to recognize.

We define them mechanically (no human chart-reading) using recent N-bar
extremes as the reference level. The model gets simple per-bar counts of
recent events so it can learn whether to weight signals near these patterns
differently.

Features (5 total):
  • swp_sweep_above_recent     # of sweep-above events in last 10 bars
  • swp_sweep_below_recent     # of sweep-below events in last 10 bars
  • swp_failed_break_above     # of failed-break-above events in last 10 bars
  • swp_failed_break_below     # of failed-break-below events in last 10 bars
  • swp_bars_since_last        bars since most recent sweep or failed break (capped at 100)

CAUSALITY: at bar T, we know if a sweep happened in bars [T-10, T-1] only.
The "level" being swept = max/min of bars [t-30, t-1] for each candidate bar t.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _detect_sweep_events(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    level_lookback: int = 30,
    failed_break_lookback: int = 5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Detect per-bar sweep + failed-break events.

    Returns 4 boolean arrays of length len(close):
      sweep_above, sweep_below, failed_break_above, failed_break_below

    Definitions:
      sweep_above at bar t:
          high[t] > max(high[t-L..t-1]) AND close[t] <= max(high[t-L..t-1])
          (wick poked above the recent high but closed back inside)
      failed_break_above at bar t (signal at bar t):
          some prior bar t-k (1<=k<=K) had close[t-k] > max(high[t-k-L..t-k-1]),
          and now close[t] <= max(high[t-k-L..t-k-1])
          (a true close above the level that has since reverted below it)
    """
    n = len(close)
    swp_above = np.zeros(n, dtype=bool)
    swp_below = np.zeros(n, dtype=bool)
    fb_above = np.zeros(n, dtype=bool)
    fb_below = np.zeros(n, dtype=bool)

    if n < level_lookback + failed_break_lookback + 1:
        return swp_above, swp_below, fb_above, fb_below

    # Precompute rolling max/min of HIGH/LOW over [t-L..t-1]
    # Note: must exclude bar t itself, hence shift by 1
    high_shifted = np.concatenate([[np.nan], high[:-1]])
    low_shifted = np.concatenate([[np.nan], low[:-1]])
    # Rolling window over the shifted array gives [t-L..t-1] when read at t
    # Use a simple loop since N is bounded; vectorize if performance bites
    for t in range(level_lookback + 1, n):
        level_high = high_shifted[t - level_lookback + 1 : t + 1].max()
        level_low = low_shifted[t - level_lookback + 1 : t + 1].min()
        if np.isnan(level_high) or np.isnan(level_low):
            continue
        # Sweep above: wick above prior high, close back below
        if high[t] > level_high and close[t] <= level_high:
            swp_above[t] = True
        # Sweep below: wick below prior low, close back above
        if low[t] < level_low and close[t] >= level_low:
            swp_below[t] = True

    # Failed-break above: check if any of last K bars broke and we've now reverted
    # At bar t, look at each t-k for k in [1, K]:
    #   if close[t-k] > recent_level_high(at t-k) AND close[t] <= that same level
    for t in range(level_lookback + failed_break_lookback + 1, n):
        for k in range(1, failed_break_lookback + 1):
            tk = t - k
            level_high_at_tk = high_shifted[tk - level_lookback + 1 : tk + 1].max()
            if not np.isnan(level_high_at_tk) and close[tk] > level_high_at_tk and close[t] <= level_high_at_tk:
                fb_above[t] = True
                break
        for k in range(1, failed_break_lookback + 1):
            tk = t - k
            level_low_at_tk = low_shifted[tk - level_lookback + 1 : tk + 1].min()
            if not np.isnan(level_low_at_tk) and close[tk] < level_low_at_tk and close[t] >= level_low_at_tk:
                fb_below[t] = True
                break

    return swp_above, swp_below, fb_above, fb_below


def compute(
    df_1m: pd.DataFrame,
    level_lookback: int = 30,
    recent_window: int = 10,
    failed_break_lookback: int = 5,
) -> pd.DataFrame:
    """Compute sweep + trap features. Returns 5 columns."""
    high = df_1m["high"].to_numpy(dtype=np.float64)
    low = df_1m["low"].to_numpy(dtype=np.float64)
    close = df_1m["close"].to_numpy(dtype=np.float64)
    n = len(close)

    swp_above, swp_below, fb_above, fb_below = _detect_sweep_events(
        high, low, close,
        level_lookback=level_lookback,
        failed_break_lookback=failed_break_lookback,
    )

    # Rolling count over `recent_window` (right-aligned, excludes current bar)
    def _rolling_count(events: np.ndarray) -> np.ndarray:
        s = pd.Series(events.astype(np.float32))
        return s.rolling(recent_window, min_periods=1).sum().to_numpy(dtype=np.float32)

    swp_above_recent = _rolling_count(swp_above)
    swp_below_recent = _rolling_count(swp_below)
    fb_above_recent = _rolling_count(fb_above)
    fb_below_recent = _rolling_count(fb_below)

    # Bars since last sweep or failed-break event of ANY kind
    any_event = swp_above | swp_below | fb_above | fb_below
    bars_since = np.full(n, 100.0, dtype=np.float32)
    last_event = -1
    for t in range(n):
        if any_event[t]:
            last_event = t
        if last_event >= 0:
            bars_since[t] = min(t - last_event, 100)

    return pd.DataFrame({
        "swp_sweep_above_recent": swp_above_recent,
        "swp_sweep_below_recent": swp_below_recent,
        "swp_failed_break_above": fb_above_recent,
        "swp_failed_break_below": fb_below_recent,
        "swp_bars_since_last": bars_since,
    }, index=df_1m.index)


FEATURE_COLUMNS = (
    "swp_sweep_above_recent",
    "swp_sweep_below_recent",
    "swp_failed_break_above",
    "swp_failed_break_below",
    "swp_bars_since_last",
)
