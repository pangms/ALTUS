"""Family 9 (Phase B-2): Liquidity zones — untouched higher-timeframe extremes.

Why this matters: per Smart Money Concepts (SMC) and ICT methodology, price
gravitates toward "untouched" higher-timeframe swing highs and lows because
that's where stop-loss orders accumulate (long stops above prior highs, short
stops below prior lows). These zones act as liquidity targets for institutional
order flow. We quantify them.

An "untouched" zone = a swing high/low on a higher timeframe that price has
NOT returned to since it was formed. Once price revisits the zone, it's
considered "swept" and is no longer untouched.

Features (7 total):
  • dist_above_untouched_4h_atr    distance to nearest untouched 4h high above
  • dist_below_untouched_4h_atr    distance to nearest untouched 4h low below
  • dist_above_untouched_1d_atr    distance to nearest untouched 1d high above
  • dist_below_untouched_1d_atr    distance to nearest untouched 1d low below
  • dist_above_untouched_1w_atr    distance to nearest untouched 1w high above
  • dist_below_untouched_1w_atr    distance to nearest untouched 1w low below
  • closest_untouched_zone_atr     min distance (in ATR) to any untouched zone
                                    in either direction across all 3 TFs

CAUSALITY: at bar T we only see zone formations and touches through bar T-1.
The zone tracking is online (one pass through history) — for each bar, we
update the set of untouched zones based on what's happened before it.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from altus.features.families.trend_hurst import _resample_ohlcv


EPS = 1e-9


def _atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    """Rolling ATR for distance normalization."""
    prev_close = df["close"].shift(1)
    tr = pd.concat(
        [(df["high"] - df["low"]).abs(),
         (df["high"] - prev_close).abs(),
         (df["low"] - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    return tr.rolling(n, min_periods=2).mean()


def _detect_htf_swings(htf_df: pd.DataFrame, lookback: int = 5) -> tuple[np.ndarray, np.ndarray]:
    """Find local swing highs/lows on a higher timeframe.

    A swing high at bar i = high[i] is strictly higher than highs in
    [i-lookback, i+lookback] (excluding i itself). Same for swing low on lows.
    Returns (high_positions, low_positions) as integer bar indices into htf_df.
    """
    highs = htf_df["high"].to_numpy()
    lows = htf_df["low"].to_numpy()
    n = len(highs)
    high_swings: list[int] = []
    low_swings: list[int] = []
    for i in range(lookback, n - lookback):
        if highs[i] > highs[i - lookback : i].max() and highs[i] > highs[i + 1 : i + lookback + 1].max():
            high_swings.append(i)
        if lows[i] < lows[i - lookback : i].min() and lows[i] < lows[i + 1 : i + lookback + 1].min():
            low_swings.append(i)
    return np.array(high_swings, dtype=np.int64), np.array(low_swings, dtype=np.int64)


def _untouched_zones_at_each_1m_bar(
    df_1m: pd.DataFrame,
    htf_min: int,
    htf_label: str,
    swing_lookback: int = 3,
) -> tuple[pd.Series, pd.Series]:
    """Compute (dist_above, dist_below) to nearest untouched HTF swing per 1m bar.

    Walks through history once, maintaining sets of untouched-high and
    untouched-low zones. A zone gets removed once price touches/crosses it.
    """
    htf = _resample_ohlcv(df_1m, htf_min)
    if len(htf) < swing_lookback * 4:
        idx = df_1m.index
        return (
            pd.Series(np.nan, index=idx, name=f"dist_above_untouched_{htf_label}"),
            pd.Series(np.nan, index=idx, name=f"dist_below_untouched_{htf_label}"),
        )

    high_pos, low_pos = _detect_htf_swings(htf, lookback=swing_lookback)

    # Build a chronological list of (htf_timestamp, price, kind)
    events: list[tuple[pd.Timestamp, float, str]] = []
    for p in high_pos:
        # Swing becomes "known" at the bar that closes `swing_lookback` AFTER the swing point
        confirm_idx = min(p + swing_lookback, len(htf) - 1)
        events.append((htf.index[confirm_idx], float(htf["high"].iloc[p]), "high"))
    for p in low_pos:
        confirm_idx = min(p + swing_lookback, len(htf) - 1)
        events.append((htf.index[confirm_idx], float(htf["low"].iloc[p]), "low"))
    events.sort(key=lambda e: e[0])

    # Walk 1m bars, maintain untouched-zone sets
    untouched_highs: list[float] = []
    untouched_lows: list[float] = []
    event_ptr = 0
    n_events = len(events)

    dist_above_arr = np.full(len(df_1m), np.nan, dtype=np.float32)
    dist_below_arr = np.full(len(df_1m), np.nan, dtype=np.float32)

    highs_1m = df_1m["high"].to_numpy(dtype=np.float64)
    lows_1m = df_1m["low"].to_numpy(dtype=np.float64)
    closes_1m = df_1m["close"].to_numpy(dtype=np.float64)
    idx_1m = df_1m.index.to_numpy()

    for i in range(len(df_1m)):
        # Add any events whose confirmation time is <= this bar
        while event_ptr < n_events and events[event_ptr][0] <= idx_1m[i]:
            _, price, kind = events[event_ptr]
            if kind == "high":
                untouched_highs.append(price)
            else:
                untouched_lows.append(price)
            event_ptr += 1

        cur_close = closes_1m[i]
        cur_high = highs_1m[i]
        cur_low = lows_1m[i]

        # Sweep: any untouched-high <= cur_high is touched
        untouched_highs = [p for p in untouched_highs if p > cur_high]
        # Any untouched-low >= cur_low is touched
        untouched_lows = [p for p in untouched_lows if p < cur_low]

        # Distance features
        above = [p for p in untouched_highs if p > cur_close]
        below = [p for p in untouched_lows if p < cur_close]
        if above:
            # Nearest above = smallest price strictly greater than cur_close.
            dist_above_arr[i] = min(above) - cur_close
        if below:
            # Nearest below = largest price strictly less than cur_close.
            # Earlier versions used `min(below, key=lambda p: cur_close - p)`
            # which returns the FARTHEST below — caught in 2026-05-24 audit.
            dist_below_arr[i] = cur_close - max(below)

    above_s = pd.Series(dist_above_arr, index=df_1m.index)
    below_s = pd.Series(dist_below_arr, index=df_1m.index)
    return above_s, below_s


NEEDS_RAW_1M = True  # uses _resample_ohlcv → must see clean non-overlapping 1m bars


def compute(df_primary: pd.DataFrame, df_1m: pd.DataFrame | None = None) -> pd.DataFrame:
    """Compute liquidity-zone features. Returns 7 columns.

    Uses raw 1m bars throughout — rolling-primary bars would produce wrong
    high/low extremes for the HTF zone-detection logic.
    """
    if df_1m is None:
        df_1m = df_primary  # back-compat / PRIMARY_WINDOW_MIN=1 path
    atr = _atr(df_1m, n=14).replace(0, np.nan)
    atr_safe = atr.ffill().fillna(1.0)

    out: dict[str, pd.Series] = {}
    htf_specs = [(240, "4h"), (1440, "1d"), (10_080, "1w")]

    aggregate_dists: list[pd.Series] = []
    for htf_min, label in htf_specs:
        above_s, below_s = _untouched_zones_at_each_1m_bar(df_1m, htf_min, label)
        # Cap missing distances at 6 ATR so we don't bias on warmup / no-zones
        above_atr = (above_s / atr_safe).fillna(6.0).clip(upper=6.0)
        below_atr = (below_s / atr_safe).fillna(6.0).clip(upper=6.0)
        out[f"lz_dist_above_{label}_atr"] = above_atr.astype(np.float32)
        out[f"lz_dist_below_{label}_atr"] = below_atr.astype(np.float32)
        aggregate_dists.append(pd.concat([above_atr, below_atr], axis=1).min(axis=1))

    # Closest untouched zone in any direction across all 3 TFs
    out["lz_closest_zone_atr"] = pd.concat(aggregate_dists, axis=1).min(axis=1).astype(np.float32)

    return pd.DataFrame(out, index=df_1m.index)


FEATURE_COLUMNS = (
    "lz_dist_above_4h_atr",
    "lz_dist_below_4h_atr",
    "lz_dist_above_1d_atr",
    "lz_dist_below_1d_atr",
    "lz_dist_above_1w_atr",
    "lz_dist_below_1w_atr",
    "lz_closest_zone_atr",
)
