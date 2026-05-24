"""Family E5 (Phase E): Move extension from origin. Answers Q13 (R:R compressed).

Why this matters: a long signal 5 bars after the move started has a different
risk:reward profile than the same signal 50 bars in. Late-entry trades in
extended moves offer compressed reward with expanded downside. The model
should explicitly know "how far into this move are we" so it can penalize
late entries.

We define "origin" as the most recent swing high or swing low (whichever is
opposite to current price direction). Distance from origin is in ATR units.

Features (3 total):
  • ext_dist_from_swing_atr     distance from most recent swing in ATR (always >= 0)
  • ext_bars_from_swing         number of bars since swing formed
  • ext_overextended            1.0 if dist > 2.5 ATR (statistically extended), else 0.0

CAUSALITY: swing detection uses bars ≤ T-1 only (rolling max/min of past
window). At bar T the swing reference is the most recent confirmed swing.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


EPS = 1e-9


def _atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    prev_close = df["close"].shift(1)
    tr = pd.concat(
        [(df["high"] - df["low"]).abs(),
         (df["high"] - prev_close).abs(),
         (df["low"] - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    return tr.rolling(n, min_periods=2).mean()


def compute(df_1m: pd.DataFrame, swing_lookback: int = 30) -> pd.DataFrame:
    close = df_1m["close"].to_numpy(dtype=np.float64)
    high = df_1m["high"].to_numpy(dtype=np.float64)
    low = df_1m["low"].to_numpy(dtype=np.float64)
    n = len(close)

    # Rolling extremes over [t-lookback, t-1] (shift excludes current bar)
    high_past = pd.Series(high).shift(1).rolling(swing_lookback, min_periods=2).max().to_numpy()
    low_past = pd.Series(low).shift(1).rolling(swing_lookback, min_periods=2).min().to_numpy()
    high_past_argmax = pd.Series(high).shift(1).rolling(swing_lookback, min_periods=2).apply(
        lambda x: float(np.argmax(x[::-1])), raw=True
    ).to_numpy()
    low_past_argmin = pd.Series(low).shift(1).rolling(swing_lookback, min_periods=2).apply(
        lambda x: float(np.argmin(x[::-1])), raw=True
    ).to_numpy()

    atr = _atr(df_1m, n=14).replace(0, np.nan).ffill().fillna(1.0).to_numpy()

    # Pick the opposite-side swing: if price is currently up from session-ish midpoint,
    # measure from the recent swing LOW (origin of move); else from recent swing HIGH.
    # Simpler heuristic: measure from whichever extreme is farther — i.e., the swing
    # that price has moved away from.
    dist_from_high = np.abs(close - high_past) / np.maximum(atr, EPS)
    dist_from_low = np.abs(close - low_past) / np.maximum(atr, EPS)
    use_low = dist_from_low > dist_from_high  # true if move is up from swing low
    dist_from_swing = np.where(use_low, dist_from_low, dist_from_high)
    bars_from_swing = np.where(use_low, low_past_argmin, high_past_argmax)
    bars_from_swing = np.nan_to_num(bars_from_swing, nan=0.0).astype(np.float32)

    overextended = (dist_from_swing > 2.5).astype(np.float32)

    return pd.DataFrame({
        "ext_dist_from_swing_atr": dist_from_swing.astype(np.float32),
        "ext_bars_from_swing": bars_from_swing,
        "ext_overextended": overextended,
    }, index=df_1m.index)


FEATURE_COLUMNS = (
    "ext_dist_from_swing_atr",
    "ext_bars_from_swing",
    "ext_overextended",
)
