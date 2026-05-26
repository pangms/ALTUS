"""A4 — Trend Pullback Continuation.

Thesis: In a confirmed multi-TF trend, a pullback to the 8/21 EMA gives
better entry price for a continuation trade.

Detection conditions:
  * Confirmed trend: long-only EMAs aligned + last 20 bars all higher_high+higher_low pattern (or mirror)
  * In pullback: close < EMA(8) by 0.3-1.5 ATR (for long) AND close > EMA(21)
  * Not too deep: retracement 30-62% of most recent swing
  * Momentum oversold-in-trend: 1m RSI(14) between 30-50 for long

Outputs (5 features):
  spb_active             1.0 if pullback setup conditions met
  spb_strength           0-1 continuous match score
  spb_direction          +1 long (bull trend), -1 short (bear trend)
  spb_pullback_depth     normalized retracement [0,1]
  spb_dist_to_ema21_atr  signed distance to EMA(21) in ATR units
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from altus.features.families.setup_utils import EPS, atr_safe, clip_clamp


def _rsi(close: pd.Series, n: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(n, min_periods=n).mean()
    loss = (-delta.clip(upper=0)).rolling(n, min_periods=n).mean()
    rs = gain / (loss + EPS)
    return (100 - 100 / (1 + rs))


def _detect_recent_swing_extreme(highs: np.ndarray, lows: np.ndarray, n: int, lookback: int = 60) -> tuple[float, float]:
    """Most recent swing high and low in lookback window (looking BACK from bar n)."""
    start = max(0, n - lookback)
    if n <= start:
        return float("nan"), float("nan")
    return float(np.max(highs[start:n])), float(np.min(lows[start:n]))


def compute(df_primary: pd.DataFrame, df_1m: pd.DataFrame | None = None) -> pd.DataFrame:
    if df_1m is None:
        df_1m = df_primary
    n = len(df_1m)
    idx = df_1m.index
    closes = df_1m["close"].to_numpy(dtype=np.float64)
    highs = df_1m["high"].to_numpy(dtype=np.float64)
    lows = df_1m["low"].to_numpy(dtype=np.float64)
    atr_arr = atr_safe(df_1m, n=14)

    close_s = df_1m["close"]
    # EMAs — causal by nature (only uses past)
    ema8 = close_s.ewm(span=8, adjust=False).mean().shift(1).to_numpy()
    ema21 = close_s.ewm(span=21, adjust=False).mean().shift(1).to_numpy()
    ema50 = close_s.ewm(span=50, adjust=False).mean().shift(1).to_numpy()
    rsi = _rsi(close_s, n=14).shift(1).fillna(50.0).to_numpy()

    # Trend confirmation: all EMAs in order
    bull_trend = (ema8 > ema21) & (ema21 > ema50)
    bear_trend = (ema8 < ema21) & (ema21 < ema50)

    # Distance to EMAs in ATR units
    dist_ema8 = (closes - ema8) / np.maximum(atr_arr, EPS)
    dist_ema21 = (closes - ema21) / np.maximum(atr_arr, EPS)

    # Long pullback condition: bull trend + close below EMA8 by 0.3-1.5 ATR + above EMA21 + RSI 30-50
    long_pullback = (
        bull_trend &
        (dist_ema8 <= -0.3) & (dist_ema8 >= -1.5) &
        (dist_ema21 >= -0.5) &
        (rsi >= 30.0) & (rsi <= 50.0)
    )
    short_pullback = (
        bear_trend &
        (dist_ema8 >= 0.3) & (dist_ema8 <= 1.5) &
        (dist_ema21 <= 0.5) &
        (rsi >= 50.0) & (rsi <= 70.0)
    )

    active = (long_pullback | short_pullback).astype(np.int32)
    direction = np.where(long_pullback, 1, np.where(short_pullback, -1, 0)).astype(np.int32)

    # Pullback depth: how far back from recent swing
    pullback_depth = np.zeros(n, dtype=np.float32)
    for i in range(60, n):
        if not active[i]:
            continue
        swing_h, swing_l = _detect_recent_swing_extreme(highs, lows, i, lookback=60)
        if direction[i] == 1 and swing_h > 0 and swing_l > 0 and swing_h > swing_l:
            # Retracement from swing high
            range_swing = swing_h - swing_l
            if range_swing > EPS:
                depth = (swing_h - closes[i]) / range_swing
                pullback_depth[i] = float(np.clip(depth, 0.0, 1.0))
        elif direction[i] == -1 and swing_h > 0 and swing_l > 0 and swing_h > swing_l:
            range_swing = swing_h - swing_l
            if range_swing > EPS:
                depth = (closes[i] - swing_l) / range_swing
                pullback_depth[i] = float(np.clip(depth, 0.0, 1.0))

    # Strength: combination of trend strength + sweet-spot retracement + ema proximity
    trend_strength = clip_clamp(np.abs(ema8 - ema21) / np.maximum(atr_arr, EPS), 0.0, 2.0) / 2.0
    sweet_spot = clip_clamp(1.0 - np.abs(pullback_depth - 0.5) * 2.0, 0.0, 1.0)
    ema_proximity = clip_clamp(1.0 - np.abs(dist_ema8) / 1.5, 0.0, 1.0).astype(np.float32)
    strength = (0.4 * active.astype(np.float32) + 0.25 * trend_strength + 0.20 * sweet_spot + 0.15 * ema_proximity)
    strength = clip_clamp(strength, 0.0, 1.0)

    return pd.DataFrame({
        "spb_active": active.astype(np.float32),
        "spb_strength": strength,
        "spb_direction": direction.astype(np.float32),
        "spb_pullback_depth": pullback_depth,
        "spb_dist_to_ema21_atr": clip_clamp(dist_ema21.astype(np.float32), -3.0, 3.0),
    }, index=idx)


FEATURE_COLUMNS = (
    "spb_active",
    "spb_strength",
    "spb_direction",
    "spb_pullback_depth",
    "spb_dist_to_ema21_atr",
)
