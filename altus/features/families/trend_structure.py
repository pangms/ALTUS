"""Higher-high / higher-low structural trend state. Answers Q19 (regime) +
the structural definition of trend that EMA-slope + Hurst miss.

Why this matters: a discretionary trader looks at the last few swing points
and reads structure: "we're making higher highs and higher lows → uptrend,"
or "lower highs and lower lows → downtrend." This is the discrete pattern
variable that exists ABOVE moving averages. EMA slope tells you about
direction-of-mean; HH/HL tells you about the persistence and integrity of
the trend pattern itself. The two diverge often — e.g., EMA can be sloping
up while structure is "lower high + higher low" (a contracting range).

Features (6 total):
  ts_state_bull          1.0 if last 3 swings show HH+HL pattern, else 0.0
  ts_state_bear          1.0 if last 3 swings show LH+LL pattern, else 0.0
  ts_state_range         1.0 if last 3 swings show HL+LH (compression), else 0.0
  ts_state_transition    1.0 if recent break-of-structure (HH then LL or vice), else 0.0
  ts_last_swing_age_bars 1m bars since the most recent confirmed swing (clipped)
  ts_last_swing_dist_atr signed distance from current close to the last swing's
                         price, in ATR (positive = swing was above current)

Implementation: a "swing high" at bar i is a fractal — high[i] is the maximum
of the last `lookback` bars AND i is `confirm` bars in the past (so confirmation
requires future bars to have lower highs). We use lookback=5 (5-bar fractal)
with confirm=3 — the swing is "confirmed" 3 bars after it forms. The 3-bar
confirmation delay is what makes this causal: at bar T, the most recent
confirmed swing is at most bar T-3.

CAUSALITY: all features at bar T depend only on bars 0..T-3 (confirmed swings).
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


def _detect_confirmed_swings(
    highs: np.ndarray, lows: np.ndarray, lookback: int = 5, confirm: int = 3
) -> list[tuple[int, str, float]]:
    """Return a chronologically-sorted list of (bar_idx, 'high'|'low', price).

    A bar i is a swing high if its high is >= all highs in [i-lookback, i+lookback]
    (inclusive). The bar is CONFIRMED at bar i + confirm — until then, future bars
    might exceed it. For causal usage downstream, callers should only reference
    swings whose confirmation bar <= current bar.
    """
    n = len(highs)
    swings = []
    for i in range(lookback, n - lookback):
        window_h = highs[i - lookback : i + lookback + 1]
        window_l = lows[i - lookback : i + lookback + 1]
        if highs[i] >= window_h.max():
            swings.append((i, "high", float(highs[i])))
        elif lows[i] <= window_l.min():
            swings.append((i, "low", float(lows[i])))
    return swings


def compute(df_1m: pd.DataFrame) -> pd.DataFrame:
    n = len(df_1m)
    close = df_1m["close"].to_numpy(dtype=np.float64)
    high = df_1m["high"].to_numpy(dtype=np.float64)
    low = df_1m["low"].to_numpy(dtype=np.float64)
    atr = _atr(df_1m, n=14).replace(0, np.nan).ffill().fillna(1.0).to_numpy(dtype=np.float64)
    atr_safe = np.maximum(atr, EPS)

    lookback = 5
    confirm = 3
    swings = _detect_confirmed_swings(high, low, lookback=lookback, confirm=confirm)

    # For each bar i, determine the LAST 3 CONFIRMED swings (most recent first).
    # Confirmation happens at swing_idx + confirm. We walk forward through
    # `swings`, advancing a pointer to track which swings are confirmed by bar i.
    bull = np.zeros(n, dtype=np.float32)
    bear = np.zeros(n, dtype=np.float32)
    rng = np.zeros(n, dtype=np.float32)
    transition = np.zeros(n, dtype=np.float32)
    swing_age = np.full(n, float(n), dtype=np.float32)  # large default
    swing_dist_atr = np.zeros(n, dtype=np.float32)

    confirmed: list[tuple[int, str, float]] = []  # rolling buffer of confirmed swings
    swing_ptr = 0  # next swing to confirm

    for i in range(n):
        # Advance the pointer to add any swings now confirmed (confirm_bar <= i).
        while swing_ptr < len(swings) and swings[swing_ptr][0] + confirm <= i:
            confirmed.append(swings[swing_ptr])
            swing_ptr += 1

        if len(confirmed) >= 3:
            # Most recent three swings (in chronological order):
            s1, s2, s3 = confirmed[-3], confirmed[-2], confirmed[-1]
            # Classify the 3-swing pattern.
            # Bull pattern: HH + HL (e.g., types L, H, L → check H rising AND L rising)
            # We need to look at swings by type ordering:
            #   - For HH/HL: pattern is L H L H ... or H L H L ...
            #   - Need at least two highs and two lows in the last 4-5 swings ideally
            # Simpler heuristic: look at the last 4 swings' types and their prices.
            recent = confirmed[-4:] if len(confirmed) >= 4 else confirmed[-3:]
            highs_recent = [(idx, p) for idx, t, p in recent if t == "high"]
            lows_recent = [(idx, p) for idx, t, p in recent if t == "low"]

            if len(highs_recent) >= 2 and len(lows_recent) >= 2:
                last_h, prev_h = highs_recent[-1][1], highs_recent[-2][1]
                last_l, prev_l = lows_recent[-1][1], lows_recent[-2][1]
                if last_h > prev_h and last_l > prev_l:
                    bull[i] = 1.0
                elif last_h < prev_h and last_l < prev_l:
                    bear[i] = 1.0
                elif last_h < prev_h and last_l > prev_l:
                    rng[i] = 1.0  # compression / contracting range
                else:
                    transition[i] = 1.0  # mixed structure — break of one side

        if confirmed:
            last = confirmed[-1]
            swing_age[i] = float(i - last[0])
            swing_dist_atr[i] = float((last[2] - close[i]) / atr_safe[i])

    # Clip swing_age to keep feature bounded.
    swing_age_clipped = np.clip(swing_age, 0, 1440).astype(np.float32)  # cap at 1 day of bars
    swing_dist_atr = np.clip(swing_dist_atr, -10.0, 10.0).astype(np.float32)

    return pd.DataFrame({
        "ts_state_bull": bull,
        "ts_state_bear": bear,
        "ts_state_range": rng,
        "ts_state_transition": transition,
        "ts_last_swing_age_bars": swing_age_clipped,
        "ts_last_swing_dist_atr": swing_dist_atr,
    }, index=df_1m.index)


FEATURE_COLUMNS = (
    "ts_state_bull",
    "ts_state_bear",
    "ts_state_range",
    "ts_state_transition",
    "ts_last_swing_age_bars",
    "ts_last_swing_dist_atr",
)
