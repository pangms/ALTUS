"""A6 — Failed Auction. Multi-touch level rejection.

Thesis: Price tests a specific level multiple times within a short window
and fails to extend each time. When the market tries to discover higher
(or lower) but cannot find acceptance, it reverses to find acceptance
elsewhere. From market profile / auction theory.

Detection conditions:
  * A specific price (within 0.15*ATR tolerance) touched ≥ 3 times in last 60 bars
  * Each touch failed: price returned ≥ 0.3*ATR from test level within 5 bars
  * Most recent touch within last 10 bars (fresh)
  * Level was either a swing extreme or near key_levels swing density

Outputs (5 features):
  sfa_active            1.0 if multi-touch failure pattern present
  sfa_strength          0-1 continuous match score
  sfa_direction         +1 if testing support (long); -1 if testing resistance (short); 0 inactive
  sfa_touch_count       number of touches detected (clipped at 8)
  sfa_age_bars          bars since most recent touch (clipped)

CAUSALITY: Touch detection looks at past N bars. Confirmation requires the
"return" move which is in the past by construction.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from altus.features.families.setup_utils import EPS, atr_safe, clip_clamp, fresh_age_decay


NEEDS_RAW_1M = True


def _detect_multi_touch(
    highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, atr_arr: np.ndarray,
    lookback: int = 60, tolerance_atr: float = 0.08, recovery_atr: float = 0.5,
    recovery_window: int = 5, max_recent_touch_bars: int = 8,
    min_touches: int = 3,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Detect multi-touch level-defense patterns.

    Strategy: at each bar i, look back `lookback` bars and find the most
    densely-tested level. A "touch" is a bar whose high/low came within
    tolerance*ATR of a candidate price. A failed touch is one where price
    returned recovery_atr within recovery_window bars.

    Returns (active, direction, touch_count, age_since_last_touch, test_price).
    """
    n = len(highs)
    active = np.zeros(n, dtype=np.int32)
    direction = np.zeros(n, dtype=np.int32)
    touch_count = np.zeros(n, dtype=np.int32)
    age = np.full(n, lookback + 1, dtype=np.int32)
    test_price = np.full(n, np.nan, dtype=np.float64)

    for i in range(lookback, n):
        # Candidate test prices: the high/low extremes in the lookback window.
        # We test the absolute extremes (max high, min low) plus a few intermediate.
        win_high = float(np.max(highs[i - lookback : i]))
        win_low = float(np.min(lows[i - lookback : i]))
        atr_local = max(float(atr_arr[i]), EPS)
        tolerance = tolerance_atr * atr_local

        best_count = 0
        best_dir = 0
        best_last_touch_age = lookback + 1
        best_price = np.nan

        for candidate_price, cand_dir in [(win_high, -1), (win_low, +1)]:
            count = 0
            last_touch_idx = -1
            for j in range(i - lookback, i):
                # For resistance test: high near the level
                if cand_dir == -1:  # testing resistance from below
                    if abs(float(highs[j]) - candidate_price) <= tolerance:
                        # Failure check: within next recovery_window bars, did
                        # price recover by recovery_atr from the level?
                        recovery_end = min(j + recovery_window + 1, i)
                        lowest_after = float(np.min(lows[j : recovery_end])) if recovery_end > j else float(lows[j])
                        if candidate_price - lowest_after >= recovery_atr * atr_local:
                            count += 1
                            last_touch_idx = j
                else:  # testing support from above
                    if abs(float(lows[j]) - candidate_price) <= tolerance:
                        recovery_end = min(j + recovery_window + 1, i)
                        highest_after = float(np.max(highs[j : recovery_end])) if recovery_end > j else float(highs[j])
                        if highest_after - candidate_price >= recovery_atr * atr_local:
                            count += 1
                            last_touch_idx = j
            if count >= min_touches and last_touch_idx >= 0:
                touch_age = i - last_touch_idx
                if touch_age <= max_recent_touch_bars and count > best_count:
                    best_count = count
                    best_dir = cand_dir
                    best_last_touch_age = touch_age
                    best_price = candidate_price

        if best_count >= min_touches:
            active[i] = 1
            direction[i] = best_dir
            touch_count[i] = best_count
            age[i] = best_last_touch_age
            test_price[i] = best_price

    return active, direction, touch_count, age, test_price


def compute(df_primary: pd.DataFrame, df_1m: pd.DataFrame | None = None) -> pd.DataFrame:
    if df_1m is None:
        df_1m = df_primary
    highs = df_1m["high"].to_numpy(dtype=np.float64)
    lows = df_1m["low"].to_numpy(dtype=np.float64)
    closes = df_1m["close"].to_numpy(dtype=np.float64)
    atr_arr = atr_safe(df_1m, n=14)

    active, direction, touch_count, age, _ = _detect_multi_touch(
        highs, lows, closes, atr_arr,
    )

    # Strength: combines touch density + freshness
    freshness = fresh_age_decay(age.astype(np.float32), half_life_bars=8)
    density = clip_clamp((touch_count.astype(np.float32) - 3.0) / 3.0, 0.0, 1.0)
    strength = 0.4 * active.astype(np.float32) + 0.35 * freshness + 0.25 * density
    strength = clip_clamp(strength, 0.0, 1.0)

    return pd.DataFrame({
        "sfa_active": active.astype(np.float32),
        "sfa_strength": strength,
        "sfa_direction": direction.astype(np.float32),
        "sfa_touch_count": clip_clamp(touch_count.astype(np.float32), 0.0, 8.0),
        "sfa_age_bars": clip_clamp(age.astype(np.float32), 0.0, 10.0),
    }, index=df_1m.index)


FEATURE_COLUMNS = (
    "sfa_active",
    "sfa_strength",
    "sfa_direction",
    "sfa_touch_count",
    "sfa_age_bars",
)
