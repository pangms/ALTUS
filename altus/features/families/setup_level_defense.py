"""A8 — Multi-Touch Level Defense. Proven level defends Nth time.

Thesis: A level (key swing, PDH/PDL, VWAP, round number) defended ≥3 times
within 240 bars becomes "proven" institutional defense. When price returns
for the Nth time, P(another defense) > P(break) for N=3-5.

Distinct from A6: A6 fires when the failed touches have ALREADY happened and
price is no longer at the level (looking BACK at the multi-touch pattern).
A8 fires when price is APPROACHING the proven level for the next test
(looking FORWARD at the upcoming test).

Detection conditions:
  * A specific price (within 0.10*ATR tolerance) touched + defended ≥3 times in last 240 bars
  * Each defense: price came within 0.10*ATR, then moved ≥0.4*ATR away within 5 bars
  * Current bar: price approaching the level from outside (within 0.3*ATR but not yet at it)
  * No structural break: no close beyond level by >0.3*ATR in defending history

Outputs (5 features):
  sld_active            1.0 if approaching a proven defended level
  sld_strength          0-1 continuous match score
  sld_direction         +1 (long: approaching support); -1 (short: approaching resistance)
  sld_defense_count     number of successful defenses (clipped at 8)
  sld_dist_to_level_atr signed distance to the defended level in ATR units

CAUSALITY: Defense history is all past bars; approach is current bar's close
vs past level (causal under per-family shift(1)).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from altus.features.families.setup_utils import EPS, atr_safe, clip_clamp


NEEDS_RAW_1M = True


def _find_proven_levels(
    highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, atr_arr: np.ndarray,
    lookback: int = 240, tolerance_atr: float = 0.10, defense_atr: float = 0.4,
    defense_window: int = 5,
):
    """For each bar i, find the BEST proven level (most-defended) within the
    lookback window. Return (level, defense_count, level_type_dir) per bar.
    level_type_dir: +1 if support (defended from above), -1 if resistance.
    """
    n = len(highs)
    level_arr = np.full(n, np.nan, dtype=np.float64)
    defense_count = np.zeros(n, dtype=np.int32)
    level_type = np.zeros(n, dtype=np.int32)

    for i in range(lookback, n):
        atr_local = max(float(atr_arr[i]), EPS)
        tolerance = tolerance_atr * atr_local
        defense_thr = defense_atr * atr_local

        # Candidate levels: lookback-window high, lookback-window low
        win_high = float(np.max(highs[i - lookback : i]))
        win_low = float(np.min(lows[i - lookback : i]))

        best_count = 0
        best_level = np.nan
        best_dir = 0

        for cand_price, cand_dir in [(win_high, -1), (win_low, +1)]:
            # Count defenses
            count = 0
            broken = False
            for j in range(i - lookback, i):
                cur_close = float(closes[j])
                if cand_dir == -1:  # resistance — defended from above
                    if cur_close > cand_price + 0.3 * atr_local:
                        broken = True
                        break
                    if abs(float(highs[j]) - cand_price) <= tolerance:
                        recovery_end = min(j + defense_window + 1, i)
                        if recovery_end > j:
                            recovered = (cand_price - float(np.min(lows[j : recovery_end]))) >= defense_thr
                            if recovered:
                                count += 1
                else:  # support — defended from below
                    if cur_close < cand_price - 0.3 * atr_local:
                        broken = True
                        break
                    if abs(float(lows[j]) - cand_price) <= tolerance:
                        recovery_end = min(j + defense_window + 1, i)
                        if recovery_end > j:
                            recovered = (float(np.max(highs[j : recovery_end])) - cand_price) >= defense_thr
                            if recovered:
                                count += 1
            if not broken and count >= 3 and count > best_count:
                best_count = count
                best_level = cand_price
                best_dir = cand_dir

        if best_count >= 3:
            level_arr[i] = best_level
            defense_count[i] = best_count
            level_type[i] = best_dir

    return level_arr, defense_count, level_type


def compute(df_primary: pd.DataFrame, df_1m: pd.DataFrame | None = None) -> pd.DataFrame:
    if df_1m is None:
        df_1m = df_primary
    n = len(df_1m)
    highs = df_1m["high"].to_numpy(dtype=np.float64)
    lows = df_1m["low"].to_numpy(dtype=np.float64)
    closes = df_1m["close"].to_numpy(dtype=np.float64)
    atr_arr = atr_safe(df_1m, n=14)

    levels, defense_count, level_type = _find_proven_levels(highs, lows, closes, atr_arr)

    # Activate when proven level found AND price is approaching (within 0.3*ATR
    # but not yet touching = beyond tolerance).
    dist_to_level = np.where(np.isnan(levels), 0.0, (levels - closes) / np.maximum(atr_arr, EPS))
    approaching_resistance = (level_type == -1) & (dist_to_level >= 0.10) & (dist_to_level <= 0.5)
    approaching_support = (level_type == +1) & (dist_to_level <= -0.10) & (dist_to_level >= -0.5)

    active = (approaching_resistance | approaching_support).astype(np.int32)
    # Direction for entry = opposite of approach direction
    # (approaching resistance from below → expect rejection → SHORT)
    direction = np.where(active.astype(bool), level_type, 0).astype(np.int32)

    # Strength: density of defenses + level proximity (closer = stronger signal)
    density = clip_clamp((defense_count.astype(np.float32) - 3.0) / 4.0, 0.0, 1.0)
    proximity = clip_clamp(1.0 - np.abs(dist_to_level) / 0.5, 0.0, 1.0).astype(np.float32)
    strength = 0.4 * active.astype(np.float32) + 0.35 * density + 0.25 * proximity
    strength = clip_clamp(strength, 0.0, 1.0)

    return pd.DataFrame({
        "sld_active": active.astype(np.float32),
        "sld_strength": strength,
        "sld_direction": direction.astype(np.float32),
        "sld_defense_count": clip_clamp(defense_count.astype(np.float32), 0.0, 8.0),
        "sld_dist_to_level_atr": clip_clamp(dist_to_level.astype(np.float32), -2.0, 2.0),
    }, index=df_1m.index)


FEATURE_COLUMNS = (
    "sld_active",
    "sld_strength",
    "sld_direction",
    "sld_defense_count",
    "sld_dist_to_level_atr",
)
