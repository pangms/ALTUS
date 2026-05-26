"""A3 — Failed Sweep / Liquidity Trap. The highest-WR setup in the library.

Thesis: A specific HTF level (PDH/PDL/ONH/ONL) gets *swept* (price briefly
extends beyond it), then *fails to hold* (price returns through the level
within a few bars). Traders who had stops above/below the level got
triggered and are now wrong-sided; their unwinding fuels the reversal.

Detection conditions:
  * Reference level: PDH, PDL, ONH, ONL (computed by prior_day_anchors)
  * Sweep event within last 12 bars: price.high > level + 0.1*ATR OR
    price.low < level - 0.1*ATR
  * Failure: current close on opposite side of level, within 8 bars of sweep
  * Magnitude: sweep extension ≤ 0.6 ATR (stop hunt, not a real break)

Outputs (5 features):
  sfs_active           1.0 if any of {PDH, PDL, ONH, ONL} shows fresh failed sweep
  sfs_strength         0-1 continuous match score
  sfs_direction        +1 long, -1 short, 0 inactive
  sfs_age_bars         bars since most recent failed-sweep trigger (clipped)
  sfs_level_type       1=PDH/PDL (intraday), 2=ONH/ONL (overnight), 0=none

NEEDS_RAW_1M = True (uses prior_day_anchors which needs raw 1m).

CAUSALITY: All anchors come from prior_day_anchors (already causal). Sweep
detection looks at past N bars only. The "failed" check looks at current bar
close vs the level — uses bar T's data, which is causally safe under the
per-family shift(1) applied in structural.py.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from altus.features.families.prior_day_anchors import _compute_anchors
from altus.features.families.setup_utils import EPS, atr_safe, clip_clamp, fresh_age_decay


NEEDS_RAW_1M = True


def _detect_sweeps_for_level(
    highs: np.ndarray, lows: np.ndarray, closes: np.ndarray,
    level: np.ndarray, atr_arr: np.ndarray,
    sweep_lookback: int = 12, max_sweep_extension_atr: float = 0.6,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Detect failed-sweep events against a per-bar level array.

    Returns (active, direction, age_bars). For each bar T:
      - active=1 if within last `sweep_lookback` bars there was a sweep
        (above or below) that subsequently failed (close back on opposite side)
      - direction = +1 (long: sweep below failed, reverse up)
                    -1 (short: sweep above failed, reverse down)
                     0 (no active failed sweep)
      - age_bars = bars since the failure was confirmed (= bars since sweep)
    """
    n = len(highs)
    active = np.zeros(n, dtype=np.int32)
    direction = np.zeros(n, dtype=np.int32)
    age = np.full(n, sweep_lookback + 1, dtype=np.int32)  # large default

    for i in range(1, n):
        if np.isnan(level[i]):
            continue
        lvl = float(level[i])
        atr_local = max(float(atr_arr[i]), EPS)
        # Look back through recent bars for a sweep event
        lookback_start = max(0, i - sweep_lookback)
        for j in range(i - 1, lookback_start - 1, -1):
            # Sweep above: high crossed level by 0.1*ATR but not more than 0.6*ATR
            extension_above = float(highs[j]) - lvl
            extension_below = lvl - float(lows[j])
            atr_at_sweep = max(float(atr_arr[j]), EPS)
            swept_above = (extension_above > 0.1 * atr_at_sweep) and \
                          (extension_above <= max_sweep_extension_atr * atr_at_sweep)
            swept_below = (extension_below > 0.1 * atr_at_sweep) and \
                          (extension_below <= max_sweep_extension_atr * atr_at_sweep)
            if not swept_above and not swept_below:
                continue
            # Verify the failure: current close back on opposite side of level
            cur_close = float(closes[i])
            if swept_above and cur_close < lvl:
                active[i] = 1
                direction[i] = -1  # short (revert from above)
                age[i] = i - j
                break
            if swept_below and cur_close > lvl:
                active[i] = 1
                direction[i] = +1  # long (revert from below)
                age[i] = i - j
                break

    return active, direction, age


def compute(df_primary: pd.DataFrame, df_1m: pd.DataFrame | None = None) -> pd.DataFrame:
    if df_1m is None:
        df_1m = df_primary

    n = len(df_1m)
    idx = df_1m.index
    highs = df_1m["high"].to_numpy(dtype=np.float64)
    lows = df_1m["low"].to_numpy(dtype=np.float64)
    closes = df_1m["close"].to_numpy(dtype=np.float64)
    atr_arr = atr_safe(df_1m, n=14)

    # Pull PDH/PDL/ONH/ONL from prior_day_anchors.
    anchors = _compute_anchors(df_1m)
    pdh = anchors["_pdh"].to_numpy()
    pdl = anchors["_pdl"].to_numpy()
    onh = anchors["_onh"].to_numpy()
    onl = anchors["_onl"].to_numpy()

    # Detect failed sweeps against each level. We OR across levels: any failed
    # sweep against any of the four levels counts.
    pdh_active, pdh_dir, pdh_age = _detect_sweeps_for_level(highs, lows, closes, pdh, atr_arr)
    pdl_active, pdl_dir, pdl_age = _detect_sweeps_for_level(highs, lows, closes, pdl, atr_arr)
    onh_active, onh_dir, onh_age = _detect_sweeps_for_level(highs, lows, closes, onh, atr_arr)
    onl_active, onl_dir, onl_age = _detect_sweeps_for_level(highs, lows, closes, onl, atr_arr)

    # Combine: for each bar, pick the FRESHEST active sweep across levels.
    # Priority: PDH/PDL > ONH/ONL (intraday levels are stronger reads).
    active = (pdh_active | pdl_active | onh_active | onl_active).astype(np.int32)
    # Build per-bar (direction, age, level_type) by picking freshest with priority
    direction = np.zeros(n, dtype=np.int32)
    age = np.full(n, 13, dtype=np.int32)
    level_type = np.zeros(n, dtype=np.int32)
    for i in range(n):
        if not active[i]:
            continue
        candidates = []
        if pdh_active[i]: candidates.append((pdh_age[i], pdh_dir[i], 1, "pdh"))
        if pdl_active[i]: candidates.append((pdl_age[i], pdl_dir[i], 1, "pdl"))
        if onh_active[i]: candidates.append((onh_age[i], onh_dir[i], 2, "onh"))
        if onl_active[i]: candidates.append((onl_age[i], onl_dir[i], 2, "onl"))
        # Pick lowest age (freshest); break ties by priority (lower level_type)
        candidates.sort(key=lambda t: (t[0], t[2]))
        age[i] = int(candidates[0][0])
        direction[i] = int(candidates[0][1])
        level_type[i] = int(candidates[0][2])

    # Strength: combines freshness + sweep magnitude (already filtered) + level priority.
    freshness = fresh_age_decay(age.astype(np.float32), half_life_bars=6)
    level_priority_bonus = np.where(level_type == 1, 0.10, 0.0)  # PDH/PDL slight bonus
    strength = 0.5 * active.astype(np.float32) + 0.3 * freshness + level_priority_bonus
    strength = clip_clamp(strength, 0.0, 1.0)

    age_clipped = clip_clamp(age.astype(np.float32), 0.0, 12.0)

    return pd.DataFrame({
        "sfs_active": active.astype(np.float32),
        "sfs_strength": strength,
        "sfs_direction": direction.astype(np.float32),
        "sfs_age_bars": age_clipped,
        "sfs_level_type": level_type.astype(np.float32),
    }, index=idx)


FEATURE_COLUMNS = (
    "sfs_active",
    "sfs_strength",
    "sfs_direction",
    "sfs_age_bars",
    "sfs_level_type",
)
