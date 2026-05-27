"""Path Clearance — bidirectional confidence booster.

Composite feature answering: "Given a directional trade in either direction,
how much room is there before hitting a major obstacle?"

The engine has scattered descriptive distance features (PDA, VWAP bands,
key_levels, liquidity_zones, round_levels, volume_profile). This family
composes them into a single per-side "clearance" score.

For a LONG trade: large `pc_clearance_above_atr` = lots of runway = HIGHER
conviction → bigger size + larger targets.
For a SHORT trade: large `pc_clearance_below_atr` = lots of runway = HIGHER
conviction.

This is a CONFIDENCE MODULATOR at L2, NOT a hard veto at L3. The model
learns the relationship between clearance and outcomes — we don't hardcode
"clearance < 0.5 ATR → block trade."

Features (6 total):
  pc_clearance_above_atr        signed distance to nearest obstacle ABOVE (always >= 0)
  pc_clearance_below_atr        signed distance to nearest obstacle BELOW (always >= 0)
  pc_obstacle_above_strength    0-1 score for "stoppy-ness" of the nearest above obstacle
  pc_obstacle_below_strength    same for below
  pc_clearance_asymmetry        (above - below) / (above + below) — direction of gravity
  pc_clearance_min_atr          min(above, below) — overall "boxed in" measure

CAUSALITY: pulls from already-causal feature families. No additional shift
applied here (the per-family causal_shift in structural.py handles it).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from altus.features.families.prior_day_anchors import _compute_anchors
from altus.features.families.setup_utils import EPS, atr_safe, clip_clamp


NEEDS_RAW_1M = True


# Obstacle strength weights: how "stoppy" each level type is.
# Higher = stronger resistance/support = bigger expected price reaction.
STRENGTH_WEIGHTS = {
    "pdh_pdl": 1.0,        # prior-day high/low — top-tier institutional level
    "onh_onl": 0.8,        # overnight high/low
    "vwap_sigma": 0.7,     # session VWAP ±σ bands
    "key_swing": 0.6,      # recent swing extreme
    "round_100": 0.5,      # round 100-pt
    "round_50": 0.4,       # round 50-pt
    "hvn": 0.4,            # high volume node
}


def _collect_levels_per_bar(df_1m: pd.DataFrame, anchors: pd.DataFrame, atr_arr: np.ndarray):
    """For each bar, return arrays of (above_levels, above_weights) and
    (below_levels, below_weights).

    To keep this efficient, we use a per-bar lookup of just the most-recent
    anchor values + a few key swing extremes from a sliding window.
    """
    n = len(df_1m)
    closes = df_1m["close"].to_numpy(dtype=np.float64)
    highs = df_1m["high"].to_numpy(dtype=np.float64)
    lows = df_1m["low"].to_numpy(dtype=np.float64)

    # Anchors (from prior_day_anchors): PDH/PDL/ONH/ONL/RTH-open
    pdh = anchors["_pdh"].to_numpy()
    pdl = anchors["_pdl"].to_numpy()
    onh = anchors["_onh"].to_numpy()
    onl = anchors["_onl"].to_numpy()

    # Round 50/100-pt levels (closest above + below current close, per bar)
    round_100_above = np.where(closes > 0, np.ceil(closes / 100.0) * 100.0, np.nan)
    round_100_below = np.where(closes > 0, np.floor(closes / 100.0) * 100.0, np.nan)
    round_50_above = np.where(closes > 0, np.ceil(closes / 50.0) * 50.0, np.nan)
    round_50_below = np.where(closes > 0, np.floor(closes / 50.0) * 50.0, np.nan)

    # Recent swing high/low over 240-bar lookback (4h on 1m grid)
    swing_lookback = 240
    swing_high = (
        df_1m["high"].rolling(swing_lookback, min_periods=2).max().shift(1).to_numpy()
    )
    swing_low = (
        df_1m["low"].rolling(swing_lookback, min_periods=2).min().shift(1).to_numpy()
    )

    return {
        "pdh": pdh, "pdl": pdl, "onh": onh, "onl": onl,
        "round_100_above": round_100_above, "round_100_below": round_100_below,
        "round_50_above": round_50_above, "round_50_below": round_50_below,
        "swing_high": swing_high, "swing_low": swing_low,
    }


def _compose_clearance(
    closes: np.ndarray, atr_arr: np.ndarray, levels: dict,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Returns (clearance_above_atr, clearance_below_atr,
    strength_above, strength_below) per bar.
    """
    n = len(closes)
    clearance_above = np.full(n, np.nan, dtype=np.float64)
    clearance_below = np.full(n, np.nan, dtype=np.float64)
    strength_above = np.zeros(n, dtype=np.float64)
    strength_below = np.zeros(n, dtype=np.float64)

    # Define candidate above-levels per bar with their type weights
    above_candidates = [
        ("pdh", levels["pdh"], STRENGTH_WEIGHTS["pdh_pdl"]),
        ("onh", levels["onh"], STRENGTH_WEIGHTS["onh_onl"]),
        ("round_100_above", levels["round_100_above"], STRENGTH_WEIGHTS["round_100"]),
        ("round_50_above", levels["round_50_above"], STRENGTH_WEIGHTS["round_50"]),
        ("swing_high", levels["swing_high"], STRENGTH_WEIGHTS["key_swing"]),
    ]
    below_candidates = [
        ("pdl", levels["pdl"], STRENGTH_WEIGHTS["pdh_pdl"]),
        ("onl", levels["onl"], STRENGTH_WEIGHTS["onh_onl"]),
        ("round_100_below", levels["round_100_below"], STRENGTH_WEIGHTS["round_100"]),
        ("round_50_below", levels["round_50_below"], STRENGTH_WEIGHTS["round_50"]),
        ("swing_low", levels["swing_low"], STRENGTH_WEIGHTS["key_swing"]),
    ]

    for i in range(n):
        c = float(closes[i])
        atr_i = max(float(atr_arr[i]), EPS)
        # Above: nearest level > c
        best_above_dist = np.inf
        best_above_weight = 0.0
        for _, level_arr, weight in above_candidates:
            v = level_arr[i] if i < len(level_arr) else np.nan
            if np.isnan(v) or v <= c:
                continue
            d = (v - c) / atr_i
            if d < best_above_dist:
                best_above_dist = d
                best_above_weight = weight
        if best_above_dist != np.inf:
            clearance_above[i] = best_above_dist
            strength_above[i] = best_above_weight
        # Below: nearest level < c
        best_below_dist = np.inf
        best_below_weight = 0.0
        for _, level_arr, weight in below_candidates:
            v = level_arr[i] if i < len(level_arr) else np.nan
            if np.isnan(v) or v >= c:
                continue
            d = (c - v) / atr_i
            if d < best_below_dist:
                best_below_dist = d
                best_below_weight = weight
        if best_below_dist != np.inf:
            clearance_below[i] = best_below_dist
            strength_below[i] = best_below_weight

    return clearance_above, clearance_below, strength_above, strength_below


def compute(df_primary: pd.DataFrame, df_1m: pd.DataFrame | None = None) -> pd.DataFrame:
    if df_1m is None:
        df_1m = df_primary
    closes = df_1m["close"].to_numpy(dtype=np.float64)
    atr_arr = atr_safe(df_1m, n=14)
    anchors = _compute_anchors(df_1m)

    levels = _collect_levels_per_bar(df_1m, anchors, atr_arr)
    clear_above, clear_below, str_above, str_below = _compose_clearance(closes, atr_arr, levels)

    # Fill missing (no level found above/below) with a "max-out" value of 10 ATR.
    clear_above = clip_clamp(np.nan_to_num(clear_above, nan=10.0), 0.0, 10.0)
    clear_below = clip_clamp(np.nan_to_num(clear_below, nan=10.0), 0.0, 10.0)

    total = np.maximum(clear_above + clear_below, EPS)
    asymmetry = (clear_above - clear_below) / total
    asymmetry = clip_clamp(asymmetry, -1.0, 1.0)

    min_clear = np.minimum(clear_above, clear_below)
    min_clear = clip_clamp(min_clear, 0.0, 10.0)

    return pd.DataFrame({
        "pc_clearance_above_atr": clear_above,
        "pc_clearance_below_atr": clear_below,
        "pc_obstacle_above_strength": str_above.astype(np.float32),
        "pc_obstacle_below_strength": str_below.astype(np.float32),
        "pc_clearance_asymmetry": asymmetry,
        "pc_clearance_min_atr": min_clear,
    }, index=df_1m.index)


FEATURE_COLUMNS = (
    "pc_clearance_above_atr",
    "pc_clearance_below_atr",
    "pc_obstacle_above_strength",
    "pc_obstacle_below_strength",
    "pc_clearance_asymmetry",
    "pc_clearance_min_atr",
)
