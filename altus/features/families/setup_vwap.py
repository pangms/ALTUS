"""A2 — VWAP Rejection / Reclaim.

Thesis: Session-anchored VWAP is institutional benchmark. In trending
sessions, VWAP gets HELD (touch + bounce) → mean-revert entry in trend
direction. In ranging sessions, VWAP gets RECLAIMED from one side →
trend resumption.

Distinct from existing vwap_anchors (which gives raw distance + bands):
this family detects the SETUP — recent holds and reclaims with trend
context — for an actionable signal.

Detection conditions:
  * In NY RTH AND past first 60min
  * mtf-alignment-style regime: bull (slope > 0) or bear (slope < 0), NOT chop
  * Price near VWAP: |vwap_dist_atr| < 0.3
  * Recent behavior: at least 1 prior touch+reject in same session OR
    clean reclaim from one side
  * VWAP slope aligns with regime

Outputs (5 features):
  svwap_active           1.0 if VWAP setup conditions met
  svwap_strength         0-1 continuous match score
  svwap_direction        +1 (long: bull regime, VWAP held); -1 (short: bear regime)
  svwap_dist_atr         signed dist to VWAP in ATR units (positive = price above)
  svwap_holds_count      number of recent VWAP holds in this session
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from altus.features.families.setup_utils import (
    EPS, atr_safe, clip_clamp,
    in_ny_rth, session_id, NY_RTH_START_UTC, NY_RTH_END_UTC, hour_of_day_utc,
)


NEEDS_RAW_1M = True


def _session_vwap(df_1m: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Session-anchored VWAP + slope (per-minute). Causal via shift(1)."""
    idx = df_1m.index
    in_rth = in_ny_rth(idx)
    sid = session_id(idx)

    tp = ((df_1m["high"] + df_1m["low"] + df_1m["close"]) / 3.0).to_numpy(dtype=np.float64)
    vol = df_1m["volume"].to_numpy(dtype=np.float64)
    tp_vol = tp * vol

    df_tmp = pd.DataFrame({"sid": sid, "tp_vol": tp_vol, "vol": vol}, index=idx)
    cum_tpv = df_tmp.groupby("sid")["tp_vol"].cumsum().shift(1)
    cum_vol = df_tmp.groupby("sid")["vol"].cumsum().shift(1)
    vwap = (cum_tpv / cum_vol.replace(0, np.nan)).to_numpy()

    # Slope = change over last 15 bars, expressed in price-points
    vwap_s = pd.Series(vwap, index=idx)
    slope = (vwap_s - vwap_s.shift(15)).fillna(0.0).to_numpy()

    return vwap, slope


def compute(df_primary: pd.DataFrame, df_1m: pd.DataFrame | None = None) -> pd.DataFrame:
    if df_1m is None:
        df_1m = df_primary
    n = len(df_1m)
    idx = df_1m.index
    closes = df_1m["close"].to_numpy(dtype=np.float64)
    atr_arr = atr_safe(df_1m, n=14)

    vwap, slope = _session_vwap(df_1m)
    in_rth = in_ny_rth(idx)
    hour = hour_of_day_utc(idx)
    past_first_hour = (hour >= NY_RTH_START_UTC + 1.0) & (hour < NY_RTH_END_UTC)

    dist_atr = np.where(np.isnan(vwap), 0.0, (closes - vwap) / np.maximum(atr_arr, EPS))
    dist_atr = np.nan_to_num(dist_atr, nan=0.0, posinf=0.0, neginf=0.0)
    near_vwap = np.abs(dist_atr) < 0.3

    # Regime detection from VWAP slope (in ATR units of recent vol).
    slope_atr = slope / np.maximum(atr_arr * 15.0, EPS)  # normalize over 15-bar window
    # Looser thresholds — 0.02 is "any directional VWAP drift" (was 0.05)
    bull_regime = slope_atr > 0.02
    bear_regime = slope_atr < -0.02

    # Count recent VWAP touches that held (within last 60 bars in same session)
    # A "hold" = a bar where price came within 0.15 ATR of VWAP and then moved
    # away by ≥ 0.25 ATR in the regime direction within next 5 bars.
    holds_count = np.zeros(n, dtype=np.int32)
    sid = session_id(idx)
    sess_to_holds = {}
    for i in range(n):
        if not in_rth[i] or np.isnan(vwap[i]):
            holds_count[i] = 0
            continue
        if sid[i] not in sess_to_holds:
            sess_to_holds[sid[i]] = []
        # Check: was bar i-5..i-1 a hold?
        if i >= 5 and not np.isnan(vwap[i - 5]):
            atr_at_i5 = max(float(atr_arr[i - 5]), EPS)
            close_at_i5 = float(closes[i - 5])
            vwap_at_i5 = float(vwap[i - 5])
            dist_at_i5 = abs(close_at_i5 - vwap_at_i5) / atr_at_i5
            if dist_at_i5 < 0.15:
                # Did price move away by 0.25 ATR in regime direction within 5 bars?
                local_window = closes[i - 4 : i + 1]
                local_window_dist = (local_window - vwap_at_i5) / atr_at_i5
                if bull_regime[i] and np.max(local_window_dist) >= 0.25:
                    sess_to_holds[sid[i]].append(i - 5)
                elif bear_regime[i] and np.min(local_window_dist) <= -0.25:
                    sess_to_holds[sid[i]].append(i - 5)
        # Count holds in last 60 bars
        cutoff = i - 60
        holds_count[i] = sum(1 for h in sess_to_holds[sid[i]] if h >= cutoff)

    # Active condition: in trending session, near VWAP, slope-aligned.
    # The `holds_count` is used in STRENGTH not GATE — prior holds boost
    # conviction but aren't required (was a 0% gate).
    active = (past_first_hour & near_vwap & (bull_regime | bear_regime)).astype(np.int32)
    direction = np.where(active.astype(bool), np.where(bull_regime, 1, np.where(bear_regime, -1, 0)), 0).astype(np.int32)

    # Strength
    proximity = clip_clamp(1.0 - np.abs(dist_atr) / 0.3, 0.0, 1.0)
    regime_strength = clip_clamp(np.abs(slope_atr) / 0.5, 0.0, 1.0)
    holds_density = clip_clamp(holds_count.astype(np.float32) / 3.0, 0.0, 1.0)
    strength = 0.4 * active.astype(np.float32) + 0.25 * proximity + 0.20 * regime_strength + 0.15 * holds_density
    strength = clip_clamp(strength, 0.0, 1.0)

    return pd.DataFrame({
        "svwap_active": active.astype(np.float32),
        "svwap_strength": strength,
        "svwap_direction": direction.astype(np.float32),
        "svwap_dist_atr": clip_clamp(dist_atr.astype(np.float32), -3.0, 3.0),
        "svwap_holds_count": clip_clamp(holds_count.astype(np.float32), 0.0, 8.0),
    }, index=idx)


FEATURE_COLUMNS = (
    "svwap_active",
    "svwap_strength",
    "svwap_direction",
    "svwap_dist_atr",
    "svwap_holds_count",
)
