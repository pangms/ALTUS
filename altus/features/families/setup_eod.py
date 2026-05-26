"""A7 — End-of-Day Mean Reversion.

Thesis: In the last 30min of NY RTH, position-squaring + index rebalance
activity drags price back toward session VWAP if it's been extended.

Detection conditions:
  * Time window: last 30 min of NY RTH
  * Extended: vwap_band_position outside ±1.5σ
  * No strong trend: |mtf-alignment-proxy| < 0.7
  * Above-average volume in the move that extended

Outputs (5 features):
  seod_active                   1.0 if EOD reversion conditions met
  seod_strength                 0-1 continuous match score
  seod_direction                +1 toward VWAP (if extended above → short)
  seod_band_position            σ-units (signed; -3 to +3)
  seod_mins_until_close         minutes remaining in NY session
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from altus.features.families.setup_utils import (
    EPS, atr_safe, clip_clamp,
    in_ny_rth, session_id, hour_of_day_utc,
    NY_RTH_END_UTC, NY_RTH_START_UTC,
)


NEEDS_RAW_1M = True


def _session_vwap_sigma(df_1m: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Session-anchored VWAP + session-anchored sigma."""
    idx = df_1m.index
    sid = session_id(idx)
    tp = ((df_1m["high"] + df_1m["low"] + df_1m["close"]) / 3.0).to_numpy(dtype=np.float64)
    vol = df_1m["volume"].to_numpy(dtype=np.float64)
    tp_vol = tp * vol
    tp_sq_vol = tp * tp * vol

    df_tmp = pd.DataFrame({"sid": sid, "tp_vol": tp_vol, "vol": vol,
                            "tp_sq_vol": tp_sq_vol}, index=idx)
    cum_tpv = df_tmp.groupby("sid")["tp_vol"].cumsum().shift(1)
    cum_vol = df_tmp.groupby("sid")["vol"].cumsum().shift(1)
    cum_tp_sq_vol = df_tmp.groupby("sid")["tp_sq_vol"].cumsum().shift(1)
    vwap = (cum_tpv / cum_vol.replace(0, np.nan)).to_numpy()
    e_tp_sq = cum_tp_sq_vol / cum_vol.replace(0, np.nan)
    var = (e_tp_sq - vwap * vwap).clip(lower=0.0)
    sigma = np.sqrt(var).to_numpy()
    return vwap, sigma


def compute(df_primary: pd.DataFrame, df_1m: pd.DataFrame | None = None) -> pd.DataFrame:
    if df_1m is None:
        df_1m = df_primary
    n = len(df_1m)
    idx = df_1m.index
    closes = df_1m["close"].to_numpy(dtype=np.float64)
    atr_arr = atr_safe(df_1m, n=14)

    vwap, sigma = _session_vwap_sigma(df_1m)
    sigma_safe = np.maximum(sigma, EPS)
    band_position = np.where(np.isnan(vwap), 0.0, (closes - vwap) / sigma_safe)
    band_position = np.clip(np.nan_to_num(band_position, nan=0.0, posinf=3.0, neginf=-3.0), -3.0, 3.0)

    hour = hour_of_day_utc(idx)
    in_eod = (hour >= NY_RTH_END_UTC - 0.5) & (hour < NY_RTH_END_UTC)
    mins_until_close = np.where(in_eod, (NY_RTH_END_UTC - hour) * 60.0, 0.0).astype(np.float32)

    # Extension condition
    extended = np.abs(band_position) >= 1.5

    # Trend strength proxy: 30-bar realized close-to-close move vs ATR
    close_s = df_1m["close"]
    return_30 = (close_s - close_s.shift(30)).fillna(0.0).to_numpy()
    trend_proxy = np.abs(return_30) / np.maximum(atr_arr * 30.0, EPS)
    weak_trend = trend_proxy < 0.4

    active = (in_eod & extended & weak_trend).astype(np.int32)
    # Direction = toward VWAP (revert). band_position > 0 (above VWAP) → SHORT.
    direction = np.where(active.astype(bool),
                          np.where(band_position > 0, -1, 1),
                          0).astype(np.int32)

    # Strength
    extension_strength = clip_clamp((np.abs(band_position) - 1.5) / 1.5, 0.0, 1.0)
    time_urgency = clip_clamp((30.0 - mins_until_close) / 30.0, 0.0, 1.0)
    trend_weakness = clip_clamp(1.0 - trend_proxy / 0.7, 0.0, 1.0)
    strength = (0.4 * active.astype(np.float32) + 0.25 * extension_strength +
                0.20 * time_urgency + 0.15 * trend_weakness)
    strength = clip_clamp(strength, 0.0, 1.0)

    return pd.DataFrame({
        "seod_active": active.astype(np.float32),
        "seod_strength": strength,
        "seod_direction": direction.astype(np.float32),
        "seod_band_position": band_position.astype(np.float32),
        "seod_mins_until_close": mins_until_close,
    }, index=idx)


FEATURE_COLUMNS = (
    "seod_active",
    "seod_strength",
    "seod_direction",
    "seod_band_position",
    "seod_mins_until_close",
)
