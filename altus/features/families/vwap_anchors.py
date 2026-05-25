"""Session-anchored VWAP + standard-deviation bands. Answers Q6 / Q7 / Q9.

Why this matters: VWAP is the single most-watched intraday anchor on equity
index futures. Institutional traders measure execution against it; algos pile
mean-reversion logic around it. The ±1σ and ±2σ bands give explicit "deep in
value vs at the edge vs outside value" thresholds. POC from volume_profile is
path-INDEPENDENT (it's a total-volume aggregation); VWAP is path-DEPENDENT
(weighted by the sequence of volume), so they capture different "value" angles.

Features (5 total, all causal — VWAP at bar T uses only bars in the current
session up to and including T-1's close):
  vwap_dist_atr               signed distance to VWAP (positive = price above)
  vwap_band_position          where in [-2, +2] are we, in σ units (clipped)
  vwap_dist_to_upper1_atr     signed distance to +1σ band
  vwap_dist_to_lower1_atr     signed distance to -1σ band
  vwap_slope_15m              VWAP slope over last 15m, normalized

Session anchor: each RTH session (13:30-20:00 UTC) starts a new VWAP. Outside
RTH, we accumulate a "globex" VWAP that resets when RTH starts. Two anchors
let downstream consumers see both.

CAUSALITY: VWAP at bar T computed strictly from bars before T (we use .shift(1)
on the cumulative sums). No within-bar peek.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from altus.features.families.session_time import NY_RTH_END_UTC, NY_RTH_START_UTC


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


def _session_anchored_vwap(df_1m: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """Compute session-anchored VWAP + the running std-dev of (price - vwap).

    Returns (vwap, sigma) — both aligned to df_1m.index. VWAP resets to NaN at
    each session boundary; values at bar T use only bars 0..T-1 (causal via shift).
    """
    idx = df_1m.index
    hour_utc = (idx.hour + idx.minute / 60.0).to_numpy()
    in_rth = (hour_utc >= NY_RTH_START_UTC) & (hour_utc < NY_RTH_END_UTC)

    # Session id: increments on every RTH start. Within a session, all bars
    # share the same id, so cumsum() over a session is straightforward.
    rth_start = np.zeros_like(in_rth, dtype=bool)
    rth_start[1:] = in_rth[1:] & ~in_rth[:-1]
    # We also want a session boundary when RTH ends → groupby resets.
    rth_end = np.zeros_like(in_rth, dtype=bool)
    rth_end[1:] = ~in_rth[1:] & in_rth[:-1]
    boundary = rth_start | rth_end
    session_id = np.cumsum(boundary)  # int session counter

    # Typical price × volume.
    tp = (df_1m["high"] + df_1m["low"] + df_1m["close"]).to_numpy(dtype=np.float64) / 3.0
    vol = df_1m["volume"].to_numpy(dtype=np.float64)
    tp_vol = tp * vol

    # Cumulative within each session, then shift(1) to make causal.
    df_tmp = pd.DataFrame({
        "tp_vol": tp_vol,
        "vol": vol,
        "tp_sq_vol": (tp * tp) * vol,
        "sid": session_id,
    }, index=idx)
    cum_tpv = df_tmp.groupby("sid")["tp_vol"].cumsum().shift(1)
    cum_vol = df_tmp.groupby("sid")["vol"].cumsum().shift(1)
    cum_tp_sq_vol = df_tmp.groupby("sid")["tp_sq_vol"].cumsum().shift(1)

    vwap = cum_tpv / cum_vol.replace(0, np.nan)
    # E[tp²]_w - (E[tp]_w)² — weighted variance
    e_tp = cum_tpv / cum_vol.replace(0, np.nan)
    e_tp_sq = cum_tp_sq_vol / cum_vol.replace(0, np.nan)
    variance = (e_tp_sq - e_tp * e_tp).clip(lower=0.0)
    sigma = np.sqrt(variance)

    return vwap, sigma


def compute(df_1m: pd.DataFrame) -> pd.DataFrame:
    close = df_1m["close"].to_numpy(dtype=np.float64)
    atr = _atr(df_1m, n=14).replace(0, np.nan).ffill().fillna(1.0).to_numpy(dtype=np.float64)
    atr_safe = np.maximum(atr, EPS)

    vwap_s, sigma_s = _session_anchored_vwap(df_1m)
    vwap = vwap_s.to_numpy()
    sigma = sigma_s.to_numpy()

    # Distance from VWAP in ATR.
    dist_vwap = (close - vwap) / atr_safe
    dist_vwap = np.nan_to_num(dist_vwap, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    # Band position in σ units, clipped to [-3, 3].
    sigma_safe = np.maximum(sigma, EPS)
    band_pos = (close - vwap) / sigma_safe
    band_pos = np.nan_to_num(band_pos, nan=0.0, posinf=3.0, neginf=-3.0)
    band_pos = np.clip(band_pos, -3.0, 3.0).astype(np.float32)

    # Distance to +1σ and -1σ bands in ATR.
    upper1 = vwap + sigma
    lower1 = vwap - sigma
    dist_upper1 = np.nan_to_num((upper1 - close) / atr_safe, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    dist_lower1 = np.nan_to_num((close - lower1) / atr_safe, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    # VWAP slope over 15m (15 bars), normalized by ATR.
    vwap_lag = vwap_s.shift(15)
    slope_raw = (vwap_s - vwap_lag) / atr_safe
    slope = slope_raw.fillna(0.0).clip(-5.0, 5.0).astype(np.float32).to_numpy()

    return pd.DataFrame({
        "vwap_dist_atr": dist_vwap,
        "vwap_band_position": band_pos,
        "vwap_dist_to_upper1_atr": dist_upper1,
        "vwap_dist_to_lower1_atr": dist_lower1,
        "vwap_slope_15m": slope,
    }, index=df_1m.index)


FEATURE_COLUMNS = (
    "vwap_dist_atr",
    "vwap_band_position",
    "vwap_dist_to_upper1_atr",
    "vwap_dist_to_lower1_atr",
    "vwap_slope_15m",
)
