"""Family 12 (Phase C): Order-flow proxy (VPIN) + cross-asset lead-lag.

PART 1 — VPIN (Volume-synchronized Probability of Informed trading)
====================================================================
Easley, Lopez de Prado, O'Hara (2012). Proxy for institutional order-flow
toxicity using only OHLCV (no order book). Standard approach:

  1. Bucket consecutive 1m bars into equal-volume buckets (each bucket has
     approx V/total_bars worth of volume).
  2. For each bucket, classify the bucket's net volume as buy or sell using
     bulk volume classification (BVC): buy_vol = total_vol × N((close - prev_close) / sigma),
     where N is the standard normal CDF and sigma is recent return volatility.
  3. VPIN = |buy_vol - sell_vol| / total_vol per bucket, averaged over the
     last K buckets.

High VPIN → recent flow is one-sided → institutional pressure. Trade differently.

We compute VPIN at 3 horizons (5m, 30m, 4h equivalents) for multi-scale signal.

PART 2 — Cross-asset lagged correlations (PCMCI+ simplified)
=============================================================
Full PCMCI+ does iterative conditioning to find true causal lag relationships.
That's a heavy precomputation and adds complexity that may not earn its place
for a 4-asset system. We use a simpler version: rolling lagged correlations
between MNQ returns and each cross-asset's lagged returns.

For each (asset, lag) pair we compute the rolling correlation of:
  cross_asset_return[t-lag] vs MNQ_return[t]
Lags: 1, 5, 15 bars. Assets: NQ, ES, ZB. → 9 features.

A high positive value at lag K means the cross-asset reliably leads MNQ by K
minutes — the model can use that as a directional signal.

Features (12 total):
  • flow_vpin_5m, flow_vpin_30m, flow_vpin_4h
  • flow_ll_{asset}_lag{1,5,15} for asset in {nq, es, zb} — 9 features
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm


EPS = 1e-9


def _bvc_vpin(
    log_returns: np.ndarray,
    volumes: np.ndarray,
    bucket_vol: float,
    n_buckets_window: int = 50,
) -> np.ndarray:
    """Compute VPIN time series via bulk volume classification.

    Returns an array of length len(volumes) with per-bar VPIN value (computed
    over the last `n_buckets_window` complete volume buckets ending at that bar;
    NaN where we haven't accumulated enough).
    """
    n = len(volumes)
    if n == 0 or bucket_vol <= 0:
        return np.full(n, np.nan, dtype=np.float32)

    # Recent return volatility for BVC scaling (rolling, causal)
    ret_std = pd.Series(log_returns).rolling(60, min_periods=10).std().to_numpy()

    # Walk bars, accumulate volume into the current bucket; close bucket when full
    cur_buy = 0.0
    cur_sell = 0.0
    cur_vol = 0.0
    bucket_imbalances: list[float] = []  # |buy - sell| / total per closed bucket
    vpin_out = np.full(n, np.nan, dtype=np.float32)
    bucket_end_bars: list[int] = []  # bar index at which each bucket closed

    for t in range(n):
        v = volumes[t]
        if v <= 0:
            continue
        # BVC split: buy_share = N(r/sigma)
        sigma = ret_std[t] if t < len(ret_std) and np.isfinite(ret_std[t]) and ret_std[t] > 0 else 1e-4
        z = log_returns[t] / sigma if sigma > 0 else 0.0
        buy_share = norm.cdf(z)
        bar_buy = v * buy_share
        bar_sell = v * (1.0 - buy_share)
        cur_buy += bar_buy
        cur_sell += bar_sell
        cur_vol += v

        while cur_vol >= bucket_vol:
            # Close a bucket; allocate exactly bucket_vol from current accumulator
            scale = bucket_vol / cur_vol
            bucket_buy = cur_buy * scale
            bucket_sell = cur_sell * scale
            imbalance = abs(bucket_buy - bucket_sell) / max(bucket_vol, EPS)
            bucket_imbalances.append(float(imbalance))
            bucket_end_bars.append(t)
            # Leave the remainder in the accumulator for the next bucket
            cur_buy -= bucket_buy
            cur_sell -= bucket_sell
            cur_vol -= bucket_vol

        if len(bucket_imbalances) >= n_buckets_window:
            vpin_out[t] = float(np.mean(bucket_imbalances[-n_buckets_window:]))

    # Forward-fill VPIN value to all bars between bucket closes
    s = pd.Series(vpin_out).ffill().to_numpy(dtype=np.float32)
    return s


def _lagged_corr(
    a: pd.Series,
    b: pd.Series,
    lag: int,
    window_bars: int = 60,
) -> pd.Series:
    """Rolling correlation of a[t-lag] vs b[t] over `window_bars`."""
    a_lagged = a.shift(lag)
    return a_lagged.rolling(window_bars, min_periods=max(10, window_bars // 6)).corr(b)


def compute(df_1m: pd.DataFrame) -> pd.DataFrame:
    """Compute VPIN + lead-lag features. Returns 12 columns."""
    close = df_1m["close"]
    volume = df_1m["volume"]
    log_ret = np.log(close / close.shift(1)).fillna(0).to_numpy(dtype=np.float64)
    vol_arr = volume.to_numpy(dtype=np.float64)

    # ---- VPIN at 3 horizons ----------------------------------------------
    # Bucket size proportional to median 1m volume × horizon_minutes
    median_vol = float(pd.Series(vol_arr[vol_arr > 0]).median()) if (vol_arr > 0).any() else 1.0
    out = {}
    for label, horizon_min in (("5m", 5), ("30m", 30), ("4h", 240)):
        bucket_vol = median_vol * horizon_min
        n_buckets = 50  # smoothing window in bucket count
        vpin = _bvc_vpin(log_ret, vol_arr, bucket_vol=bucket_vol, n_buckets_window=n_buckets)
        out[f"flow_vpin_{label}"] = pd.Series(vpin, index=df_1m.index).fillna(0).astype(np.float32)

    # ---- Lead-lag correlations with NQ/ES/ZB ------------------------------
    from altus.features.families.cross_asset import _load_aligned_cross_assets, CROSS_ASSETS
    cross = _load_aligned_cross_assets(df_1m)
    mnq_ret_1m = np.log(close / close.shift(1))

    for sym in CROSS_ASSETS:
        if sym not in cross:
            continue
        sym_close = cross[sym]["close"]
        sym_ret_1m = np.log(sym_close / sym_close.shift(1))
        for lag in (1, 5, 15):
            corr = _lagged_corr(sym_ret_1m, mnq_ret_1m, lag=lag, window_bars=60)
            out[f"flow_ll_{sym}_lag{lag}"] = corr.fillna(0).astype(np.float32)

    return pd.DataFrame(out, index=df_1m.index)


FEATURE_COLUMNS = (
    "flow_vpin_5m", "flow_vpin_30m", "flow_vpin_4h",
    "flow_ll_nq_lag1", "flow_ll_nq_lag5", "flow_ll_nq_lag15",
    "flow_ll_es_lag1", "flow_ll_es_lag5", "flow_ll_es_lag15",
    "flow_ll_zb_lag1", "flow_ll_zb_lag5", "flow_ll_zb_lag15",
)
