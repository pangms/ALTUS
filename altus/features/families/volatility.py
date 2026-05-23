"""Family 3: Volatility regime features.

Why this matters: 30pt TP/SL means very different things in different vol
regimes. Layer 1 can implicitly learn current vol from recent ATR-like features,
but lacks *regime context* — e.g., "this week is in the 90th percentile of
yearly vol" vs "this is a typical low-vol week." These features supply that.

Method choices:
  * Realized volatility (sum of squared log-returns) over multiple windows —
    direct, no model assumptions, captures the present level cleanly.
  * Vol of vol — captures vol *regime instability* (turbulent vs calm).
  * Hurst exponent on log(vol) time series — captures whether vol is trending
    (persistent regime) vs mean-reverting (chaotic).
  * Percentile of current 1d vol vs trailing 60d — contextual ranking.

Why NOT GARCH: in our prior research we agreed GARCH adds marginal value once
realized vol is computed across multiple windows + vol-of-vol + Hurst already
in the feature set. Keep simple; revisit if OOS shows we're leaving signal.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from altus.features.families.trend_hurst import _hurst_dfa


EPS = 1e-9


def _realized_vol(log_returns: pd.Series, window_bars: int) -> pd.Series:
    """Sum of squared 1m log-returns over the given window, annualized."""
    # 1m bars per trading year ~ 525,600 (24/5 trading); annualization is
    # informational — for ML it doesn't matter, but it puts numbers on a
    # sane scale that's regime-comparable.
    annualization = np.sqrt(525_600.0 / max(window_bars, 1))
    return np.sqrt(log_returns.pow(2).rolling(window_bars, min_periods=window_bars // 4).sum()) * annualization


def compute(df_1m: pd.DataFrame) -> pd.DataFrame:
    """Compute volatility regime features aligned to df_1m.index.

    Returns 8 columns: vol_realized_{5m, 30m, 4h, 1d}, vol_of_vol, vol_hurst,
    vol_percentile_60d, vol_regime_score.
    """
    close = df_1m["close"]
    log_ret = np.log(close / close.shift(1))

    rv_5m = _realized_vol(log_ret, 5)
    rv_30m = _realized_vol(log_ret, 30)
    rv_4h = _realized_vol(log_ret, 240)
    rv_1d = _realized_vol(log_ret, 1440)

    # Vol-of-vol: std of 1-hour rolling vol over the last 24 hours
    rv_60m = _realized_vol(log_ret, 60)
    vol_of_vol = rv_60m.rolling(1440, min_periods=360).std()

    # Hurst on log(vol) — captures persistence of vol regime.
    # Sample the vol series at 4h cadence (downsample) before computing Hurst,
    # then broadcast back to 1m. Vol regime changes slowly; computing Hurst
    # every minute is wasteful and would take ~30 min on 5yr data.
    rv_4h_at_4h = rv_4h.iloc[::240]
    log_rv_4h = np.log(rv_4h_at_4h.replace(0, np.nan).dropna())
    hurst_at_4h = _hurst_rolling_downsampled(log_rv_4h.to_numpy(), window=120)
    hurst_at_4h_series = pd.Series(hurst_at_4h, index=log_rv_4h.index, dtype=np.float32)
    vol_hurst = hurst_at_4h_series.reindex(df_1m.index).ffill().fillna(0.5)

    # Percentile of current 1d vol vs trailing 60d window.
    # rolling().rank(pct=True) is C-implemented and ~50x faster than .apply(lambda).
    vol_percentile_60d = rv_1d.rolling(86_400, min_periods=14_400).rank(pct=True)

    vol_regime_score = vol_percentile_60d.fillna(0.5)

    return pd.DataFrame(
        {
            "vol_realized_5m": rv_5m.astype(np.float32),
            "vol_realized_30m": rv_30m.astype(np.float32),
            "vol_realized_4h": rv_4h.astype(np.float32),
            "vol_realized_1d": rv_1d.astype(np.float32),
            "vol_of_vol": vol_of_vol.astype(np.float32),
            "vol_hurst": vol_hurst.astype(np.float32),
            "vol_percentile_60d": vol_percentile_60d.astype(np.float32),
            "vol_regime_score": vol_regime_score.astype(np.float32),
        },
        index=df_1m.index,
    )


def _hurst_rolling_downsampled(series: np.ndarray, window: int) -> np.ndarray:
    """Hurst over rolling window on a pre-downsampled series. Returns 0.5 for warmup."""
    n = len(series)
    out = np.full(n, 0.5, dtype=np.float32)
    if n < window:
        return out
    for t in range(window, n):
        x = series[t - window : t]
        if np.all(np.isfinite(x)) and x.std() > EPS:
            out[t] = _hurst_dfa(x)
    return out


FEATURE_COLUMNS = (
    "vol_realized_5m",
    "vol_realized_30m",
    "vol_realized_4h",
    "vol_realized_1d",
    "vol_of_vol",
    "vol_hurst",
    "vol_percentile_60d",
    "vol_regime_score",
)
