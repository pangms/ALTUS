"""Family E6 (Phase E): Volatility regime. Answers Q23 (vol expanding/contracting/stable).

Why this matters: every other question's answer depends on vol regime. The same
setup in a vol-expansion regime (move likely to continue) vs vol-contraction
regime (compression before move) has opposite expectancy. We need an explicit
vol-regime signal beyond raw realized vol.

Features (4 total):
  • vr_realized_z_60      Z-score of 60-bar realized vol vs 1440-bar (1-day) baseline
  • vr_realized_z_240     same over 240-bar (4h) vs 1440-bar
  • vr_expanding          slope of vol over last 60 bars (positive = expanding)
  • vr_regime_score       composite in [-1, 1]: -1 = strong contraction, +1 = strong expansion

CAUSALITY: rolling std + slope, both use only past data when shifted by
orchestrator. At bar T the windows close at T (orchestrator shifts to T-1).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


EPS = 1e-9


def compute(df_1m: pd.DataFrame) -> pd.DataFrame:
    close = df_1m["close"]
    log_ret = np.log(close / close.shift(1)).fillna(0.0)

    # Realized vol at multiple horizons
    rv_60 = log_ret.rolling(60, min_periods=10).std()
    rv_240 = log_ret.rolling(240, min_periods=20).std()
    rv_1440 = log_ret.rolling(1440, min_periods=100).std()

    # Z-score short-horizon vol against day baseline
    rv_1440_mean = rv_1440.rolling(1440, min_periods=100).mean()
    rv_1440_std = rv_1440.rolling(1440, min_periods=100).std().replace(0, np.nan)
    z_60 = ((rv_60 - rv_1440_mean) / rv_1440_std).fillna(0.0).clip(-5, 5).astype(np.float32)
    z_240 = ((rv_240 - rv_1440_mean) / rv_1440_std).fillna(0.0).clip(-5, 5).astype(np.float32)

    # Vol-expansion slope: (rv_now - rv_60ago) / rv_60ago
    rv_60_lagged = rv_60.shift(60)
    expanding = ((rv_60 - rv_60_lagged) / rv_60_lagged.replace(0, np.nan)).fillna(0.0).clip(-2, 2).astype(np.float32)

    # Composite regime score in [-1, 1]: combine z_60 + expanding slope
    regime_score = (np.tanh(z_60 * 0.5) + np.tanh(expanding * 0.5)).clip(-2, 2) / 2.0
    regime_score = regime_score.astype(np.float32)

    return pd.DataFrame({
        "vr_realized_z_60": z_60,
        "vr_realized_z_240": z_240,
        "vr_expanding": expanding,
        "vr_regime_score": regime_score,
    }, index=df_1m.index)


FEATURE_COLUMNS = (
    "vr_realized_z_60",
    "vr_realized_z_240",
    "vr_expanding",
    "vr_regime_score",
)
