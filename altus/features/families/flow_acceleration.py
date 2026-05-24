"""Family E11 (Phase E): Flow acceleration. Answers Q32 (second derivative of imbalance).

Why this matters: most directional moves don't end because opposing flow shows up
— they end when the original flow runs out of steam. Flow DECELERATION is a
leading indicator of imminent reversal, often visible minutes before price.
Flow ACCELERATION is the "real move starting" signal.

We don't have tick-level signed flow; we use volume-weighted price-change as a
delta proxy, then compute its rate-of-change.

Features (3 total):
  • fa_delta_ema30          EMA-30 of vol-weighted price-change (delta proxy)
  • fa_delta_accel_5         rate of change of delta over last 5 bars
  • fa_delta_accel_30        rate of change over last 30 bars

CAUSALITY: EMA + diff, both causal when shifted by orchestrator.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def compute(df_1m: pd.DataFrame) -> pd.DataFrame:
    close = df_1m["close"]
    volume = df_1m["volume"]

    price_change = close.diff().fillna(0.0)
    # Volume-weighted price change as delta proxy
    delta_proxy = (price_change * np.sign(price_change)) * volume  # always positive but signed by direction
    delta_signed = price_change * volume  # signed: +ve = buying pressure proxy

    # Smooth the signed delta proxy
    delta_ema30 = delta_signed.ewm(span=30, adjust=False).mean()

    # Acceleration = current EMA - lagged EMA
    accel_5 = (delta_ema30 - delta_ema30.shift(5))
    accel_30 = (delta_ema30 - delta_ema30.shift(30))

    # Normalize by recent magnitude so the feature is scale-free
    norm_factor = delta_proxy.rolling(240, min_periods=20).mean().replace(0, np.nan)
    delta_ema30_n = (delta_ema30 / norm_factor).fillna(0.0).clip(-5, 5).astype(np.float32)
    accel_5_n = (accel_5 / norm_factor).fillna(0.0).clip(-5, 5).astype(np.float32)
    accel_30_n = (accel_30 / norm_factor).fillna(0.0).clip(-5, 5).astype(np.float32)

    return pd.DataFrame({
        "fa_delta_ema30": delta_ema30_n,
        "fa_delta_accel_5": accel_5_n,
        "fa_delta_accel_30": accel_30_n,
    }, index=df_1m.index)


FEATURE_COLUMNS = (
    "fa_delta_ema30",
    "fa_delta_accel_5",
    "fa_delta_accel_30",
)
