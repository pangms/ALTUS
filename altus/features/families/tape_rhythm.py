"""Family E10 (Phase E): Tape rhythm. Answers Q29 (steady vs spasmodic flow).

Why this matters: a steady, orderly tape = sustainable participation, moves carry
through. A burst-like, spasmodic tape = pump-and-dump character, false breakouts,
algo games. The engine should know whether the current flow rhythm is the kind
where breakouts tend to follow through or fade.

We measure tape "burstiness" via variance of volume-Z over a rolling window
(Goh & Barabási burstiness coefficient adapted). High variance + occasional
spikes = bursty. Low variance + steady flow = orderly.

Features (3 total):
  • tr_burstiness            burstiness coefficient in [-1, 1] over last 60 bars
                              positive = bursty, negative = steady
  • tr_vol_z_var_240         variance of volume Z-score over last 240 bars
  • tr_inter_arrival_cv      coefficient of variation of inter-spike intervals
                              (high = irregular pulse, low = regular cadence)

CAUSALITY: rolling windows ending at bar T; orchestrator shift handles
feature-at-T → data-≤-T-1.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


EPS = 1e-9


def compute(df_1m: pd.DataFrame) -> pd.DataFrame:
    volume = df_1m["volume"]

    vol_mean = volume.rolling(200, min_periods=20).mean()
    vol_std = volume.rolling(200, min_periods=20).std().replace(0, np.nan)
    vol_z = ((volume - vol_mean) / vol_std).fillna(0.0).clip(-5, 5)

    # Burstiness coefficient over last 60 bars: B = (std - mean) / (std + mean)
    rolling_mean = vol_z.abs().rolling(60, min_periods=20).mean()
    rolling_std = vol_z.abs().rolling(60, min_periods=20).std()
    burstiness = ((rolling_std - rolling_mean) / (rolling_std + rolling_mean + EPS)).fillna(0.0).clip(-1, 1).astype(np.float32)

    vol_z_var_240 = vol_z.rolling(240, min_periods=30).var().fillna(0.0).clip(0, 25).astype(np.float32)

    # Inter-arrival CV: time between volume spikes (z > 1.5).
    is_spike = (vol_z > 1.5).astype(int)
    # Count spikes in last 240 bars; CV ≈ std / mean of gaps.
    # Simple proxy: rolling sum of spikes, and the spike-density variance.
    spike_density = is_spike.rolling(240, min_periods=30).mean().fillna(0.0)
    density_std = is_spike.rolling(240, min_periods=30).std().fillna(0.0)
    inter_arr_cv = (density_std / (spike_density + EPS)).clip(0, 5).fillna(0.0).astype(np.float32)

    return pd.DataFrame({
        "tr_burstiness": burstiness,
        "tr_vol_z_var_240": vol_z_var_240,
        "tr_inter_arrival_cv": inter_arr_cv,
    }, index=df_1m.index)


FEATURE_COLUMNS = (
    "tr_burstiness",
    "tr_vol_z_var_240",
    "tr_inter_arrival_cv",
)
