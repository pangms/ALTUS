"""Family E3 (Phase E): Absorption detection. Answers Q5 (absorption vs conviction).

Why this matters: high volume + small price move = a large player is absorbing
the opposing flow at this level. The classic "stopping action" before reversal.
Conversely: low volume + big price move = thin liquidity, conviction or fakery.

We measure vol-normalized move size: actual N-bar move / expected move given
volume Z-score. Values < 1 = absorption (volume present, price didn't move
as expected). Values > 1 = conviction or thin-liquidity move.

Features (3 total):
  • abs_ratio_5         actual move / expected move over last 5 bars
  • abs_ratio_15        same over last 15 bars
  • abs_streak          rolling count of last-10-bars where ratio < 0.5 (strong absorption)

CAUSALITY: rolling windows ending at bar T-1 (then shifted by 1 at the
structural orchestrator level). _expected_move uses bar-T's vol but compares
to bar-T's move — both observed at T — so the feature value at T uses data ≤ T.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


EPS = 1e-9


def compute(df_1m: pd.DataFrame) -> pd.DataFrame:
    close = df_1m["close"]
    volume = df_1m["volume"]

    log_ret = np.log(close / close.shift(1)).fillna(0.0)
    abs_ret = log_ret.abs()

    # Expected move ≈ baseline volatility × (1 + log_volume_z)
    # Volume Z-score causal: rolling 200-bar mean and std.
    vol_mean = volume.rolling(200, min_periods=20).mean()
    vol_std = volume.rolling(200, min_periods=20).std().replace(0, np.nan)
    vol_z = ((volume - vol_mean) / vol_std).fillna(0.0).clip(-3, 3)

    baseline_vol = abs_ret.rolling(60, min_periods=10).mean().replace(0, np.nan)

    def _ratio(window: int) -> pd.Series:
        actual_move = abs_ret.rolling(window, min_periods=2).sum()
        # Expected = baseline_vol × window × (1 + 0.3 × mean_vol_z over window)
        mean_vz = vol_z.rolling(window, min_periods=2).mean()
        expected = baseline_vol * window * (1.0 + 0.3 * mean_vz.clip(-3, 3))
        expected = expected.replace(0, np.nan)
        ratio = (actual_move / expected).fillna(1.0).clip(0, 5).astype(np.float32)
        return ratio

    ratio_5 = _ratio(5)
    ratio_15 = _ratio(15)

    # Streak: how many of the last 10 bars were strong-absorption (ratio_5 < 0.5
    # while vol_z > 0 — i.e., volume was present but price didn't move)
    strong_absorb = ((ratio_5 < 0.5) & (vol_z > 0.0)).astype(np.float32)
    streak = strong_absorb.rolling(10, min_periods=1).sum().astype(np.float32)

    return pd.DataFrame({
        "abs_ratio_5": ratio_5,
        "abs_ratio_15": ratio_15,
        "abs_streak": streak,
    }, index=df_1m.index)


FEATURE_COLUMNS = (
    "abs_ratio_5",
    "abs_ratio_15",
    "abs_streak",
)
