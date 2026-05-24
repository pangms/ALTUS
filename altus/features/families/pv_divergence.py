"""Family E4 (Phase E): Price-volume divergence. Answers Q4 (volume confirming/diverging).

Why this matters: when price rises on declining volume, the move is losing
sponsorship. When price rises on surging volume at a known extreme, that's
often absorption (large players selling into retail buying). Explicit
correlation between price-direction and volume-direction lets the model
discriminate these cases cleanly.

Features (3 total):
  • pvd_corr_30           rolling 30-bar sign-correlation of price-direction × volume-direction
  • pvd_corr_120          same over 120 bars
  • pvd_disagreement      1.0 if 30-bar and 120-bar corrs have opposite signs (regime change in progress)

CAUSALITY: rolling.corr() over windows ending at bar T uses data ≤ T;
structural orchestrator shift(1) ensures feature-at-T uses data ≤ T-1.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def compute(df_1m: pd.DataFrame) -> pd.DataFrame:
    close = df_1m["close"]
    volume = df_1m["volume"]

    price_dir = np.sign(close.diff()).fillna(0.0)
    # Volume "direction" = deviation from rolling mean (volume Z-score sign)
    vol_mean = volume.rolling(100, min_periods=10).mean()
    vol_dir = np.sign((volume - vol_mean).fillna(0.0))

    corr_30 = price_dir.rolling(30, min_periods=10).corr(vol_dir).fillna(0.0).astype(np.float32)
    corr_120 = price_dir.rolling(120, min_periods=20).corr(vol_dir).fillna(0.0).astype(np.float32)

    disagree = (np.sign(corr_30) * np.sign(corr_120) < 0).astype(np.float32)

    return pd.DataFrame({
        "pvd_corr_30": corr_30,
        "pvd_corr_120": corr_120,
        "pvd_disagreement": disagree,
    }, index=df_1m.index)


FEATURE_COLUMNS = (
    "pvd_corr_30",
    "pvd_corr_120",
    "pvd_disagreement",
)
