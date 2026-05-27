"""Vol Sweet-Spot — per-setup vol-regime fitness booster.

Each setup has an ideal volatility regime. ORB works in moderate-to-high
vol; EOD reversion works in low vol; failed-sweep works in any vol but
prefers expansion phases. Same setup, different vol regimes, different
historical edge.

This family computes a fitness score [0, 1] per setup × current vol regime.
Feeds L2 as a confidence modulator — boosts conviction when active setup
matches its preferred vol regime.

Hardcoded mapping per setup (calibratable from training data later):

  | Setup | Low vol (z<-0.5) | Med vol | High vol (z>+0.5) |
  |-------|------------------|---------|--------------------|
  | sfs   | 0.7              | 1.0     | 0.9                |
  | sfa   | 1.0              | 0.9     | 0.5                |
  | sld   | 0.8              | 1.0     | 0.7                |
  | orb   | 0.3              | 0.8     | 1.0                |
  | svwap | 0.4              | 1.0     | 0.7                |
  | spb   | 0.6              | 1.0     | 0.8                |
  | scomp | 1.0              | 0.5     | 0.2                |
  | seod  | 0.9              | 0.9     | 0.7                |

Features (1 column per setup × match score = 8 features):
  vss_match_sfs / vss_match_sfa / ... / vss_match_seod
  vss_avg_match    — average match across all 8 setups
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from altus.features.families.setup_utils import EPS, clip_clamp


# Vol regime classifications: based on vol_regime_score or vol_realized_*.
# We use a simple z-score-on-volatility proxy here: realized 30-min vol vs
# its 240-bar (4hr) rolling baseline.

# Per-setup fitness matrix: [low_vol, med_vol, high_vol] → match score
FITNESS_MATRIX = {
    "sfs":   [0.7, 1.0, 0.9],   # failed sweep — versatile, prefers normal-to-high
    "sfa":   [1.0, 0.9, 0.5],   # failed auction — works best in lower vol
    "sld":   [0.8, 1.0, 0.7],   # level defense — normal vol ideal
    "orb":   [0.3, 0.8, 1.0],   # ORB — needs high opening vol
    "svwap": [0.4, 1.0, 0.7],   # VWAP — trending sessions, mod vol
    "spb":   [0.6, 1.0, 0.8],   # pullback — normal-to-high vol
    "scomp": [1.0, 0.5, 0.2],   # compression — must START in low vol
    "seod":  [0.9, 0.9, 0.7],   # EOD reversion — works in low-to-mod
}


def compute(df_primary: pd.DataFrame, df_1m: pd.DataFrame | None = None) -> pd.DataFrame:
    """Computes per-setup vol-regime fitness per bar."""
    if df_1m is None:
        df_1m = df_primary
    n = len(df_1m)
    idx = df_1m.index

    # Compute realized 30-min vol as a simple z-score
    close_s = df_1m["close"]
    ret_s = np.log(close_s / close_s.shift(1)).fillna(0.0)
    vol_30 = ret_s.rolling(30, min_periods=10).std().shift(1).to_numpy()
    vol_baseline = pd.Series(vol_30, index=idx).rolling(240, min_periods=50).mean().shift(1).to_numpy()
    vol_baseline_std = pd.Series(vol_30, index=idx).rolling(240, min_periods=50).std().shift(1).to_numpy()
    vol_z = (vol_30 - vol_baseline) / np.maximum(vol_baseline_std, EPS)
    vol_z = np.nan_to_num(vol_z, nan=0.0, posinf=3.0, neginf=-3.0)
    vol_z = np.clip(vol_z, -3.0, 3.0)

    # Classify each bar into low/med/high
    low_mask = vol_z < -0.5
    high_mask = vol_z > 0.5
    med_mask = ~low_mask & ~high_mask

    # Per-setup match score
    out = {}
    avg_scores = np.zeros(n, dtype=np.float32)
    for sid, fitness in FITNESS_MATRIX.items():
        scores = np.full(n, fitness[1], dtype=np.float32)  # default = med
        scores[low_mask] = fitness[0]
        scores[high_mask] = fitness[2]
        out[f"vss_match_{sid}"] = scores
        avg_scores += scores / len(FITNESS_MATRIX)

    out["vss_avg_match"] = avg_scores

    return pd.DataFrame(out, index=idx)


FEATURE_COLUMNS = (
    "vss_match_sfs", "vss_match_sfa", "vss_match_sld", "vss_match_orb",
    "vss_match_svwap", "vss_match_spb", "vss_match_scomp", "vss_match_seod",
    "vss_avg_match",
)
