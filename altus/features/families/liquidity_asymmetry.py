"""Family E9 (Phase E): Liquidity asymmetry. Answers Q28 (asymmetry → directional gravity).

Why this matters: where the liquidity pools are above vs below current price
creates a structural directional bias from geometry alone. Three untouched
stop pools below and one above = downward gravity, regardless of trend. The
engine should know whether the liquidity map is symmetric or pulling one way.

Built on top of liquidity_zones outputs — we don't duplicate the detection,
we summarize the asymmetry.

Features (3 total):
  • la_count_asymmetry      (n_above - n_below) / (n_above + n_below + 1)
  • la_dist_asymmetry       (min_dist_below - min_dist_above) in ATR units
                              positive = nearer untouched pool ABOVE (upward gravity)
                              negative = nearer untouched pool BELOW (downward gravity)
  • la_strength_imbalance   (closest_below_strength - closest_above_strength) in [-1,1]

CAUSALITY: re-runs liquidity_zones internally, which is already verified causal.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from altus.features.families.liquidity_zones import compute as compute_liquidity_zones


NEEDS_RAW_1M = True  # delegates to liquidity_zones which needs raw 1m


def compute(df_primary: pd.DataFrame, df_1m: pd.DataFrame | None = None) -> pd.DataFrame:
    if df_1m is None:
        df_1m = df_primary  # back-compat / PRIMARY_WINDOW_MIN=1 path
    lz = compute_liquidity_zones(df_primary, df_1m=df_1m)

    # lz exposes per-TF dist above/below in ATR + closest_zone. We use the
    # min-distance asymmetry across timeframes as a proxy for "geometric pull".
    above_cols = [c for c in lz.columns if "dist_above" in c]
    below_cols = [c for c in lz.columns if "dist_below" in c]

    # Min distance above/below across all TFs (smaller = nearer)
    min_above = lz[above_cols].min(axis=1)
    min_below = lz[below_cols].min(axis=1)

    # Count asymmetry: how many TFs have a closer-than-6-ATR pool on each side
    # (6.0 is the cap value used inside liquidity_zones for "no pool")
    n_above = (lz[above_cols] < 5.5).sum(axis=1).astype(np.float32)
    n_below = (lz[below_cols] < 5.5).sum(axis=1).astype(np.float32)
    count_asym = ((n_above - n_below) / (n_above + n_below + 1.0)).astype(np.float32)

    # Distance asymmetry: positive = above pool is nearer (price gravity up)
    dist_asym = (min_below - min_above).clip(-6.0, 6.0).astype(np.float32)

    # Strength imbalance proxy: use 1/(min_dist+0.5) as "pull strength"
    strength_above = 1.0 / (min_above + 0.5)
    strength_below = 1.0 / (min_below + 0.5)
    strength_imb = ((strength_below - strength_above) / (strength_below + strength_above + 1e-6)).astype(np.float32)

    return pd.DataFrame({
        "la_count_asymmetry": count_asym,
        "la_dist_asymmetry": dist_asym,
        "la_strength_imbalance": strength_imb,
    }, index=df_1m.index)


FEATURE_COLUMNS = (
    "la_count_asymmetry",
    "la_dist_asymmetry",
    "la_strength_imbalance",
)
