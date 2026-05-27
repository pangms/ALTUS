"""Stop Pool — fuel detector.

Composite feature answering: "How much resting liquidity (= stop cluster)
sits beyond the recent swing in each direction?"

Stops cluster just past swing highs (where breakout-stoppers and short stops
sit) and just past swing lows (mirror). When price approaches a swing
extreme, the size of the pool BEYOND that extreme tells you how much
"fuel" a directional breakout would have.

For setups that EXPECT continuation past a swing (ORB, compression breakout,
trend pullback): bigger pool above = bigger expected continuation magnitude.
For setups that EXPECT a sweep + reversal (failed sweep, failed auction):
bigger pool above = bigger sweep target = bigger reversal fuel afterward.

Same feature, two interpretations — the model learns which is which by
setup_id × pool_size interaction.

Features (5 total):
  sp_pool_above_size_atr        estimated stop cluster size beyond nearest swing high
  sp_pool_below_size_atr        same for swing low
  sp_trigger_distance_above_atr distance to the cluster-trigger price (= swing high)
  sp_trigger_distance_below_atr distance to swing low
  sp_pool_imminent              binary: within 0.3 ATR of either trigger

CAUSALITY: Pool sizing uses swing-window highs/lows (rolling, past only).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from altus.features.families.setup_utils import EPS, atr_safe, clip_clamp


def _estimate_pool_above(highs: np.ndarray, lows: np.ndarray, swing_high: float,
                          atr: float, look_after: int = 60) -> float:
    """Heuristic: stop cluster size above a swing high =
    the average true-range continuation we'd EXPECT if stops fire.

    Without tick data we can't measure exact stop density. Proxy:
    estimate based on the SIZE of recent moves into similar swing highs
    and how far they continued after breaking.

    Simpler heuristic used here: pool size = function of (swing prominence,
    recent average range). Bigger swings + more vol = bigger pool.
    """
    # Use swing prominence + ATR as proxy
    prominence_pts = swing_high - float(np.min(lows))  # rough swing-low-to-high range
    pool_atr = (prominence_pts / max(atr, EPS)) * 0.3  # 30% of swing prominence as pool estimate
    return float(np.clip(pool_atr, 0.5, 5.0))


def compute(df_primary: pd.DataFrame, df_1m: pd.DataFrame | None = None) -> pd.DataFrame:
    if df_1m is None:
        df_1m = df_primary
    n = len(df_1m)
    closes = df_1m["close"].to_numpy(dtype=np.float64)
    highs = df_1m["high"].to_numpy(dtype=np.float64)
    lows = df_1m["low"].to_numpy(dtype=np.float64)
    atr_arr = atr_safe(df_1m, n=14)

    # Recent swing extremes — 60-bar window for "tactical" swings (1hr)
    swing_window = 60
    sh_arr = (
        df_1m["high"].rolling(swing_window, min_periods=10).max().shift(1).to_numpy()
    )
    sl_arr = (
        df_1m["low"].rolling(swing_window, min_periods=10).min().shift(1).to_numpy()
    )

    # Per-bar pool size estimates
    pool_above = np.zeros(n, dtype=np.float64)
    pool_below = np.zeros(n, dtype=np.float64)
    trig_dist_above = np.full(n, 10.0, dtype=np.float64)
    trig_dist_below = np.full(n, 10.0, dtype=np.float64)

    for i in range(swing_window, n):
        atr_i = max(float(atr_arr[i]), EPS)
        sh = sh_arr[i]
        sl = sl_arr[i]
        if not np.isnan(sh) and sh > closes[i]:
            trig_dist_above[i] = (sh - closes[i]) / atr_i
            # Pool size: range of the swing × 30%, clipped
            prominence = sh - sl if not np.isnan(sl) else sh - closes[i]
            pool_above[i] = float(np.clip((prominence / atr_i) * 0.3, 0.5, 5.0))
        if not np.isnan(sl) and sl < closes[i]:
            trig_dist_below[i] = (closes[i] - sl) / atr_i
            prominence = sh - sl if not np.isnan(sh) else closes[i] - sl
            pool_below[i] = float(np.clip((prominence / atr_i) * 0.3, 0.5, 5.0))

    # Imminent: within 0.3 ATR of either trigger
    imminent = ((trig_dist_above < 0.3) | (trig_dist_below < 0.3)).astype(np.float32)

    return pd.DataFrame({
        "sp_pool_above_size_atr": clip_clamp(pool_above.astype(np.float32), 0.0, 5.0),
        "sp_pool_below_size_atr": clip_clamp(pool_below.astype(np.float32), 0.0, 5.0),
        "sp_trigger_distance_above_atr": clip_clamp(trig_dist_above.astype(np.float32), 0.0, 10.0),
        "sp_trigger_distance_below_atr": clip_clamp(trig_dist_below.astype(np.float32), 0.0, 10.0),
        "sp_pool_imminent": imminent,
    }, index=df_1m.index)


FEATURE_COLUMNS = (
    "sp_pool_above_size_atr",
    "sp_pool_below_size_atr",
    "sp_trigger_distance_above_atr",
    "sp_trigger_distance_below_atr",
    "sp_pool_imminent",
)
