"""A5 — Compression Breakout.

Thesis: Markets cycle through low-vol consolidation → vol expansion. A run
of bars with decreasing range followed by an expansion bar with directional
close tends to mark the start of a directional move.

Detection conditions:
  * Compression: last 20 bars had avg range < 0.7 × ATR(60min) avg
  * Vol declining: recent realized vol below recent median
  * Expansion bar: current bar range > 1.5 × ATR
  * Directional close: close in top/bottom 25% of bar's range

Outputs (5 features):
  scomp_active                  1.0 if compression-then-expansion detected
  scomp_strength                0-1 continuous match score
  scomp_direction               +1/-1 from expansion-bar direction
  scomp_compression_ratio       compression-bars avg range / baseline ATR
  scomp_expansion_magnitude     expansion-bar range in ATR units
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from altus.features.families.setup_utils import EPS, atr_safe, clip_clamp


def compute(df_primary: pd.DataFrame, df_1m: pd.DataFrame | None = None) -> pd.DataFrame:
    df_in = df_primary  # compression detection uses the primary candle base
    n = len(df_in)
    idx = df_in.index
    opens = df_in["open"].to_numpy(dtype=np.float64)
    highs = df_in["high"].to_numpy(dtype=np.float64)
    lows = df_in["low"].to_numpy(dtype=np.float64)
    closes = df_in["close"].to_numpy(dtype=np.float64)

    # ATR baseline. Use 60-bar window so it reflects "hourly" vol context.
    atr_60 = atr_safe(df_in, n=60)
    bar_range = highs - lows

    # Compression metric: rolling 20-bar avg range, normalized by atr_60.
    compression_window = 20
    avg_range_20 = pd.Series(bar_range, index=idx).rolling(
        compression_window, min_periods=10
    ).mean().shift(1).to_numpy()
    compression_ratio = np.where(atr_60 > 0, avg_range_20 / atr_60, 1.0)
    is_compressed = compression_ratio < 0.7

    # Current bar's expansion magnitude
    expansion_magnitude = np.where(atr_60 > 0, bar_range / atr_60, 0.0)
    is_expansion = expansion_magnitude > 1.5

    # Directional close: close in top/bottom 25% of bar
    bar_pos = np.where(bar_range > EPS, (closes - lows) / bar_range, 0.5)
    is_directional = (bar_pos >= 0.75) | (bar_pos <= 0.25)

    active = (is_compressed & is_expansion & is_directional).astype(np.int32)
    direction = np.where(active.astype(bool),
                          np.where(bar_pos >= 0.75, 1, -1),
                          0).astype(np.int32)

    # Strength
    compression_intensity = clip_clamp(1.0 - compression_ratio / 0.7, 0.0, 1.0)
    expansion_strength = clip_clamp((expansion_magnitude - 1.5) / 1.5, 0.0, 1.0)
    close_quality = clip_clamp(np.abs(bar_pos - 0.5) * 2.0, 0.0, 1.0)
    strength = (0.4 * active.astype(np.float32) + 0.25 * compression_intensity +
                0.20 * expansion_strength + 0.15 * close_quality)
    strength = clip_clamp(strength, 0.0, 1.0)

    return pd.DataFrame({
        "scomp_active": active.astype(np.float32),
        "scomp_strength": strength,
        "scomp_direction": direction.astype(np.float32),
        "scomp_compression_ratio": clip_clamp(compression_ratio.astype(np.float32), 0.0, 2.0),
        "scomp_expansion_magnitude": clip_clamp(expansion_magnitude.astype(np.float32), 0.0, 5.0),
    }, index=idx)


FEATURE_COLUMNS = (
    "scomp_active",
    "scomp_strength",
    "scomp_direction",
    "scomp_compression_ratio",
    "scomp_expansion_magnitude",
)
