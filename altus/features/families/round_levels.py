"""Family E1 (Phase E): Round-number proximity. Answers Q11 (resting liquidity).

Why this matters: futures traders cluster stops at round numbers — every 100,
50, and 25 pts on MNQ. The liquidity pool at each round level is larger than at
non-round levels for purely psychological reasons (humans round). Price gets
drawn to round numbers and accelerates through them when swept.

Features (4 total, all in ATR units so they normalize across vol regimes):
  • rnd_dist_to_100_atr      signed distance to nearest 100-pt round (negative if above)
  • rnd_dist_to_50_atr       signed distance to nearest 50-pt round
  • rnd_dist_to_25_atr       signed distance to nearest 25-pt round
  • rnd_in_round_zone        1.0 if within 0.1 ATR of any 25/50/100 level, else 0.0

CAUSALITY: features at bar T computed from close[T] only — no history needed.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


EPS = 1e-9


def _atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    prev_close = df["close"].shift(1)
    tr = pd.concat(
        [(df["high"] - df["low"]).abs(),
         (df["high"] - prev_close).abs(),
         (df["low"] - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    return tr.rolling(n, min_periods=2).mean()


def _signed_dist_to_round(price: np.ndarray, step: float) -> np.ndarray:
    """Signed distance to nearest multiple of step (negative if level is above price)."""
    nearest = np.round(price / step) * step
    return nearest - price  # negative = level above, positive = level below


def compute(df_1m: pd.DataFrame) -> pd.DataFrame:
    close = df_1m["close"].to_numpy(dtype=np.float64)
    atr = _atr(df_1m, n=14).replace(0, np.nan).ffill().fillna(1.0).to_numpy(dtype=np.float64)

    dist_100 = _signed_dist_to_round(close, 100.0)
    dist_50 = _signed_dist_to_round(close, 50.0)
    dist_25 = _signed_dist_to_round(close, 25.0)

    dist_100_atr = dist_100 / np.maximum(atr, EPS)
    dist_50_atr = dist_50 / np.maximum(atr, EPS)
    dist_25_atr = dist_25 / np.maximum(atr, EPS)

    in_zone = (
        (np.abs(dist_100_atr) < 0.1)
        | (np.abs(dist_50_atr) < 0.1)
        | (np.abs(dist_25_atr) < 0.1)
    ).astype(np.float32)

    return pd.DataFrame({
        "rnd_dist_to_100_atr": dist_100_atr.astype(np.float32),
        "rnd_dist_to_50_atr": dist_50_atr.astype(np.float32),
        "rnd_dist_to_25_atr": dist_25_atr.astype(np.float32),
        "rnd_in_round_zone": in_zone,
    }, index=df_1m.index)


FEATURE_COLUMNS = (
    "rnd_dist_to_100_atr",
    "rnd_dist_to_50_atr",
    "rnd_dist_to_25_atr",
    "rnd_in_round_zone",
)
