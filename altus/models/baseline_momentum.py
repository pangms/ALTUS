"""Naive momentum baseline — NOT a competing model, a sanity-check rule.

Purpose: if the hybrid's predictions correlate strongly with this rule, we
haven't actually learned anything beyond "go with recent momentum." If the
hybrid outperforms it cleanly, the ML is adding genuine value.

Rule: classify long_tp=1 if last-K-bar return is positive AND > threshold * ATR,
mirror for short_tp. Otherwise no signal (both predict 0).

We score it like the ML model — same eval metrics, same splits.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class MomentumConfig:
    lookback_bars: int = 10
    atr_period: int = 14
    threshold_atr_mult: float = 0.5  # require move > 0.5 * ATR


def momentum_signal(
    df_1m: pd.DataFrame,
    cfg: MomentumConfig | None = None,
) -> pd.DataFrame:
    """Compute naive long/short momentum signals (in [0, 1]) for each bar.

    Output columns aligned to df_1m.index:
      - long_tp_pred:  pseudo-probability in [0, 1] for long-TP signal
      - short_tp_pred: pseudo-probability in [0, 1] for short-TP signal

    The 'probability' is just min(1, |move| / threshold_move) clipped — enough
    for AUC/ranking metrics to work, but it's not a learned model.
    """
    cfg = cfg or MomentumConfig()
    close = df_1m["close"]
    # Returns over the lookback window
    move = close - close.shift(cfg.lookback_bars)

    # ATR (same form as in features.pipeline._atr but inlined to avoid import cycle)
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            (df_1m["high"] - df_1m["low"]).abs(),
            (df_1m["high"] - prev_close).abs(),
            (df_1m["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr = tr.rolling(cfg.atr_period, min_periods=cfg.atr_period).mean()

    threshold = cfg.threshold_atr_mult * atr * cfg.lookback_bars ** 0.5  # vol-scaled
    # Convert magnitude to a [0, 1] pseudo-prob for ranking.
    # Positive move -> long signal; negative -> short signal.
    long_score = (move / (threshold + 1e-9)).clip(lower=0, upper=2.0) / 2.0
    short_score = (-move / (threshold + 1e-9)).clip(lower=0, upper=2.0) / 2.0

    out = pd.DataFrame(
        {"long_tp_pred": long_score, "short_tp_pred": short_score},
        index=df_1m.index,
    )
    # Causal: at row T we'd only know everything through T-1. Shift by 1.
    return out.shift(1).fillna(0.0)
