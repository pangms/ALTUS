"""Multi-timeframe feature pipeline for ALTUS Layer 1.

Design contract — this is the most important guarantee in the file:

    FEATURES AT ROW T USE ONLY INFORMATION KNOWABLE AT THE START OF BAR T.
    LABELS AT ROW T ARE COMPUTED FROM BAR T FORWARD (using bar T's open as
    the entry price).

The convention: bar timestamps are START-of-bar (verified from the MNQ
parquet). So at the start of bar T, the most recently completed bar is T-1.
We compute features naively then `.shift(1)` the whole matrix so row T
contains only what was knowable at the end of bar T-1 — i.e., at the moment
we'd make a "trade at the open of bar T" decision.

Multi-timeframe alignment uses the same principle: a 3m bar starting at S
isn't available until S+3min. We resample to higher TFs, shift forward by
tf_min, then reindex to the 1m grid with forward-fill so each 1m row reads
the most-recently-completed higher-TF bar.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

from altus.config import ROLL_NORM_WINDOW, TIMEFRAMES_MIN


EPS = 1e-9


# ---------------------------------------------------------------------------
# Per-timeframe primitives
# ---------------------------------------------------------------------------

def _resample_ohlcv(df_1m: pd.DataFrame, tf_min: int) -> pd.DataFrame:
    """Aggregate 1-min OHLCV to a higher timeframe. Bars labeled at START."""
    if tf_min == 1:
        return df_1m.copy()
    rule = f"{tf_min}min"
    out = df_1m.resample(rule, label="left", closed="left").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    )
    return out.dropna(how="any")


def _atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    """Average true range — used as a vol-normalizer, not a feature itself."""
    prev_close = df["close"].shift(1)
    tr = pd.concat(
        [
            (df["high"] - df["low"]).abs(),
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.rolling(n, min_periods=n).mean()


def _rsi(close: pd.Series, n: int = 14) -> pd.Series:
    """RSI normalized to [-1, 1] (instead of standard [0, 100])."""
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(n, min_periods=n).mean()
    loss = (-delta.clip(upper=0)).rolling(n, min_periods=n).mean()
    rs = gain / (loss + EPS)
    rsi_0_100 = 100 - 100 / (1 + rs)
    return (rsi_0_100 - 50) / 50


def _rolling_zscore(s: pd.Series, window: int) -> pd.Series:
    mu = s.rolling(window, min_periods=window // 4).mean()
    sd = s.rolling(window, min_periods=window // 4).std()
    return (s - mu) / (sd + EPS)


def _features_one_tf(ohlcv: pd.DataFrame, tf_min: int) -> pd.DataFrame:
    """Compute the per-timeframe feature block. All features are unit-free or
    self-normalized so they're comparable across timeframes and regimes."""
    c, o, h, l, v = ohlcv["close"], ohlcv["open"], ohlcv["high"], ohlcv["low"], ohlcv["volume"]
    rng = (h - l)
    atr14 = _atr(ohlcv, 14)
    atr_safe = atr14.replace(0, np.nan)

    log_ret = np.log(c / c.shift(1))
    # Vol-normalized return — keeps the model focused on *abnormal* moves
    # rather than absolute magnitude (which depends on price level and regime).
    vol_norm_ret = log_ret / (atr_safe / c).replace(0, np.nan)

    body = (c - o) / (rng + EPS)                 # bar direction & decisiveness
    upper_wick = (h - np.maximum(o, c)) / (rng + EPS)
    lower_wick = (np.minimum(o, c) - l) / (rng + EPS)
    range_pct = rng / c                          # bar size as % of price
    range_z = _rolling_zscore(range_pct, 240 // max(tf_min, 1))

    rsi = _rsi(c, 14)
    rsi_fast = _rsi(c, 5)

    # Distance of close from short/long EMA (mean-reversion / trend signal)
    ema_fast = c.ewm(span=8, adjust=False).mean()
    ema_slow = c.ewm(span=32, adjust=False).mean()
    dist_fast = (c - ema_fast) / (atr_safe + EPS)
    dist_slow = (c - ema_slow) / (atr_safe + EPS)
    trend = (ema_fast - ema_slow) / (atr_safe + EPS)

    # Volume regime — z-score across recent same-timeframe window
    vol_z = _rolling_zscore(v.astype(float), 240 // max(tf_min, 1))

    feats = pd.DataFrame(
        {
            f"tf{tf_min}_logret": log_ret,
            f"tf{tf_min}_vnret": vol_norm_ret,
            f"tf{tf_min}_body": body,
            f"tf{tf_min}_uwick": upper_wick,
            f"tf{tf_min}_lwick": lower_wick,
            f"tf{tf_min}_rangepct": range_pct,
            f"tf{tf_min}_rangez": range_z,
            f"tf{tf_min}_rsi14": rsi,
            f"tf{tf_min}_rsi5": rsi_fast,
            f"tf{tf_min}_distfast": dist_fast,
            f"tf{tf_min}_distslow": dist_slow,
            f"tf{tf_min}_trend": trend,
            f"tf{tf_min}_volz": vol_z,
        },
        index=ohlcv.index,
    )
    return feats


# ---------------------------------------------------------------------------
# Multi-timeframe alignment
# ---------------------------------------------------------------------------

def _align_to_1m_grid(higher_tf_feats: pd.DataFrame, tf_min: int, grid: pd.DatetimeIndex) -> pd.DataFrame:
    """Shift higher-TF features forward by tf_min so they become 'available'
    at their close, then reindex onto the 1m grid with forward fill.

    Result: at 1m timestamp T, the higher-TF columns reflect the most recently
    COMPLETED higher-TF bar before T. No look-ahead.
    """
    if tf_min == 1:
        return higher_tf_feats.reindex(grid)
    shifted = higher_tf_feats.shift(freq=pd.Timedelta(minutes=tf_min))
    return shifted.reindex(grid).ffill()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

@dataclass
class FeatureSpec:
    timeframes_min: tuple[int, ...] = TIMEFRAMES_MIN
    roll_norm_window: int = ROLL_NORM_WINDOW
    causal_shift: bool = True       # shift the whole matrix by 1 — see module docstring


def build_features(df_1m: pd.DataFrame, spec: FeatureSpec | None = None) -> pd.DataFrame:
    """Build the multi-timeframe feature matrix on the 1-min grid.

    Input: df_1m with columns ['open','high','low','close','volume'], UTC-indexed.
    Output: causally-shifted, NaN-trimmed feature DataFrame on the same 1m index.
    """
    spec = spec or FeatureSpec()
    grid = df_1m.index

    all_blocks: list[pd.DataFrame] = []
    for tf in spec.timeframes_min:
        ohlcv_tf = _resample_ohlcv(df_1m, tf)
        feats_tf = _features_one_tf(ohlcv_tf, tf)
        feats_aligned = _align_to_1m_grid(feats_tf, tf, grid)
        all_blocks.append(feats_aligned)

    X = pd.concat(all_blocks, axis=1)

    # Replace inf -> NaN before any downstream processing
    X = X.replace([np.inf, -np.inf], np.nan)

    if spec.causal_shift:
        # Critical: features at row T must NOT include bar T itself, because
        # at the moment we trade at the open of bar T, bar T hasn't happened.
        X = X.shift(1)

    # Drop the warmup region where rolling windows haven't filled in.
    X = X.dropna(how="any")
    return X


def feature_column_count(spec: FeatureSpec | None = None) -> int:
    """Number of features the pipeline will produce per row, given a spec."""
    spec = spec or FeatureSpec()
    # 13 per-timeframe features (see _features_one_tf)
    return 13 * len(spec.timeframes_min)
