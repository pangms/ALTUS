"""Family E2 (Phase E): Multi-timeframe trend alignment. Answers Q8 (timeframe in control).

Why this matters: a setup means different things when all timeframes agree vs
when they conflict. Bullish 1m setup with bullish 4h trend = high-quality long.
Bullish 1m setup against bearish 4h trend = either a counter-trend trade (low
expectancy unless inflection is confirmed) or a fade setup. The model needs to
see the alignment intersection directly, not infer it from raw prices.

Features (5 total):
  • mtf_trend_sign_5m       trend direction at 5m (signed EMA slope, clipped)
  • mtf_trend_sign_15m      trend direction at 15m
  • mtf_trend_sign_60m      trend direction at 60m
  • mtf_trend_sign_240m     trend direction at 4h
  • mtf_alignment_score     mean of the four signs (in [-1, 1])

CAUSALITY: per-TF resample uses label=left/closed=left + shift(freq=tf_min)
+ ffill, matching the trend_hurst pattern. At bar T, the value for a higher TF
reflects the most recently CLOSED bar at that TF.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from altus.features.families.trend_hurst import _resample_ohlcv


def _trend_sign(df_htf: pd.DataFrame, ema_span: int = 20) -> pd.Series:
    """Signed EMA slope, clipped to [-1, 1]. Positive = up, negative = down."""
    close = df_htf["close"]
    ema = close.ewm(span=ema_span, adjust=False).mean()
    slope = ema - ema.shift(1)
    # Normalize by recent range so the magnitude is scale-free
    range_proxy = (df_htf["high"] - df_htf["low"]).rolling(20, min_periods=2).mean()
    norm_slope = slope / range_proxy.replace(0, np.nan)
    return np.tanh(norm_slope.fillna(0.0) * 5.0)  # squash to [-1, 1]


def _project_to_1m(htf_sign: pd.Series, df_1m: pd.DataFrame, tf_min: int) -> pd.Series:
    """Project HTF series to 1m grid using shift+ffill, matching trend_hurst convention."""
    shifted = htf_sign.shift(freq=pd.Timedelta(minutes=tf_min))
    union_idx = shifted.index.union(df_1m.index).sort_values()
    return shifted.reindex(union_idx).ffill().reindex(df_1m.index).fillna(0.0)


def compute(df_1m: pd.DataFrame) -> pd.DataFrame:
    out: dict[str, pd.Series] = {}
    tfs = [(5, "5m"), (15, "15m"), (60, "60m"), (240, "240m")]
    for tf_min, label in tfs:
        htf = _resample_ohlcv(df_1m, tf_min)
        if len(htf) < 25:
            out[f"mtf_trend_sign_{label}"] = pd.Series(0.0, index=df_1m.index, dtype=np.float32)
        else:
            sign = _trend_sign(htf, ema_span=20)
            out[f"mtf_trend_sign_{label}"] = _project_to_1m(sign, df_1m, tf_min).astype(np.float32)

    sign_cols = [out[f"mtf_trend_sign_{label}"] for _, label in tfs]
    out["mtf_alignment_score"] = pd.concat(sign_cols, axis=1).mean(axis=1).astype(np.float32)

    return pd.DataFrame(out, index=df_1m.index)


FEATURE_COLUMNS = (
    "mtf_trend_sign_5m",
    "mtf_trend_sign_15m",
    "mtf_trend_sign_60m",
    "mtf_trend_sign_240m",
    "mtf_alignment_score",
)
