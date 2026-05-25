"""Rolling-window OHLCV — the "trailing 3m bar updated every 1m" primitive.

Why this exists: 1m MNQ OHLCV is dominated by microstructure noise — bid/ask
bounce, single-order ticks, spread crossing. Aggregating to 3m kills most of
that noise. But fixed 3m bars (00, 03, 06, …) lose the entry-time granularity
that intraday trading needs. The rolling 3m bar — recomputed every 1m — gives
both: each row's OHLCV summarizes the trailing 3 minutes, indexed at every
1m timestamp.

Properties at minute T (with `window_min=3`):
    open   = open of the bar that started at T-window_min+1 minutes
    high   = max high in (T-window_min, T]
    low    = min low  in (T-window_min, T]
    close  = close at T
    volume = sum of volumes in (T-window_min, T]

The output index matches df_1m.index 1:1 so any existing causal-shift(1)
machinery still works — feature at T+1 uses rolling bar at T, which itself
consumed data up to T's close. No leakage.

First `window_min - 1` rows are NaN (insufficient history for a full window).
The feature pipeline's `.dropna()` removes them; we accept ~2 rows of warmup.
"""
from __future__ import annotations

import pandas as pd


def build_rolling_ohlcv(df_1m: pd.DataFrame, window_min: int = 3) -> pd.DataFrame:
    """Convert raw 1m OHLCV into rolling-window OHLCV bars.

    Parameters
    ----------
    df_1m : DataFrame with columns ['open', 'high', 'low', 'close', 'volume'],
            indexed at 1-minute resolution (UTC).
    window_min : Width of the rolling window in minutes. 1 returns a copy of
                 the input (no aggregation). Default 3 — see module docstring.

    Returns
    -------
    DataFrame with the same columns and index as df_1m, where each row is the
    OHLCV summary of the trailing `window_min` minutes ending at that row.
    """
    if window_min < 1:
        raise ValueError(f"window_min must be >= 1, got {window_min}")
    if window_min == 1:
        return df_1m.copy()

    out = pd.DataFrame(index=df_1m.index)
    # close at T is just the 1m close at T — the rolling window ends here.
    out["close"] = df_1m["close"]
    # open at T is the open of the bar window_min-1 rows back (start of window).
    out["open"] = df_1m["open"].shift(window_min - 1)
    # high/low: rolling max/min over `window_min` bars, min_periods enforces
    # warmup discipline so partial windows yield NaN (caught by .dropna()).
    out["high"] = df_1m["high"].rolling(window_min, min_periods=window_min).max()
    out["low"] = df_1m["low"].rolling(window_min, min_periods=window_min).min()
    out["volume"] = df_1m["volume"].rolling(window_min, min_periods=window_min).sum()

    return out[["open", "high", "low", "close", "volume"]]
