"""Family 4: Momentum exhaustion features.

Why this matters: this family directly speaks to Layer 1's short-term job.
"Is the recent move running out of gas?" is information that doesn't show up
cleanly in raw OHLCV. Exhaustion signals (RSI divergences, BB-extreme positions,
fading volume on extension) are exactly when stop hunts and reversals happen.

Features:
  * RSI divergence on 5m and 15m bars (regular bearish/bullish over recent swings)
  * BB position on 15m — z-score from rolling 20-bar mean
  * Consecutive same-direction 1m bars (signed count, signals exhaustion at extremes)
  * Volume decay slope on recent same-direction bars (volume fading on extension)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from altus.features.families.trend_hurst import _resample_ohlcv


EPS = 1e-9


def _rsi(close: pd.Series, n: int) -> pd.Series:
    """Standard 0-100 RSI."""
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(n, min_periods=n).mean()
    loss = (-delta.clip(upper=0)).rolling(n, min_periods=n).mean()
    rs = gain / (loss + EPS)
    return 100 - 100 / (1 + rs)


def _detect_divergence(price: np.ndarray, rsi: np.ndarray, lookback: int = 15) -> np.ndarray:
    """Detect RSI divergence over a rolling window.

    Returns int8 array same length as input:
      +1 if BULLISH divergence detected (lower low in price + higher low in RSI)
      -1 if BEARISH divergence detected (higher high in price + lower high in RSI)
       0 otherwise

    Method: compare current bar's local high/low against the previous local
    high/low in the lookback window. Conservative: only flags when both swings
    are real local extrema.
    """
    n = len(price)
    out = np.zeros(n, dtype=np.int8)
    if n < lookback * 2:
        return out

    for t in range(lookback * 2, n):
        # Find the two most recent local highs/lows in price within the lookback
        window_price = price[t - lookback * 2 : t + 1]
        window_rsi = rsi[t - lookback * 2 : t + 1]
        if not (np.all(np.isfinite(window_price)) and np.all(np.isfinite(window_rsi))):
            continue

        # Local high at index i: window_price[i] is the max of a small neighborhood
        # We just compare two halves of the window.
        h1 = lookback - 1 + window_price[: lookback].argmax()
        h2 = lookback + window_price[lookback :].argmax()
        l1 = lookback - 1 + window_price[: lookback].argmin()
        l2 = lookback + window_price[lookback :].argmin()

        # Bearish: higher high in price, lower high in RSI
        if window_price[h2] > window_price[h1] and window_rsi[h2] < window_rsi[h1]:
            out[t] = -1
        # Bullish: lower low in price, higher low in RSI
        elif window_price[l2] < window_price[l1] and window_rsi[l2] > window_rsi[l1]:
            out[t] = 1

    return out


def _bb_position(close: pd.Series, n: int = 20) -> pd.Series:
    """Z-score of close vs rolling N-bar mean (BB-like position in std devs)."""
    mu = close.rolling(n, min_periods=n).mean()
    sd = close.rolling(n, min_periods=n).std()
    return (close - mu) / (sd + EPS)


def _signed_consecutive_color(open_: pd.Series, close: pd.Series) -> pd.Series:
    """Count of consecutive same-direction bars. Positive for green, negative for red."""
    direction = np.sign(close - open_).astype(np.int8)
    # When direction changes, reset counter. When same, increment.
    out = np.zeros(len(direction), dtype=np.int16)
    n = len(direction)
    if n == 0:
        return pd.Series(out, index=close.index)
    out[0] = int(direction.iloc[0])
    for i in range(1, n):
        d_now = int(direction.iloc[i])
        d_prev = int(direction.iloc[i - 1])
        if d_now == 0:
            out[i] = 0
        elif d_now == d_prev:
            out[i] = out[i - 1] + d_now
        else:
            out[i] = d_now
    return pd.Series(out.astype(np.float32), index=close.index)


def _volume_decay_slope(open_: pd.Series, close: pd.Series, volume: pd.Series, lookback: int = 10) -> pd.Series:
    """Slope of log(volume) over the last `lookback` same-direction bars.

    Negative slope means volume is FADING on extension — classic exhaustion signal.
    Positive slope means volume is BUILDING — momentum still alive.
    """
    direction = np.sign(close - open_).astype(np.int8).to_numpy()
    log_v = np.log1p(volume.to_numpy())
    n = len(log_v)
    out = np.full(n, 0.0, dtype=np.float32)

    for t in range(lookback, n):
        # Walk back collecting same-direction bars
        d_cur = direction[t]
        if d_cur == 0:
            continue
        same_dir = []
        for j in range(t, max(t - lookback * 2, -1), -1):
            if direction[j] == d_cur:
                same_dir.append(log_v[j])
            else:
                break
            if len(same_dir) >= lookback:
                break
        if len(same_dir) < 4:
            continue
        # Slope on the LAST n same-dir bars
        y = np.array(same_dir[::-1])
        x = np.arange(len(y), dtype=np.float64)
        if y.std() > EPS:
            slope = np.polyfit(x, y, 1)[0]
            out[t] = float(slope)
    return pd.Series(out, index=close.index)


def compute(df_1m: pd.DataFrame) -> pd.DataFrame:
    """Compute exhaustion features. Returns 5 columns."""
    # 5m and 15m RSI divergences
    df_5m = _resample_ohlcv(df_1m, 5)
    df_15m = _resample_ohlcv(df_1m, 15)

    rsi_5m = _rsi(df_5m["close"], 14)
    div_5m = pd.Series(
        _detect_divergence(df_5m["close"].to_numpy(), rsi_5m.to_numpy(), lookback=15),
        index=df_5m.index, dtype=np.float32,
    )

    rsi_15m = _rsi(df_15m["close"], 14)
    div_15m = pd.Series(
        _detect_divergence(df_15m["close"].to_numpy(), rsi_15m.to_numpy(), lookback=10),
        index=df_15m.index, dtype=np.float32,
    )

    # Broadcast onto 1m grid (shift forward by TF length so we read most-recent COMPLETED bar)
    div_5m_1m = div_5m.shift(freq=pd.Timedelta(minutes=5)).reindex(df_1m.index).ffill().fillna(0)
    div_15m_1m = div_15m.shift(freq=pd.Timedelta(minutes=15)).reindex(df_1m.index).ffill().fillna(0)

    # 15m BB position broadcast onto 1m
    bb_15m = _bb_position(df_15m["close"], n=20)
    bb_15m_1m = bb_15m.shift(freq=pd.Timedelta(minutes=15)).reindex(df_1m.index).ffill()

    # 1m-native features
    cons_color = _signed_consecutive_color(df_1m["open"], df_1m["close"])
    vol_decay = _volume_decay_slope(df_1m["open"], df_1m["close"], df_1m["volume"], lookback=10)

    return pd.DataFrame({
        "exhaust_rsi_div_5m": div_5m_1m.astype(np.float32),
        "exhaust_rsi_div_15m": div_15m_1m.astype(np.float32),
        "exhaust_bb_position_15m": bb_15m_1m.astype(np.float32),
        "exhaust_consecutive_color": cons_color.astype(np.float32),
        "exhaust_volume_decay": vol_decay.astype(np.float32),
    }, index=df_1m.index)


FEATURE_COLUMNS = (
    "exhaust_rsi_div_5m",
    "exhaust_rsi_div_15m",
    "exhaust_bb_position_15m",
    "exhaust_consecutive_color",
    "exhaust_volume_decay",
)
