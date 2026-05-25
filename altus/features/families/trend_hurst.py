"""Family 2: Multi-timeframe trend + Hurst exponent (DFA).

Why this matters: Layer 1 sees 240 1m bars (4 hours). It has no clean view of
whether the higher timeframes (4h, 1d, 1w) are in a trend, a range, or a
reversal regime. A long setup on the 1m is much more likely to succeed if the
4h is trending up too. These features supply the HTF context.

Two complementary measures per timeframe:
  * EMA slope normalized by ATR — direction + magnitude (signed)
  * Hurst exponent via DFA — persistence quality (H>0.5 trending; H<0.5 mean-reverting)

Plus 2 alignment features that summarize trend agreement across timeframes.

Why DFA for Hurst: Detrended Fluctuation Analysis is robust to non-stationary
trends (which financial data clearly has). R/S analysis is biased on short
windows; DFA stays principled. Implementation is ~30 lines, no external dep.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


EPS = 1e-9
_HUSRT_SCALES = (4, 8, 16, 32, 64)  # box sizes for DFA — geometric


def _hurst_dfa(series: np.ndarray, scales: tuple[int, ...] = _HUSRT_SCALES) -> float:
    """Detrended Fluctuation Analysis estimator of the Hurst exponent.

    Returns H ∈ (0, 1):
      H ≈ 0.5 → random walk (no edge from direction)
      H > 0.6 → trending (autocorrelation persists)
      H < 0.4 → mean-reverting (price reverses)

    Algorithm:
      1. Integrate the series: Y_i = cumsum(x - mean(x))
      2. For each scale s, divide Y into segments of length s; fit a linear
         trend per segment; compute RMS detrended residual.
      3. F(s) ∝ s^H, so plot log F(s) vs log s and fit slope.
    """
    n = len(series)
    if n < max(scales) * 2:
        return 0.5  # not enough data; return random-walk default
    series = np.asarray(series, dtype=np.float64)
    if not np.all(np.isfinite(series)) or series.std() < EPS:
        return 0.5
    y = np.cumsum(series - series.mean())

    log_s = []
    log_F = []
    for s in scales:
        if n < s * 2:
            continue
        # Number of complete segments
        n_segs = n // s
        if n_segs < 2:
            continue
        rms = np.empty(n_segs)
        idx = np.arange(s, dtype=np.float64)
        for k in range(n_segs):
            seg = y[k * s : (k + 1) * s]
            # Fit a linear trend; compute RMS residual.
            slope, intercept = np.polyfit(idx, seg, 1)
            resid = seg - (slope * idx + intercept)
            rms[k] = np.sqrt((resid ** 2).mean())
        F_s = np.sqrt((rms ** 2).mean())
        if F_s > EPS:
            log_s.append(np.log(s))
            log_F.append(np.log(F_s))

    if len(log_s) < 2:
        return 0.5
    slope, _ = np.polyfit(log_s, log_F, 1)
    # Clamp to (0, 1) — Hurst should never exit this range; small numerical drift OK
    return float(np.clip(slope, 0.05, 0.95))


def _resample_ohlcv(df_1m: pd.DataFrame, tf_min: int) -> pd.DataFrame:
    """Aggregate 1-min OHLCV to a higher timeframe. Bars labeled at START."""
    if tf_min == 1:
        return df_1m.copy()
    rule = f"{tf_min}min"
    out = df_1m.resample(rule, label="left", closed="left").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    )
    return out.dropna(how="any")


def _atr(df: pd.DataFrame, n: int) -> pd.Series:
    """Rolling ATR with graceful warmup (min_periods=2) so short series still
    produce values; otherwise feature columns would be all-NaN for short
    histories and the dropna() at the end of the pipeline would eat everything."""
    prev_close = df["close"].shift(1)
    tr = pd.concat(
        [(df["high"] - df["low"]).abs(),
         (df["high"] - prev_close).abs(),
         (df["low"] - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    return tr.rolling(n, min_periods=2).mean()


def _trend_features_one_tf(df_1m: pd.DataFrame, tf_min: int, ema_n: int = 50, hurst_window: int = 120) -> pd.DataFrame:
    """Compute slope (EMA-based, ATR-normalized) and Hurst on a single higher TF.

    Returns features on the 1m grid via shift-then-ffill (causal, no look-ahead).
    """
    htf = _resample_ohlcv(df_1m, tf_min)
    if len(htf) < 2:
        # Not enough HTF bars to compute anything meaningful; return all-default block.
        out = pd.DataFrame(
            {f"trend_{tf_min}m_slope_raw": 0.0, f"trend_{tf_min}m_hurst_raw": 0.5},
            index=df_1m.index,
        )
        return out
    ema = htf["close"].ewm(span=ema_n, adjust=False).mean()
    atr = _atr(htf, ema_n).replace(0, np.nan)
    # Slope per bar: (EMA_t - EMA_{t-1}) / ATR_t; default to 0 during warmup.
    slope = ((ema - ema.shift(1)) / atr).fillna(0.0)

    # Hurst on rolling window of log returns at this TF.
    # Use 0.5 (random walk) as the default during warmup so dropna doesn't eat
    # the whole feature matrix when the test period is shorter than the warmup.
    log_ret = np.log(htf["close"] / htf["close"].shift(1))
    hurst_vals = np.full(len(htf), 0.5, dtype=np.float32)
    arr = log_ret.to_numpy()
    for t in range(hurst_window, len(htf)):
        w = arr[t - hurst_window : t]
        if np.all(np.isfinite(w)):
            hurst_vals[t] = _hurst_dfa(w)
    hurst_series = pd.Series(hurst_vals, index=htf.index, name=f"trend_{tf_min}m_hurst_raw")

    feats = pd.DataFrame({
        f"trend_{tf_min}m_slope_raw": slope.astype(np.float32),
        f"trend_{tf_min}m_hurst_raw": hurst_series,
    })

    # Shift forward by tf_min so each value becomes "available" at the close of
    # its source bar. Then broadcast onto the 1m grid via union+ffill (NOT direct
    # reindex+ffill) — because for weekly bars, the shifted timestamps land on
    # Sundays when MNQ is closed and don't exist in df_1m.index. The union pattern
    # keeps the shifted values alive so ffill can propagate them forward.
    shifted = feats.shift(freq=pd.Timedelta(minutes=tf_min))
    union_idx = shifted.index.union(df_1m.index).sort_values()
    return shifted.reindex(union_idx).ffill().reindex(df_1m.index)


NEEDS_RAW_1M = True  # uses _resample_ohlcv → must see clean non-overlapping 1m bars


def compute(df_primary: pd.DataFrame, df_1m: pd.DataFrame | None = None) -> pd.DataFrame:
    """Compute multi-TF trend features aligned to the primary's index.

    Uses raw 1m for HTF resampling (rolling-primary bars would mis-aggregate).
    Returns 8 columns:
      trend_4h_slope, trend_4h_hurst, trend_1d_slope, trend_1d_hurst,
      trend_1w_slope, trend_1w_hurst, trend_alignment, trend_strength
    """
    if df_1m is None:
        df_1m = df_primary  # back-compat / PRIMARY_WINDOW_MIN=1 path
    tfs = {"4h": 240, "1d": 1440, "1w": 10_080}
    blocks = []
    label_to_cols = {}
    for label, tf_min in tfs.items():
        block = _trend_features_one_tf(df_1m, tf_min)
        # Rename to friendly labels
        block = block.rename(columns={
            f"trend_{tf_min}m_slope_raw": f"trend_{label}_slope",
            f"trend_{tf_min}m_hurst_raw": f"trend_{label}_hurst",
        })
        blocks.append(block)
        label_to_cols[label] = (f"trend_{label}_slope", f"trend_{label}_hurst")

    feats = pd.concat(blocks, axis=1)

    # Alignment: mean of sign(slope_TF) across the three TFs ∈ {-1, -0.33, 0.33, 1}
    sign_4h = np.sign(feats["trend_4h_slope"])
    sign_1d = np.sign(feats["trend_1d_slope"])
    sign_1w = np.sign(feats["trend_1w_slope"])
    trend_alignment = (sign_4h + sign_1d + sign_1w) / 3.0

    # Strength: weighted average of |slope| by Hurst (so meaningful trends count more)
    def _w(s, h):
        return s.abs() * h.fillna(0.5)
    trend_strength = (
        _w(feats["trend_4h_slope"], feats["trend_4h_hurst"])
        + _w(feats["trend_1d_slope"], feats["trend_1d_hurst"])
        + _w(feats["trend_1w_slope"], feats["trend_1w_hurst"])
    ) / 3.0

    feats["trend_alignment"] = trend_alignment.astype(np.float32)
    feats["trend_strength"] = trend_strength.astype(np.float32)
    return feats


FEATURE_COLUMNS = (
    "trend_4h_slope", "trend_4h_hurst",
    "trend_1d_slope", "trend_1d_hurst",
    "trend_1w_slope", "trend_1w_hurst",
    "trend_alignment", "trend_strength",
)
