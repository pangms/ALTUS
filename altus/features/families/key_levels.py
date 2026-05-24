"""Family 8 (Phase B-1): Key support/resistance levels via KDE on swing points.

Why this matters: every trader watches the same key levels — prior swing highs/
lows, prior day high/low, weekly extremes. Price respects these levels far more
than chance. But "level strength" is not binary — it's a continuous quantity
that depends on how many touches, how recent, how much volume traded there,
and how prominent the swing was.

Method (advanced quant, sushi-chef compliant — earns its place over simple
'list of levels'):
  1. Detect swing highs and swing lows in recent history using peak detection
     with prominence threshold (scipy.signal.find_peaks).
  2. Weight each swing by: recency × log(volume) × prominence.
  3. Fit a 1-D Kernel Density Estimate (KDE) over the price axis using these
     weighted swing prices. Bandwidth = adaptive to recent ATR.
  4. The KDE gives a smooth "level density" curve over price. Tall peaks =
     strong levels. Distance to peaks above and below current price = features.

Features (8 total):
  - dist_above_lvl1_atr   distance (in ATR units) to the nearest strong
                          level ABOVE current price
  - dist_above_lvl1_pts   same, in raw points
  - dist_below_lvl1_atr   distance to nearest strong level BELOW current price
  - dist_below_lvl1_pts   same, in raw points
  - lvl_strength_above    KDE density value at the level above (normalized)
  - lvl_strength_below    KDE density at the level below
  - in_lvl_zone           1.0 if current price is within 0.25 ATR of any
                          strong level (above or below); 0.0 otherwise
  - n_strong_levels       count of distinct strong levels in recent window

CAUSALITY: at bar T we only see swings up through bar T-1. Recomputed every
N bars (decimation) and forward-filled — levels evolve slowly so per-bar
recomputation would waste compute without adding signal.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from scipy.stats import gaussian_kde


EPS = 1e-9


def _atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    """Rolling ATR (graceful warmup). Same convention as trend_hurst._atr."""
    prev_close = df["close"].shift(1)
    tr = pd.concat(
        [(df["high"] - df["low"]).abs(),
         (df["high"] - prev_close).abs(),
         (df["low"] - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    return tr.rolling(n, min_periods=2).mean()


def _detect_swings(
    high: np.ndarray,
    low: np.ndarray,
    volume: np.ndarray,
    bar_indices: np.ndarray,
    atr_local: float,
    lookback_bars: int = 2880,  # 2 days of 1m bars
    min_prominence_atr: float = 1.0,
    min_distance_bars: int = 30,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Detect prominent swing highs and lows in a window of OHLC.

    Returns: (swing_prices, swing_weights, swing_is_high, swing_age_bars)
      all aligned arrays; swing_weights combines recency × log(volume) × prominence.
    """
    if len(high) < 2 * min_distance_bars or atr_local <= 0:
        return np.array([]), np.array([]), np.array([]), np.array([])

    prom_threshold = max(min_prominence_atr * atr_local, EPS)

    # Swing highs
    high_peaks, props_h = find_peaks(high, prominence=prom_threshold, distance=min_distance_bars)
    # Swing lows: detect peaks in -low
    low_peaks, props_l = find_peaks(-low, prominence=prom_threshold, distance=min_distance_bars)

    all_indices = np.concatenate([high_peaks, low_peaks])
    if all_indices.size == 0:
        return np.array([]), np.array([]), np.array([]), np.array([])

    all_prices = np.concatenate([high[high_peaks], low[low_peaks]])
    all_proms = np.concatenate([props_h["prominences"], props_l["prominences"]])
    is_high = np.concatenate([np.ones(len(high_peaks)), np.zeros(len(low_peaks))])

    # Weights: recency (newer = higher) × log(volume+1) × prominence
    n = len(high)
    age = (n - 1) - all_indices  # bars ago; 0 = most recent
    recency = np.exp(-age / lookback_bars).astype(np.float64)
    vol_at_swing = volume[all_indices].astype(np.float64)
    log_vol = np.log1p(vol_at_swing)
    weights = recency * log_vol * (all_proms / max(atr_local, EPS))
    # Normalize so they sum to a sane scale for KDE
    if weights.sum() > 0:
        weights = weights / weights.sum() * len(weights)
    return all_prices, weights, is_high, age


def _kde_levels(
    swing_prices: np.ndarray,
    swing_weights: np.ndarray,
    current_price: float,
    atr_local: float,
    bandwidth_atr: float = 0.5,
) -> tuple[float | None, float | None, float, float, int]:
    """Fit a weighted KDE over swing prices, find peaks of density above and below current price.

    Returns: (level_above, level_below, strength_above, strength_below, n_strong_levels)
    Levels are price values; strengths are KDE density at the peak; n_strong is
    the count of significant density peaks across the whole range.
    """
    if len(swing_prices) < 3 or atr_local <= 0:
        return None, None, 0.0, 0.0, 0

    bandwidth = bandwidth_atr * atr_local
    if bandwidth <= 0:
        return None, None, 0.0, 0.0, 0

    # KDE in price-axis units; bandwidth in those same units. Use a fine grid
    # spanning ±6 ATR around current price.
    price_grid = np.linspace(
        current_price - 6 * atr_local,
        current_price + 6 * atr_local,
        301,
    )
    try:
        # gaussian_kde wants weights to sum to N (it normalizes internally); we
        # rescale so the kde "weight per swing" is meaningful.
        kde = gaussian_kde(swing_prices, weights=swing_weights / swing_weights.sum())
        # Force its bandwidth via scotts/scotts_factor override
        # The factor h satisfies sigma_kernel = h * data_std. We want sigma=bandwidth.
        std = swing_prices.std() if len(swing_prices) > 1 else 1.0
        if std > 0:
            kde.set_bandwidth(bw_method=bandwidth / std)
        density = kde(price_grid)
    except Exception:
        return None, None, 0.0, 0.0, 0

    # Find density peaks
    peaks, props = find_peaks(density, prominence=density.max() * 0.10)
    if len(peaks) == 0:
        return None, None, 0.0, 0.0, 0
    peak_prices = price_grid[peaks]
    peak_densities = density[peaks]

    above_mask = peak_prices > current_price
    below_mask = peak_prices < current_price

    # Pick the strongest peak above and below (by density)
    level_above = None
    strength_above = 0.0
    if above_mask.any():
        idx = np.argmax(peak_densities * above_mask.astype(np.float64))
        level_above = float(peak_prices[idx])
        strength_above = float(peak_densities[idx])

    level_below = None
    strength_below = 0.0
    if below_mask.any():
        idx = np.argmax(peak_densities * below_mask.astype(np.float64))
        level_below = float(peak_prices[idx])
        strength_below = float(peak_densities[idx])

    return level_above, level_below, strength_above, strength_below, int(len(peaks))


def compute(
    df_1m: pd.DataFrame,
    lookback_bars: int = 2880,     # 2 days of 1m for swing detection window
    decimation: int = 30,           # recompute every 30 min, ffill between
    min_prominence_atr: float = 1.0,
    min_distance_bars: int = 30,
    bandwidth_atr: float = 0.5,
) -> pd.DataFrame:
    """Compute KDE-based key-level features for each 1m bar.

    Decimated to once per 30 bars and forward-filled — levels evolve slowly so
    per-bar recomputation would waste compute without adding signal.
    Returns 8 columns aligned to df_1m.index.
    """
    n = len(df_1m)
    cols = (
        "kl_dist_above_atr", "kl_dist_above_pts",
        "kl_dist_below_atr", "kl_dist_below_pts",
        "kl_strength_above", "kl_strength_below",
        "kl_in_zone", "kl_n_strong",
    )
    out = pd.DataFrame(
        np.full((n, len(cols)), np.nan, dtype=np.float32),
        index=df_1m.index,
        columns=list(cols),
    )

    high_arr = df_1m["high"].to_numpy(dtype=np.float64)
    low_arr = df_1m["low"].to_numpy(dtype=np.float64)
    close_arr = df_1m["close"].to_numpy(dtype=np.float64)
    vol_arr = df_1m["volume"].to_numpy(dtype=np.float64)
    atr_series = _atr(df_1m, n=14).to_numpy(dtype=np.float64)
    bar_indices = np.arange(n, dtype=np.int64)

    # Compute at every `decimation`-th bar starting after warmup
    eval_positions = list(range(lookback_bars, n, decimation))
    for pos in eval_positions:
        atr_local = atr_series[pos] if pos < len(atr_series) and np.isfinite(atr_series[pos]) else 0.0
        if atr_local <= 0:
            continue
        # Causal window: bars [pos - lookback, pos - 1] only
        lo, hi = pos - lookback_bars, pos
        swings_p, swings_w, swings_h, swings_age = _detect_swings(
            high_arr[lo:hi], low_arr[lo:hi], vol_arr[lo:hi], bar_indices[lo:hi],
            atr_local=atr_local,
            lookback_bars=lookback_bars,
            min_prominence_atr=min_prominence_atr,
            min_distance_bars=min_distance_bars,
        )
        current_price = close_arr[pos]
        level_above, level_below, str_above, str_below, n_strong = _kde_levels(
            swings_p, swings_w, current_price, atr_local, bandwidth_atr,
        )

        # Distance features (in ATR and points)
        dist_above_pts = (level_above - current_price) if level_above is not None else 6.0 * atr_local
        dist_below_pts = (current_price - level_below) if level_below is not None else 6.0 * atr_local
        in_zone = 1.0 if (
            (level_above is not None and (level_above - current_price) < 0.25 * atr_local)
            or (level_below is not None and (current_price - level_below) < 0.25 * atr_local)
        ) else 0.0

        out.iloc[pos] = [
            dist_above_pts / atr_local,
            dist_above_pts,
            dist_below_pts / atr_local,
            dist_below_pts,
            str_above,
            str_below,
            in_zone,
            float(n_strong),
        ]

    # Forward-fill between recomputation points
    out = out.ffill()
    return out


FEATURE_COLUMNS = (
    "kl_dist_above_atr",
    "kl_dist_above_pts",
    "kl_dist_below_atr",
    "kl_dist_below_pts",
    "kl_strength_above",
    "kl_strength_below",
    "kl_in_zone",
    "kl_n_strong",
)
