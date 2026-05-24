"""Family 11 (Phase B-4): Kernel-smoothed volume profile.

Why this matters: a volume profile shows WHERE price spent the most time/volume
over a recent window. The "Point of Control" (POC) — the highest-volume price —
acts as a magnet (price tends to revert) or a barrier (price tends to bounce).
The "Value Area" (typically 70% of recent volume) defines the recent fair-value
range. "Low Volume Nodes" (LVNs, density troughs) are easy-passage zones where
price moves quickly.

We use a kernel-smoothed (KDE-weighted) volume profile instead of a histogram
because (a) it doesn't bin-artifact at price boundaries, (b) it gives a
continuous density we can compute peaks/troughs on cleanly.

Features (6 total):
  • vp_dist_to_poc_atr      distance to Point of Control in ATR units (signed:
                            negative = POC is above current price, positive = below)
  • vp_dist_to_vah_atr      distance to Value Area High
  • vp_dist_to_val_atr      distance to Value Area Low
  • vp_in_value_area        1.0 if current price is within VA; else 0.0
  • vp_dist_to_lvn_atr      distance to nearest Low Volume Node (signed)
  • vp_density_at_current   KDE density at current price, normalized [0,1]
                            (= 1 means we're sitting at the most-traded price)

CAUSALITY: profile built from bars [t-W, t-1] only. Recomputed every N bars
and forward-filled (profile evolves slowly within a session).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from scipy.stats import gaussian_kde


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


def _value_area_from_density(
    price_grid: np.ndarray,
    density: np.ndarray,
    target_fraction: float = 0.70,
) -> tuple[float, float]:
    """Find (VAL, VAH) — the contiguous range around POC containing target_fraction of density."""
    if density.sum() <= 0:
        return float("nan"), float("nan")
    poc_idx = int(np.argmax(density))
    total = density.sum()
    target = target_fraction * total

    lo = hi = poc_idx
    accumulated = density[poc_idx]
    while accumulated < target and (lo > 0 or hi < len(density) - 1):
        # Expand the side with higher next-step density
        next_lo = density[lo - 1] if lo > 0 else -np.inf
        next_hi = density[hi + 1] if hi < len(density) - 1 else -np.inf
        if next_hi >= next_lo:
            hi += 1
            accumulated += next_hi
        else:
            lo -= 1
            accumulated += next_lo
    return float(price_grid[lo]), float(price_grid[hi])


def _profile_features_one_window(
    closes: np.ndarray,
    volumes: np.ndarray,
    current_price: float,
    atr_local: float,
) -> tuple[float, float, float, float, float, float]:
    """Compute the 6 profile-derived metrics for a single window of bars."""
    if len(closes) < 30 or atr_local <= 0 or volumes.sum() <= 0:
        return (np.nan,) * 6

    # KDE weighted by volume, bandwidth = 0.25 ATR
    weights = volumes / max(volumes.sum(), EPS)
    bandwidth = max(0.25 * atr_local, EPS)
    try:
        kde = gaussian_kde(closes, weights=weights)
        std = closes.std() if len(closes) > 1 else 1.0
        if std > 0:
            kde.set_bandwidth(bw_method=bandwidth / std)
    except Exception:
        return (np.nan,) * 6

    # Evaluate on a grid spanning ±6 ATR around current price
    lo_p = current_price - 6 * atr_local
    hi_p = current_price + 6 * atr_local
    grid = np.linspace(lo_p, hi_p, 301)
    density = kde(grid)
    if density.sum() <= 0:
        return (np.nan,) * 6

    # POC
    poc_idx = int(np.argmax(density))
    poc_price = float(grid[poc_idx])
    dist_poc_atr = (poc_price - current_price) / atr_local  # signed

    # Value Area
    val, vah = _value_area_from_density(grid, density, target_fraction=0.70)
    dist_vah_atr = (vah - current_price) / atr_local
    dist_val_atr = (current_price - val) / atr_local
    in_va = 1.0 if (val <= current_price <= vah) else 0.0

    # LVNs (density troughs) — find peaks in (max - density)
    inverted = density.max() - density
    trough_peaks, _ = find_peaks(inverted, prominence=density.max() * 0.05)
    if len(trough_peaks) > 0:
        trough_prices = grid[trough_peaks]
        nearest_lvn_dist = float(np.min(np.abs(trough_prices - current_price)) / atr_local)
        # Sign by direction
        nearest_idx = int(np.argmin(np.abs(trough_prices - current_price)))
        if trough_prices[nearest_idx] < current_price:
            nearest_lvn_dist = -nearest_lvn_dist
    else:
        nearest_lvn_dist = 6.0  # cap

    # Density at current price, normalized so peak = 1
    cur_density_idx = int(np.argmin(np.abs(grid - current_price)))
    density_norm = float(density[cur_density_idx] / max(density.max(), EPS))

    return (
        dist_poc_atr, dist_vah_atr, dist_val_atr, in_va,
        nearest_lvn_dist, density_norm,
    )


def compute(
    df_1m: pd.DataFrame,
    window_bars: int = 480,        # 8 hours of 1m — session-ish window
    decimation: int = 30,           # recompute every 30 min
) -> pd.DataFrame:
    """Compute volume-profile features. Returns 6 columns."""
    n = len(df_1m)
    cols = (
        "vp_dist_to_poc_atr",
        "vp_dist_to_vah_atr",
        "vp_dist_to_val_atr",
        "vp_in_value_area",
        "vp_dist_to_lvn_atr",
        "vp_density_at_current",
    )
    out = pd.DataFrame(
        np.full((n, len(cols)), np.nan, dtype=np.float32),
        index=df_1m.index,
        columns=list(cols),
    )

    closes_arr = df_1m["close"].to_numpy(dtype=np.float64)
    vols_arr = df_1m["volume"].to_numpy(dtype=np.float64)
    atr_series = _atr(df_1m, n=14).to_numpy(dtype=np.float64)

    for pos in range(window_bars, n, decimation):
        atr_local = atr_series[pos] if pos < len(atr_series) and np.isfinite(atr_series[pos]) else 0.0
        if atr_local <= 0:
            continue
        lo, hi = pos - window_bars, pos  # bars [pos - W, pos - 1]
        feats = _profile_features_one_window(
            closes_arr[lo:hi], vols_arr[lo:hi], closes_arr[pos], atr_local
        )
        out.iloc[pos] = feats

    return out.ffill()


FEATURE_COLUMNS = (
    "vp_dist_to_poc_atr",
    "vp_dist_to_vah_atr",
    "vp_dist_to_val_atr",
    "vp_in_value_area",
    "vp_dist_to_lvn_atr",
    "vp_density_at_current",
)
