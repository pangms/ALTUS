"""Family 9: Mahalanobis-based anomaly detection.

Why this matters: freak candles (sharp wicks, surprise news prints, glitches)
often precede stop hunts or are stop hunts themselves. Layer 1 should be able
to weight signals near these bars differently. A single scalar "how weird is
this bar" feature is much more useful than trying to encode shape directly.

Method: exponentially-weighted (EW) Mahalanobis distance.
  * Per 1m bar we compute a 4-D feature vector: (range, abs_body, log_volume,
    vol_normalized_return).
  * Maintain online EW mean μ_t and EW covariance Σ_t with half-life ≈ 7 days.
  * Distance d_t = sqrt((x_t - μ_{t-1})ᵀ Σ_{t-1}^{-1} (x_t - μ_{t-1})).
  * Causal: stats updated AFTER distance is computed, using x_{t-1}.

Why EW not rolling window: rolling 30-day windows of 1m bars = 43k points
per update. Recomputing covariance every bar is O(43k × 16) = 700k ops/bar
× 1.8M bars = >1 trillion ops. EW is O(1) per bar; same statistical effect.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


EPS = 1e-9
_DEFAULT_HALFLIFE_BARS = 60 * 24 * 7  # 7 days of 1m bars


def _bar_feature_matrix(df_1m: pd.DataFrame) -> np.ndarray:
    """4-D per-bar feature vector for the anomaly distribution."""
    o = df_1m["open"].to_numpy(dtype=np.float64)
    h = df_1m["high"].to_numpy(dtype=np.float64)
    l = df_1m["low"].to_numpy(dtype=np.float64)
    c = df_1m["close"].to_numpy(dtype=np.float64)
    v = df_1m["volume"].to_numpy(dtype=np.float64)

    bar_range = (h - l) / (c + EPS)
    abs_body = np.abs(c - o) / (c + EPS)
    log_vol = np.log1p(v)
    # Vol-normalized return: log return scaled by a rolling-ish vol proxy (use range).
    prev_c = np.concatenate([[c[0]], c[:-1]])
    log_ret = np.log((c + EPS) / (prev_c + EPS))
    vol_norm_ret = log_ret / (bar_range + EPS)

    X = np.column_stack([bar_range, abs_body, log_vol, vol_norm_ret])
    # Replace any non-finite values with 0 (shouldn't happen with clean data)
    return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)


def _ew_mahalanobis(X: np.ndarray, halflife_bars: int) -> np.ndarray:
    """Online EW Mahalanobis distance, causal.

    For each row t, distance is measured against the EW distribution computed
    from rows [0, ..., t-1]. The stats are updated AFTER the distance is read.
    """
    n, d = X.shape
    # alpha satisfies (1-alpha)^halflife = 0.5 → alpha = 1 - 2^(-1/halflife)
    alpha = 1.0 - 0.5 ** (1.0 / max(halflife_bars, 1))

    # Initialize on the first ~halflife bars without producing useful distances —
    # we'll fill them with NaN and let downstream drop the warmup region.
    mean = X[0].copy()
    # Tiny regularizer along the diagonal to ensure Sigma is invertible early on.
    cov = np.eye(d) * 1e-3

    distances = np.empty(n, dtype=np.float64)
    distances[0] = np.nan
    warmup_until = halflife_bars // 4  # ~1.75 days of warmup before trusting

    for t in range(1, n):
        x = X[t]
        if t >= warmup_until:
            delta = x - mean
            try:
                inv_cov = np.linalg.inv(cov + np.eye(d) * 1e-9)
                d2 = float(delta @ inv_cov @ delta)
                distances[t] = float(np.sqrt(max(d2, 0.0)))
            except np.linalg.LinAlgError:
                distances[t] = np.nan
        else:
            distances[t] = np.nan

        # Update stats with the current bar (for use by the NEXT bar — keeps causality)
        delta_old = x - mean
        mean = mean + alpha * delta_old
        delta_new = x - mean
        cov = (1.0 - alpha) * cov + alpha * np.outer(delta_new, delta_new)

    return distances


def compute(df_1m: pd.DataFrame, halflife_bars: int = _DEFAULT_HALFLIFE_BARS) -> pd.DataFrame:
    """Compute the anomaly Mahalanobis distance per 1m bar.

    Returns 1 column: anomaly_mahalanobis.
    """
    X = _bar_feature_matrix(df_1m)
    d = _ew_mahalanobis(X, halflife_bars=halflife_bars)
    return pd.DataFrame({"anomaly_mahalanobis": d.astype(np.float32)}, index=df_1m.index)


FEATURE_COLUMNS = ("anomaly_mahalanobis",)
