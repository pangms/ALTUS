"""Triple-barrier labeling with MFE/MAE regression targets.

For each bar T, we simulate hypothetical entries at the OPEN of bar T and ask:
within the next H bars, did price hit the +TP barrier before the -SL barrier?
We do this for both long and short sides, and also record the max favorable
and max adverse excursions up to the resolution bar.

Critical conventions:
  * Entry price = open[T]. The decision to enter at T is made BEFORE bar T using
    features that don't include bar T — see altus.features.pipeline docstring.
  * Barrier window = bars [T, T+1, ..., T+H-1] INCLUSIVE.
  * Worst-case-within-bar: if both barriers are touched within the same bar
    (high >= entry+TP AND low <= entry-SL), we conservatively count it as SL.
    This avoids over-optimistic labels and is standard quant practice.
  * MFE/MAE are measured up to the resolution bar (stop), not the full H window.
    If SL hits at bar k, we don't get to "see" the favorable excursion that
    might have happened at bar k+1 — we were already stopped out.
  * A label is INVALID if the H-bar forward window spans a session break
    (i.e., the 1m timestamps are not exactly H-1 minutes apart). Spanning a
    break would mean we couldn't actually monitor & exit at the barrier price.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import numpy as np
import pandas as pd

from altus.config import (
    LABEL_HORIZON_BARS,
    LABEL_SCALE_MODE,
    LABEL_VOL_SCALE_ATR_BARS,
    LABEL_VOL_SCALE_K,
    SL_POINTS,
    TP_POINTS,
)


def _causal_atr_per_bar(df_1m: pd.DataFrame, n_bars: int) -> np.ndarray:
    """Per-bar ATR computed strictly from PAST bars (causal — shifted by 1).

    True Range = max(H-L, |H-prev_close|, |L-prev_close|). ATR is the rolling
    mean of TR over `n_bars`. The .shift(1) at the end ensures ATR at row T
    uses only bars strictly before T — so a label at T using `k * ATR[T]` for
    barriers is leak-free.

    Returns float32 array of length len(df_1m). Warmup rows (< n_bars history)
    are forward-filled from the first valid ATR, then nan-filled with the
    median to avoid 0/NaN propagation into barrier calculations.
    """
    h = df_1m["high"].astype(np.float32)
    l = df_1m["low"].astype(np.float32)
    c = df_1m["close"].astype(np.float32)
    prev_c = c.shift(1)
    tr = pd.concat([
        (h - l).abs(),
        (h - prev_c).abs(),
        (l - prev_c).abs(),
    ], axis=1).max(axis=1)
    atr = tr.rolling(n_bars, min_periods=max(2, n_bars // 4)).mean().shift(1)
    # Fall back to median for warmup rows so we don't end up with zero barriers.
    atr_med = float(atr.median())
    if not np.isfinite(atr_med) or atr_med <= 0:
        atr_med = 1.0
    return atr.fillna(atr_med).clip(lower=0.5).to_numpy(dtype=np.float32)


class LabelOutput(NamedTuple):
    """Aligned to the input index — rows where labels are invalid are filtered out."""
    index: pd.DatetimeIndex
    long_tp: np.ndarray   # int8, {0, 1}
    short_tp: np.ndarray  # int8, {0, 1}
    mfe_long: np.ndarray  # float32, points
    mae_long: np.ndarray  # float32, points (positive = bigger drawdown)
    mfe_short: np.ndarray
    mae_short: np.ndarray
    time_to_long_tp: np.ndarray   # int16, bar index of TP hit or H if no hit
    time_to_short_tp: np.ndarray
    entry_price: np.ndarray       # float32, the open we'd have entered at
    # Phase H: inflection auxiliary target (Q26 — inflection vs continuation).
    # = 1 if the trade resolved AGAINST the recent direction (reversal),
    # else 0. Used as an auxiliary head on L1 — regularizes the shared encoder
    # and gives Layer 2 a "is this a turning point?" signal.
    inflection_label: np.ndarray  # int8, {0, 1}
    # Per-bar barrier sizes (post-audit 2026-05-25). Under vol-scaled mode
    # these vary by bar (TP = SL = k × ATR_local); under fixed mode they're
    # the constant TP_POINTS / SL_POINTS broadcast. Downstream sims read these
    # for honest PnL when barriers vary.
    tp_points: np.ndarray         # float32, points
    sl_points: np.ndarray         # float32, points
    # Predictive framework labels (2026-05-25 — FRAMEWORK.md C-tier).
    # Signed forward returns at fixed horizons in ATR units. Causal: computed
    # from bars strictly after T over the H-bar window. These force the encoder
    # to project magnitude × direction, not just classify barrier outcomes.
    return_H15: np.ndarray        # float32, signed return at T+15 in ATR units
    return_H60: np.ndarray        # float32, signed return at T+horizon in ATR units
    # 3-class path-shape label (continuation / revert / chop). Categorizes
    # the realized H-bar window's character. Answers Q26/C3.
    #   0 = continuation: |terminal return| > 0.5 ATR (clean directional move)
    #   1 = revert:       realized range > 0.7 ATR AND |terminal return| < 0.3 ATR
    #   2 = chop:         realized range < 0.5 ATR (no meaningful move)
    path_shape_class: np.ndarray  # int8, {0, 1, 2}
    # Binary: did price extend by ≥ 1 ATR in EITHER direction within H bars?
    # Generic level-clearance proxy used by C4 head. The setup-aware
    # interpretation happens at feature time (setups know WHICH level matters).
    clears_1atr: np.ndarray       # int8, {0, 1}


def triple_barrier_labels(
    df_1m: pd.DataFrame,
    tp_points: float | None = None,
    sl_points: float | None = None,
    horizon: int = LABEL_HORIZON_BARS,
    scale_mode: str | None = None,
) -> LabelOutput:
    """Vectorized triple-barrier labeler with optional vol-scaled barriers.

    Parameters
    ----------
    df_1m : DataFrame with at least 'open', 'high', 'low', UTC-indexed at 1m.
    tp_points, sl_points : barrier sizes in index points. Only honored when
        scale_mode='fixed'. When None, falls back to config.TP_POINTS/SL_POINTS.
    horizon : H, max bars to a barrier.
    scale_mode : 'fixed' or 'vol_scaled'. Defaults to config.LABEL_SCALE_MODE
        (post-audit default 'vol_scaled' — TP=SL=k×ATR per bar). Vol-scaled
        lifts base rate from ~0.27 (with fixed 30pt) to ~0.50 by removing
        the "is this regime volatile enough" component from the label.

    Returns
    -------
    LabelOutput where all arrays are aligned to `index` (a subset of df_1m.index
    with the last `horizon` rows removed and session-break-spanning rows filtered).
    """
    if len(df_1m) <= horizon:
        raise ValueError(f"need > {horizon} bars to label; got {len(df_1m)}")

    mode = scale_mode if scale_mode is not None else LABEL_SCALE_MODE
    if mode not in ("fixed", "vol_scaled"):
        raise ValueError(f"unknown scale_mode: {mode!r}")

    opens = df_1m["open"].to_numpy(dtype=np.float32)
    highs = df_1m["high"].to_numpy(dtype=np.float32)
    lows = df_1m["low"].to_numpy(dtype=np.float32)
    n = len(opens)
    n_labels = n - horizon
    entry = opens[:n_labels]

    # Per-bar barrier sizes — same shape as `entry`. Causal: ATR computed
    # strictly from past bars so labels don't leak future vol.
    if mode == "vol_scaled":
        atr = _causal_atr_per_bar(df_1m, LABEL_VOL_SCALE_ATR_BARS)[:n_labels]
        tp_pts_arr = (LABEL_VOL_SCALE_K * atr).astype(np.float32)
        sl_pts_arr = tp_pts_arr.copy()  # 1:1 RR by construction
    else:
        tp_v = TP_POINTS if tp_points is None else float(tp_points)
        sl_v = SL_POINTS if sl_points is None else float(sl_points)
        tp_pts_arr = np.full(n_labels, tp_v, dtype=np.float32)
        sl_pts_arr = np.full(n_labels, sl_v, dtype=np.float32)

    # Sliding windows of high and low covering bars [T, T+H-1].
    from numpy.lib.stride_tricks import sliding_window_view
    high_win = sliding_window_view(highs, window_shape=horizon)[:n_labels]
    low_win = sliding_window_view(lows, window_shape=horizon)[:n_labels]

    # Per-bar barriers via (n_labels,)-broadcast — works for both modes.
    long_tp_thr = (entry + tp_pts_arr)[:, None]
    long_sl_thr = (entry - sl_pts_arr)[:, None]
    short_tp_thr = (entry - sl_pts_arr)[:, None]
    short_sl_thr = (entry + sl_pts_arr)[:, None]

    # ----- Long-side barriers --------------------------------------------------
    long_tp_mask = high_win >= long_tp_thr
    long_sl_mask = low_win <= long_sl_thr

    long_tp_first = _first_true(long_tp_mask, default=horizon)
    long_sl_first = _first_true(long_sl_mask, default=horizon)

    long_label = (long_tp_first < long_sl_first).astype(np.int8)
    long_stop = np.minimum(np.minimum(long_tp_first, long_sl_first), horizon - 1)

    mfe_long, mae_long = _excursion(
        high_win=high_win, low_win=low_win, entry=entry, stop_idx=long_stop, side="long"
    )

    # ----- Short-side barriers -------------------------------------------------
    short_tp_mask = low_win <= short_tp_thr
    short_sl_mask = high_win >= short_sl_thr

    short_tp_first = _first_true(short_tp_mask, default=horizon)
    short_sl_first = _first_true(short_sl_mask, default=horizon)
    short_label = (short_tp_first < short_sl_first).astype(np.int8)
    short_stop = np.minimum(np.minimum(short_tp_first, short_sl_first), horizon - 1)

    mfe_short, mae_short = _excursion(
        high_win=high_win, low_win=low_win, entry=entry, stop_idx=short_stop, side="short"
    )

    # ----- Predictive framework labels (2026-05-25 — FRAMEWORK.md C-tier) ------
    # Per-bar ATR for normalizing forward returns. Same window used for
    # vol-scaled barriers — reuse to keep things consistent.
    if mode == "vol_scaled":
        atr_per_bar = atr[:n_labels].astype(np.float32)
    else:
        atr_per_bar = _causal_atr_per_bar(df_1m, LABEL_VOL_SCALE_ATR_BARS)[:n_labels].astype(np.float32)
    atr_safe = np.maximum(atr_per_bar, 1e-6)

    # Forward returns at H+15 and at terminal (H+horizon).
    closes = df_1m["close"].to_numpy(dtype=np.float32)
    h15 = min(15, horizon)
    close_at_T = closes[:n_labels]
    close_at_T_plus_15 = closes[h15 : h15 + n_labels]
    close_at_T_plus_H = closes[horizon : horizon + n_labels]
    # Defensive guard against length mismatch on tail bars
    n_h15 = min(len(close_at_T_plus_15), n_labels)
    n_h = min(len(close_at_T_plus_H), n_labels)
    return_H15 = np.zeros(n_labels, dtype=np.float32)
    return_H60 = np.zeros(n_labels, dtype=np.float32)
    return_H15[:n_h15] = (close_at_T_plus_15[:n_h15] - close_at_T[:n_h15]) / atr_safe[:n_h15]
    return_H60[:n_h] = (close_at_T_plus_H[:n_h] - close_at_T[:n_h]) / atr_safe[:n_h]
    # Clip to a sane range — prevents extreme outliers from dominating regression.
    return_H15 = np.clip(return_H15, -10.0, 10.0)
    return_H60 = np.clip(return_H60, -10.0, 10.0)

    # Path-shape 3-class label: continuation / revert / chop.
    # Computed from the realized H-bar window's geometry.
    max_high_in_window = high_win.max(axis=1)
    min_low_in_window = low_win.min(axis=1)
    # Excursion magnitudes in ATR units relative to entry
    max_up_atr = (max_high_in_window - entry) / atr_safe
    max_down_atr = (entry - min_low_in_window) / atr_safe
    realized_range_atr = max_up_atr + max_down_atr
    # Terminal return magnitude (close at horizon, not barrier-resolution)
    abs_terminal = np.abs(return_H60)

    # Continuation: clean directional move — terminal magnitude >= 0.5 ATR
    # AND the move-against-terminal-direction is bounded
    sign_terminal = np.sign(return_H60)
    # Drawdown against the terminal direction
    drawdown_against = np.where(sign_terminal > 0, max_down_atr, max_up_atr)
    drawdown_against = np.where(sign_terminal == 0, np.maximum(max_up_atr, max_down_atr), drawdown_against)

    is_continuation = (abs_terminal >= 0.5) & (drawdown_against < 0.7)
    is_revert = (realized_range_atr >= 0.7) & (abs_terminal < 0.3)
    is_chop = (realized_range_atr < 0.5) & (~is_continuation) & (~is_revert)
    # Default class: continuation if signed move, else chop (cleaner than mixed)
    path_shape = np.where(is_revert, 1,
                  np.where(is_chop, 2,
                   np.where(is_continuation, 0,
                    # Fallback: closest to either continuation or chop based on magnitude
                    np.where(abs_terminal >= 0.3, 0, 2)))).astype(np.int8)

    # Generic level-clearance: did price excursion exceed 1.5 ATR in either
    # direction? Threshold chosen so the label has roughly 50/50 base rate
    # under vol-scaled barriers — informative for the binary classification head.
    # Smoke-test result: ~100% at 1.0 ATR (too easy), 1.5 ATR gives meaningful
    # split.
    clears_1atr = ((max_up_atr >= 1.5) | (max_down_atr >= 1.5)).astype(np.int8)

    # ----- Inflection auxiliary label (Phase H, Q26) ---------------------------
    # Recent direction = sign of open[T] - open[T-K] over K=10 bars.
    # Inflection = 1 if recent direction was UP but only short_tp triggered (or
    # vice versa) — i.e., the market resolved AGAINST the recent direction.
    # Continuation (label 0) when direction matches resolution OR timeout.
    INFLECTION_LOOKBACK = 10
    INFLECTION_MIN_MOVE_PTS = 2.0  # require non-trivial recent direction to label inflection
    recent_change = np.zeros(n_labels, dtype=np.float32)
    if n_labels > INFLECTION_LOOKBACK:
        recent_change[INFLECTION_LOOKBACK:] = (
            opens[INFLECTION_LOOKBACK:n_labels] - opens[:n_labels - INFLECTION_LOOKBACK]
        )
    recent_up = recent_change > INFLECTION_MIN_MOVE_PTS
    recent_down = recent_change < -INFLECTION_MIN_MOVE_PTS
    short_won = short_label == 1
    long_won = long_label == 1
    inflection_label = ((recent_up & short_won) | (recent_down & long_won)).astype(np.int8)

    # ----- Session-break filter ------------------------------------------------
    # A label is valid iff the H-bar window starting at T spans exactly
    # H-1 minutes of wall-clock time. Otherwise the window crosses a Globex
    # maintenance break and we couldn't actually trade through it.
    ts = df_1m.index
    dt_to_end = (ts[horizon - 1 : n_labels + horizon - 1] - ts[:n_labels]).total_seconds()
    valid = dt_to_end.to_numpy() == (horizon - 1) * 60

    keep = np.where(valid)[0]
    return LabelOutput(
        index=ts[:n_labels][keep],
        long_tp=long_label[keep],
        short_tp=short_label[keep],
        mfe_long=mfe_long[keep].astype(np.float32),
        mae_long=mae_long[keep].astype(np.float32),
        mfe_short=mfe_short[keep].astype(np.float32),
        mae_short=mae_short[keep].astype(np.float32),
        time_to_long_tp=long_tp_first[keep].astype(np.int16),
        time_to_short_tp=short_tp_first[keep].astype(np.int16),
        entry_price=entry[keep].astype(np.float32),
        inflection_label=inflection_label[keep],
        tp_points=tp_pts_arr[keep],
        sl_points=sl_pts_arr[keep],
        return_H15=return_H15[keep],
        return_H60=return_H60[keep],
        path_shape_class=path_shape[keep],
        clears_1atr=clears_1atr[keep],
    )


def filter_labels_to_index(labels: LabelOutput, target_index: pd.DatetimeIndex) -> LabelOutput:
    """Keep only labels whose timestamp also exists in target_index.

    Use this to align labels with the feature matrix after the feature pipeline
    drops warmup rows (or after any other filtering step).
    """
    mask = labels.index.isin(target_index)
    idx = np.where(mask)[0]
    return LabelOutput(
        index=labels.index[idx],
        long_tp=labels.long_tp[idx],
        short_tp=labels.short_tp[idx],
        mfe_long=labels.mfe_long[idx],
        mae_long=labels.mae_long[idx],
        mfe_short=labels.mfe_short[idx],
        mae_short=labels.mae_short[idx],
        time_to_long_tp=labels.time_to_long_tp[idx],
        time_to_short_tp=labels.time_to_short_tp[idx],
        entry_price=labels.entry_price[idx],
        inflection_label=labels.inflection_label[idx],
        tp_points=labels.tp_points[idx],
        sl_points=labels.sl_points[idx],
        return_H15=labels.return_H15[idx],
        return_H60=labels.return_H60[idx],
        path_shape_class=labels.path_shape_class[idx],
        clears_1atr=labels.clears_1atr[idx],
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _first_true(mask: np.ndarray, default: int) -> np.ndarray:
    """First True column index per row, or `default` if no True in that row."""
    has_any = mask.any(axis=1)
    idx = mask.argmax(axis=1)  # returns 0 when all-False — we need to override that
    return np.where(has_any, idx, default).astype(np.int32)


def _excursion(
    high_win: np.ndarray,
    low_win: np.ndarray,
    entry: np.ndarray,
    stop_idx: np.ndarray,
    side: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Max favorable / max adverse excursion up to and including the stop bar.

    For longs: MFE = max(high) - entry, MAE = entry - min(low).
    For shorts: MFE = entry - min(low), MAE = max(high) - entry.
    Both MFE and MAE are returned as non-negative magnitudes in points.
    """
    H = high_win.shape[1]
    bar_idx = np.arange(H, dtype=np.int32)
    keep_mask = bar_idx[None, :] <= stop_idx[:, None]  # (N, H) bool

    # Mask out bars after stop with values that won't affect min/max
    highs_kept = np.where(keep_mask, high_win, -np.inf)
    lows_kept = np.where(keep_mask, low_win, np.inf)

    max_high = highs_kept.max(axis=1)
    min_low = lows_kept.min(axis=1)

    if side == "long":
        mfe = max_high - entry
        mae = entry - min_low
    else:  # short
        mfe = entry - min_low
        mae = max_high - entry

    # Clip floor at 0 to avoid tiny negatives from float imprecision
    mfe = np.maximum(mfe, 0.0)
    mae = np.maximum(mae, 0.0)
    return mfe.astype(np.float32), mae.astype(np.float32)
