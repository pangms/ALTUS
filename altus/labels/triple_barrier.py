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

from altus.config import LABEL_HORIZON_BARS, SL_POINTS, TP_POINTS


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


def triple_barrier_labels(
    df_1m: pd.DataFrame,
    tp_points: float = TP_POINTS,
    sl_points: float = SL_POINTS,
    horizon: int = LABEL_HORIZON_BARS,
) -> LabelOutput:
    """Vectorized triple-barrier labeler.

    Parameters
    ----------
    df_1m : DataFrame with at least 'open', 'high', 'low', UTC-indexed at 1m.
    tp_points, sl_points : barrier sizes in index points.
    horizon : H, max bars to a barrier.

    Returns
    -------
    LabelOutput where all arrays are aligned to `index` (a subset of df_1m.index
    with the last `horizon` rows removed and session-break-spanning rows filtered).
    """
    # We need at least horizon+1 bars to label a single entry.
    if len(df_1m) <= horizon:
        raise ValueError(f"need > {horizon} bars to label; got {len(df_1m)}")

    opens = df_1m["open"].to_numpy(dtype=np.float32)
    highs = df_1m["high"].to_numpy(dtype=np.float32)
    lows = df_1m["low"].to_numpy(dtype=np.float32)
    n = len(opens)

    # Number of labelable rows: we lose the last `horizon` because their forward
    # window doesn't fit in the data.
    n_labels = n - horizon
    entry = opens[:n_labels]  # shape (n_labels,)

    # Sliding windows of high and low covering bars [T, T+H-1].
    # sliding_window_view returns a view — no big memory copy here.
    from numpy.lib.stride_tricks import sliding_window_view
    high_win = sliding_window_view(highs, window_shape=horizon)  # (n-H+1, H)
    low_win = sliding_window_view(lows, window_shape=horizon)
    # Trim to n_labels rows (sliding_window_view gives n-H+1, we want n-H).
    high_win = high_win[:n_labels]
    low_win = low_win[:n_labels]

    # ----- Long-side barriers --------------------------------------------------
    long_tp_mask = high_win >= entry[:, None] + tp_points   # (n_labels, H) bool
    long_sl_mask = low_win <= entry[:, None] - sl_points

    long_tp_first = _first_true(long_tp_mask, default=horizon)
    long_sl_first = _first_true(long_sl_mask, default=horizon)

    # Conservative within-bar tie-break: SL wins ties.
    long_label = (long_tp_first < long_sl_first).astype(np.int8)
    long_stop = np.minimum(np.minimum(long_tp_first, long_sl_first), horizon - 1)

    mfe_long, mae_long = _excursion(
        high_win=high_win, low_win=low_win, entry=entry, stop_idx=long_stop, side="long"
    )

    # ----- Short-side barriers -------------------------------------------------
    short_tp_mask = low_win <= entry[:, None] - tp_points
    short_sl_mask = high_win >= entry[:, None] + sl_points

    short_tp_first = _first_true(short_tp_mask, default=horizon)
    short_sl_first = _first_true(short_sl_mask, default=horizon)
    short_label = (short_tp_first < short_sl_first).astype(np.int8)
    short_stop = np.minimum(np.minimum(short_tp_first, short_sl_first), horizon - 1)

    mfe_short, mae_short = _excursion(
        high_win=high_win, low_win=low_win, entry=entry, stop_idx=short_stop, side="short"
    )

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
