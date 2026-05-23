"""Purged walk-forward splits with embargo.

Why purging matters: each label at row T uses information from forward bars
[T, T+H-1]. If a training row's label window overlaps with a validation row,
the model has trivially seen the validation label's answer key. Embargo of at
least H bars between train end and validation start eliminates this leakage.

Why walk-forward (not k-fold): in time-series, the past predicts the future,
not vice-versa. Random k-fold trains on future to predict past, which is both
unrealistic and leakage-prone. Walk-forward respects time.

Layout (expanding window):

    [<------ fold 0 train ------><emb><- val ->]
    [<-------- fold 1 train --------><emb><- val ->]
    [<---------- fold 2 train ----------><emb><- val ->]
    ...
    [<--------------- dev set ---------------><emb><----- OOS lockbox ----->]
                                              ^ never touched until acceptance
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from altus.config import EMBARGO_BARS, N_WALK_FORWARD_FOLDS, OOS_LOCKBOX_MONTHS


@dataclass
class SplitIndices:
    """Integer positions into the labeled-sample array."""
    train_idx: np.ndarray
    val_idx: np.ndarray
    fold: int


@dataclass
class SplitsResult:
    folds: list[SplitIndices]
    oos_idx: np.ndarray
    embargo_bars: int


def purged_walk_forward(
    timestamps: pd.DatetimeIndex,
    n_folds: int = N_WALK_FORWARD_FOLDS,
    embargo_bars: int = EMBARGO_BARS,
    oos_months: int = OOS_LOCKBOX_MONTHS,
) -> SplitsResult:
    """Build N walk-forward folds + an OOS lockbox.

    Parameters
    ----------
    timestamps : the UTC-indexed timestamps of the labeled samples (post-features,
                 post-label-filter). Splits are computed on integer positions.
    n_folds    : number of walk-forward folds inside the dev set.
    embargo_bars : minimum gap between train end and val start. Must be >= H.
    oos_months : size of the OOS lockbox at the tail.

    Returns
    -------
    SplitsResult with `folds` (length n_folds) and `oos_idx` (sealed test).
    """
    if not isinstance(timestamps, pd.DatetimeIndex):
        timestamps = pd.DatetimeIndex(timestamps)

    n = len(timestamps)
    if n < 1000:
        raise ValueError(f"need at least 1000 labeled samples; got {n}")

    # Slice off the OOS lockbox by timestamp (more honest than by row count).
    if oos_months > 0:
        cutoff = timestamps.max() - pd.DateOffset(months=oos_months)
        oos_start = int(np.searchsorted(timestamps.values, np.datetime64(cutoff.tz_localize(None))))
    else:
        oos_start = n  # no OOS — full series is dev
    dev_end = oos_start
    oos_idx = np.arange(oos_start, n, dtype=np.int64)

    if dev_end < 2 * n_folds * (embargo_bars + 1):
        raise ValueError(
            f"dev set ({dev_end} rows) too small for {n_folds} folds with embargo {embargo_bars}"
        )

    # Divide the dev set into (n_folds + 1) chunks. Use chunks 0..k-1 as train
    # for fold k, and chunk k as validation. Skip chunk 0 as a val candidate
    # because there'd be no training data.
    chunk_size = dev_end // (n_folds + 1)
    folds: list[SplitIndices] = []
    for k in range(n_folds):
        val_start = (k + 1) * chunk_size
        val_end = min((k + 2) * chunk_size, dev_end)
        train_end = val_start - embargo_bars
        if train_end <= 0:
            continue
        train_idx = np.arange(0, train_end, dtype=np.int64)
        val_idx = np.arange(val_start, val_end, dtype=np.int64)
        folds.append(SplitIndices(train_idx=train_idx, val_idx=val_idx, fold=k))

    return SplitsResult(folds=folds, oos_idx=oos_idx, embargo_bars=embargo_bars)
