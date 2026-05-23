"""Sanity tests for purged walk-forward splits.

We test:
  1. No overlap between train and val within a fold
  2. Embargo gap is >= embargo_bars between train end and val start
  3. OOS lockbox is fully disjoint from any dev fold
  4. Folds are sequential in time (val_i < val_{i+1})
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from altus.splits import purged_walk_forward


def _fake_index(n_minutes: int) -> pd.DatetimeIndex:
    return pd.date_range("2024-01-01", periods=n_minutes, freq="1min", tz="UTC")


def test_basic_layout():
    # 2 years of 1m bars
    ts = _fake_index(60 * 24 * 365 * 2)
    res = purged_walk_forward(ts, n_folds=5, embargo_bars=60, oos_months=6)

    assert len(res.folds) == 5
    print(f"folds: {len(res.folds)}, oos size: {len(res.oos_idx):,}")

    # OOS disjoint from all dev folds
    oos_set = set(res.oos_idx.tolist())
    for f in res.folds:
        train_set = set(f.train_idx.tolist())
        val_set = set(f.val_idx.tolist())
        assert not train_set & val_set, f"fold {f.fold}: train/val overlap!"
        assert not (train_set | val_set) & oos_set, f"fold {f.fold}: dev/OOS overlap!"
        # Embargo: max(train) + embargo <= min(val)
        gap = int(f.val_idx.min()) - int(f.train_idx.max())
        assert gap >= 60, f"fold {f.fold}: embargo gap {gap} < 60"
        print(f"  fold {f.fold}: train=[{f.train_idx.min():,}..{f.train_idx.max():,}] "
              f"({len(f.train_idx):,}), val=[{f.val_idx.min():,}..{f.val_idx.max():,}] "
              f"({len(f.val_idx):,}), gap={gap}")

    # Folds sequential
    val_starts = [f.val_idx.min() for f in res.folds]
    assert val_starts == sorted(val_starts), "folds not chronologically sequential"

    # OOS at the tail
    assert res.oos_idx.min() > max(f.val_idx.max() for f in res.folds), \
        "OOS should start after the last dev fold"

    print("test_basic_layout: PASS")


def test_smoke_no_oos():
    """The smoke config: 0-month OOS, 1 fold."""
    ts = _fake_index(60 * 24 * 90)  # 3 months
    res = purged_walk_forward(ts, n_folds=1, embargo_bars=60, oos_months=0)
    assert len(res.folds) == 1
    assert len(res.oos_idx) == 0
    fold = res.folds[0]
    assert fold.val_idx.min() - fold.train_idx.max() >= 60
    print(f"test_smoke_no_oos: PASS (train={len(fold.train_idx):,}, val={len(fold.val_idx):,})")


if __name__ == "__main__":
    test_basic_layout()
    test_smoke_no_oos()
