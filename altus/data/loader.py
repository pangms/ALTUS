"""Data loaders for MNQ and cross-asset 1-minute OHLCV parquets.

The MNQ parquet in /data is already back-adjusted continuous (roll gaps at known
quarterly contract changes are sub-5pt on a ~20k-point series, confirmed against
the roll_log). So we do NOT re-adjust here. Cross-asset parquets store timestamp
as a column rather than the index and only cover ~2024-2026.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

from altus.config import DATA_DIR


CROSS_ASSETS = ("es", "nq", "cl", "gc", "si", "zb", "zn", "6e")


def _read_parquet(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"missing parquet: {path}")
    return pd.read_parquet(path)


def load_mnq(start: str | None = None, end: str | None = None) -> pd.DataFrame:
    """Load MNQ 1-min OHLCV as a UTC-indexed DataFrame.

    Returns columns: ['open', 'high', 'low', 'close', 'volume'].
    """
    df = _read_parquet(DATA_DIR / "mnq_1min.parquet")
    # Already timestamp-indexed in the source — sanity check.
    if not isinstance(df.index, pd.DatetimeIndex):
        df = df.set_index("timestamp")
    df.index = pd.to_datetime(df.index, utc=True)
    df = df.sort_index()
    # Drop duplicate timestamps if any (data hygiene).
    df = df[~df.index.duplicated(keep="first")]
    if start is not None:
        df = df.loc[df.index >= pd.Timestamp(start, tz="UTC")]
    if end is not None:
        df = df.loc[df.index <= pd.Timestamp(end, tz="UTC")]
    return df


def load_cross_asset(symbol: str, start: str | None = None, end: str | None = None) -> pd.DataFrame:
    """Load a cross-asset 1-min parquet. Returns ['open','high','low','close','volume']."""
    sym = symbol.lower()
    if sym not in CROSS_ASSETS and sym != "mnq":
        raise ValueError(f"unknown symbol: {symbol}")
    df = _read_parquet(DATA_DIR / f"{sym}_1min.parquet")
    if "timestamp" in df.columns:
        df = df.set_index("timestamp")
    df.index = pd.to_datetime(df.index, utc=True)
    df = df.sort_index()
    df = df[~df.index.duplicated(keep="first")]
    if start is not None:
        df = df.loc[df.index >= pd.Timestamp(start, tz="UTC")]
    if end is not None:
        df = df.loc[df.index <= pd.Timestamp(end, tz="UTC")]
    keep = ["open", "high", "low", "close", "volume"]
    return df[keep]


def load_all_aligned(
    symbols: Iterable[str] = ("mnq", *CROSS_ASSETS),
    start: str | None = None,
    end: str | None = None,
    require_overlap: bool = True,
) -> pd.DataFrame:
    """Load multiple assets and align them on the MNQ 1-min timestamp grid.

    When `require_overlap=True`, the returned frame starts at the latest of all
    available start times (so all columns are populated). Columns are prefixed
    with the symbol, e.g. 'mnq_close', 'es_volume'.
    """
    frames: list[pd.DataFrame] = []
    for sym in symbols:
        if sym == "mnq":
            f = load_mnq(start=start, end=end)
        else:
            f = load_cross_asset(sym, start=start, end=end)
        f = f.add_prefix(f"{sym}_")
        frames.append(f)

    if require_overlap:
        common_start = max(f.index.min() for f in frames)
        common_end = min(f.index.max() for f in frames)
        frames = [f.loc[common_start:common_end] for f in frames]

    # Reindex everything onto MNQ's grid (1-min UTC bars). For cross-assets we
    # forward-fill at most a few bars to handle their occasional missing prints
    # but we cap the fill so we don't paper over true outages.
    mnq_grid = frames[0].index  # mnq is first by convention
    aligned = []
    for f in frames:
        f2 = f.reindex(mnq_grid)
        # Forward-fill price columns up to 5 bars; never fill volume (a gap is real).
        price_cols = [c for c in f2.columns if not c.endswith("_volume")]
        vol_cols = [c for c in f2.columns if c.endswith("_volume")]
        f2[price_cols] = f2[price_cols].ffill(limit=5)
        f2[vol_cols] = f2[vol_cols].fillna(0)
        aligned.append(f2)

    out = pd.concat(aligned, axis=1)
    return out
