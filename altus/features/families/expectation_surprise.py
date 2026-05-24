"""Family E12 (Phase E): Expected vs actual surprise. Answers Q33 (dog that didn't bark).

Why this matters: information lives in the GAP between what context says should
happen and what actually happens. If it's 14:30 UTC (peak NY open vol normally)
and vol is half of typical, something structural is different. If we're in a
confirmed trend regime but pullbacks aren't holding above prior swings, the
trend is fragile. These surprise signals are what experienced traders watch
for.

We build a simple normative-expectation baseline (conditional on hour-of-day
and day-of-week), compare to actual, and expose the surprise.

Features (4 total):
  • es_vol_surprise           actual_vol / expected_vol_given_hour_dow - 1.0
  • es_move_surprise          actual_abs_return / expected_abs_return - 1.0
  • es_volume_surprise        actual_volume / expected_volume - 1.0
  • es_combined_surprise      mean of the three above (scalar overall surprise)

CAUSALITY: the conditional expectation table is built from the FIRST 5000 bars
of input (warmup calibration) and is fixed thereafter. This is causal and
deterministic given the input.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


EPS = 1e-9
CALIBRATION_BARS = 5000


def _build_norm_table(
    series: pd.Series,
    hour: np.ndarray,
    dow: np.ndarray,
    n_cal: int,
) -> dict[tuple[int, int], float]:
    """Build mean(series) keyed by (hour, dow), using only first n_cal bars."""
    n_cal = min(n_cal, len(series))
    cal_series = series.iloc[:n_cal].to_numpy()
    cal_hour = hour[:n_cal]
    cal_dow = dow[:n_cal]
    table: dict[tuple[int, int], float] = {}
    for h in range(24):
        for d in range(7):
            mask = (cal_hour == h) & (cal_dow == d)
            if mask.sum() >= 10:
                table[(h, d)] = float(np.nanmean(cal_series[mask]))
    # Global fallback
    global_mean = float(np.nanmean(cal_series))
    return {"global": global_mean, **{f"{h}_{d}": v for (h, d), v in table.items()}}


def _apply_norm_table(table: dict, hour: np.ndarray, dow: np.ndarray) -> np.ndarray:
    """Look up expected value at each row using (hour, dow); fall back to global."""
    out = np.full(len(hour), table.get("global", 0.0), dtype=np.float64)
    for i in range(len(hour)):
        k = f"{int(hour[i])}_{int(dow[i])}"
        if k in table:
            out[i] = table[k]
    return out


def compute(df_1m: pd.DataFrame) -> pd.DataFrame:
    idx = df_1m.index
    hour = idx.hour.to_numpy()
    dow = idx.dayofweek.to_numpy()

    close = df_1m["close"]
    volume = df_1m["volume"]
    log_ret = np.log(close / close.shift(1)).fillna(0.0)

    abs_ret = log_ret.abs()
    realized_vol_60 = log_ret.rolling(60, min_periods=10).std().fillna(0.0)
    vol_60 = volume.rolling(60, min_periods=10).mean().fillna(0.0)

    vol_table = _build_norm_table(realized_vol_60, hour, dow, CALIBRATION_BARS)
    abs_ret_table = _build_norm_table(abs_ret, hour, dow, CALIBRATION_BARS)
    volume_table = _build_norm_table(vol_60, hour, dow, CALIBRATION_BARS)

    expected_vol = _apply_norm_table(vol_table, hour, dow)
    expected_abs_ret = _apply_norm_table(abs_ret_table, hour, dow)
    expected_volume = _apply_norm_table(volume_table, hour, dow)

    vol_surprise = (realized_vol_60.to_numpy() / np.maximum(expected_vol, EPS)) - 1.0
    move_surprise = (abs_ret.to_numpy() / np.maximum(expected_abs_ret, EPS)) - 1.0
    volume_surprise = (vol_60.to_numpy() / np.maximum(expected_volume, EPS)) - 1.0

    vol_surprise = np.clip(np.nan_to_num(vol_surprise, nan=0.0), -3, 3).astype(np.float32)
    move_surprise = np.clip(np.nan_to_num(move_surprise, nan=0.0), -3, 3).astype(np.float32)
    volume_surprise = np.clip(np.nan_to_num(volume_surprise, nan=0.0), -3, 3).astype(np.float32)

    combined = ((vol_surprise + move_surprise + volume_surprise) / 3.0).astype(np.float32)

    return pd.DataFrame({
        "es_vol_surprise": vol_surprise,
        "es_move_surprise": move_surprise,
        "es_volume_surprise": volume_surprise,
        "es_combined_surprise": combined,
    }, index=idx)


FEATURE_COLUMNS = (
    "es_vol_surprise",
    "es_move_surprise",
    "es_volume_surprise",
    "es_combined_surprise",
)
