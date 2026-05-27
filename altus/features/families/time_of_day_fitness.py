"""Time-of-Day Fitness — per-setup ideal-window match.

Each setup has a best-time window (per SETUPS.md timing specs). ORB only
fires after the opening range establishes (first 30-90min of NY RTH).
EOD reversion only works in last 30min. Trend pullback works mid-session.

This family computes a fitness score per setup × current time, boosting
conviction when the setup fires in its historically-best window.

Per-setup time windows (UTC, EDT-anchored):

  | Setup | Best window (UTC hours) |
  |-------|-------------------------|
  | sfs   | 13:30-20:00 (any RTH)  |
  | sfa   | 14:30-19:30 (mid-session) |
  | sld   | 13:30-20:00 (any RTH)  |
  | orb   | 14:00-15:00 (first hour after OR locked) |
  | svwap | 14:30-19:30 (mid-session) |
  | spb   | 14:00-19:00 (early-mid session) |
  | scomp | 14:00-19:30 (excludes opening + close noise) |
  | seod  | 19:30-20:00 (last 30min only) |

Outside the best window, fitness drops to 0.3 (still possible, just weaker
historical edge). Inside the window, fitness is 1.0.

Features (9 total):
  tof_fit_sfs / tof_fit_sfa / ... / tof_fit_seod
  tof_avg_fit — average fitness across all 8 setups
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from altus.features.families.setup_utils import hour_of_day_utc


# Best windows: (start_hour_utc, end_hour_utc) — both inclusive
BEST_WINDOWS = {
    "sfs":   (13.5, 20.0),   # any RTH
    "sfa":   (14.5, 19.5),   # mid-session
    "sld":   (13.5, 20.0),   # any RTH
    "orb":   (14.0, 15.0),   # first hour after OR locked
    "svwap": (14.5, 19.5),   # mid-session
    "spb":   (14.0, 19.0),   # early-mid session
    "scomp": (14.0, 19.5),   # excludes opening + close noise
    "seod":  (19.5, 20.0),   # last 30min only
}

# Fitness score outside the best window
OUT_OF_WINDOW_FITNESS = 0.3


def compute(df_primary: pd.DataFrame, df_1m: pd.DataFrame | None = None) -> pd.DataFrame:
    n = len(df_primary)
    idx = df_primary.index
    hour = hour_of_day_utc(idx)

    out = {}
    avg_fit = np.zeros(n, dtype=np.float32)
    for sid, (start, end) in BEST_WINDOWS.items():
        in_window = (hour >= start) & (hour < end)
        fit = np.where(in_window, 1.0, OUT_OF_WINDOW_FITNESS).astype(np.float32)
        out[f"tof_fit_{sid}"] = fit
        avg_fit += fit / len(BEST_WINDOWS)

    out["tof_avg_fit"] = avg_fit

    return pd.DataFrame(out, index=idx)


FEATURE_COLUMNS = (
    "tof_fit_sfs", "tof_fit_sfa", "tof_fit_sld", "tof_fit_orb",
    "tof_fit_svwap", "tof_fit_spb", "tof_fit_scomp", "tof_fit_seod",
    "tof_avg_fit",
)
