"""Prior-day / overnight / RTH-open anchors. Answers Q11 / Q12 / Q14.

Why this matters: discretionary index-futures traders watch a specific set of
horizontal reference levels constantly — prior-day high/low, overnight high/low,
RTH open, gap size. These are the highest-traffic intraday anchors because
algos and traders pile stops + targets at them. The system currently has KDE
swing-points (key_levels) and untouched HTF extremes (liquidity_zones), but
neither surfaces PDH/PDL/ONH/ONL sharply — they get smoothed away.

Features (9 total, all causal):
  pda_dist_to_pdh_atr        signed distance to prior-day RTH high (negative if above)
  pda_dist_to_pdl_atr        signed distance to prior-day RTH low
  pda_dist_to_onh_atr        signed distance to overnight high
  pda_dist_to_onl_atr        signed distance to overnight low
  pda_dist_to_rth_open_atr   signed distance to today's RTH open
  pda_gap_size_atr           today's RTH-open minus prior-day RTH-close, in ATR
  pda_above_pdh              1.0 if close > pdh else 0.0 (above the level entirely)
  pda_below_pdl              1.0 if close < pdl else 0.0
  pda_in_overnight_range     1.0 if onl <= close <= onh else 0.0

CAUSALITY: anchors at bar T are computed strictly from BARS BEFORE T's session
boundary. Yesterday's RTH high is known at today's RTH open and after. The
overnight high is known continuously during overnight (using only bars seen so
far), then finalized at today's RTH open. No forward leakage.

RTH defined per altus/features/families/session_time.py:
  NY_RTH_START_UTC = 13.5  (09:30 ET)
  NY_RTH_END_UTC = 20.0    (16:00 ET)
DST not corrected — we accept ±1h drift for the EDT/EST shift since the
boundaries are used as reference anchors, not gates.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from altus.features.families.session_time import NY_RTH_END_UTC, NY_RTH_START_UTC


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


def _compute_anchors(df_1m: pd.DataFrame) -> pd.DataFrame:
    """For each 1m bar, return PDH/PDL/ONH/ONL/RTH_open/PD_close as the
    most-recently-known values.

    Implementation: walk forward through the data session-by-session, accumulate
    running min/max within each session bucket, snapshot the values when each
    bucket closes. All anchors at bar T reflect only data with timestamp < T's
    session-relevant boundary.
    """
    idx = df_1m.index
    h = df_1m["high"].to_numpy(dtype=np.float64)
    l = df_1m["low"].to_numpy(dtype=np.float64)
    c = df_1m["close"].to_numpy(dtype=np.float64)
    o = df_1m["open"].to_numpy(dtype=np.float64)
    n = len(df_1m)

    hour_utc = (idx.hour + idx.minute / 60.0).to_numpy()
    in_rth = (hour_utc >= NY_RTH_START_UTC) & (hour_utc < NY_RTH_END_UTC)

    # Output arrays — initialize to NaN; we fill in causally as data arrives.
    pdh = np.full(n, np.nan, dtype=np.float32)
    pdl = np.full(n, np.nan, dtype=np.float32)
    onh = np.full(n, np.nan, dtype=np.float32)
    onl = np.full(n, np.nan, dtype=np.float32)
    rth_open = np.full(n, np.nan, dtype=np.float32)
    pd_close = np.full(n, np.nan, dtype=np.float32)

    # Running accumulators
    cur_rth_high = -np.inf
    cur_rth_low = np.inf
    cur_on_high = -np.inf
    cur_on_low = np.inf
    cur_rth_open_value = np.nan
    cur_pd_close_value = np.nan

    # Finalized previous-session snapshots — exposed once a session closes
    last_pdh = np.nan
    last_pdl = np.nan
    last_pd_close = np.nan
    last_onh = np.nan
    last_onl = np.nan
    last_rth_open = np.nan

    prev_in_rth = False
    prev_close_value = np.nan

    for i in range(n):
        # Snapshot the most-recently-known anchors BEFORE incorporating bar i.
        # This ensures bar i's feature value reflects only data strictly before i.
        pdh[i] = last_pdh
        pdl[i] = last_pdl
        onh[i] = last_onh
        onl[i] = last_onl
        rth_open[i] = last_rth_open
        pd_close[i] = last_pd_close

        bar_in_rth = bool(in_rth[i])

        # Transition: just entered RTH → finalize prior session's overnight,
        # snapshot prior-day close, set today's RTH open.
        if bar_in_rth and not prev_in_rth:
            # Lock the previous session's PDH/PDL if we had one
            if cur_rth_high > -np.inf:
                last_pdh = cur_rth_high
                last_pdl = cur_rth_low
                last_pd_close = prev_close_value
            # Lock the overnight range (since prior-session-close to now)
            if cur_on_high > -np.inf:
                last_onh = cur_on_high
                last_onl = cur_on_low
            # Today's RTH open = this bar's open
            last_rth_open = float(o[i])
            # Reset RTH accumulators for the new session
            cur_rth_high = -np.inf
            cur_rth_low = np.inf
            cur_on_high = -np.inf
            cur_on_low = np.inf

        # Transition: just exited RTH → start a new overnight accumulator
        if not bar_in_rth and prev_in_rth:
            cur_on_high = -np.inf
            cur_on_low = np.inf

        # Accumulate within whichever session we're in
        if bar_in_rth:
            cur_rth_high = max(cur_rth_high, float(h[i]))
            cur_rth_low = min(cur_rth_low, float(l[i]))
        else:
            cur_on_high = max(cur_on_high, float(h[i]))
            cur_on_low = min(cur_on_low, float(l[i]))

        # Track the running close — used to capture "prior-day close" at the
        # moment RTH ends.
        prev_close_value = float(c[i])
        prev_in_rth = bar_in_rth

    return pd.DataFrame({
        "_pdh": pdh,
        "_pdl": pdl,
        "_onh": onh,
        "_onl": onl,
        "_rth_open": rth_open,
        "_pd_close": pd_close,
    }, index=idx)


def compute(df_1m: pd.DataFrame) -> pd.DataFrame:
    close = df_1m["close"].to_numpy(dtype=np.float64)
    atr = _atr(df_1m, n=14).replace(0, np.nan).ffill().fillna(1.0).to_numpy(dtype=np.float64)
    atr_safe = np.maximum(atr, EPS)

    anchors = _compute_anchors(df_1m)
    pdh = anchors["_pdh"].to_numpy()
    pdl = anchors["_pdl"].to_numpy()
    onh = anchors["_onh"].to_numpy()
    onl = anchors["_onl"].to_numpy()
    rth_open = anchors["_rth_open"].to_numpy()
    pd_close = anchors["_pd_close"].to_numpy()

    # Signed distance: positive = anchor is above current close, negative = below.
    # Clip to a sane range — when anchors haven't been observed yet (start of data),
    # NaN propagates; we fill it with 0 (neutral signal).
    def _sd(anchor):
        return np.nan_to_num((anchor - close) / atr_safe, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    dist_pdh = _sd(pdh)
    dist_pdl = _sd(pdl)
    dist_onh = _sd(onh)
    dist_onl = _sd(onl)
    dist_rth_open = _sd(rth_open)

    # Gap: today's RTH open minus prior-day close, in ATR.
    gap = np.nan_to_num((rth_open - pd_close) / atr_safe, nan=0.0).astype(np.float32)

    # Above/below flags (0/1)
    above_pdh = np.where(np.isnan(pdh), 0.0, (close > pdh).astype(np.float32))
    below_pdl = np.where(np.isnan(pdl), 0.0, (close < pdl).astype(np.float32))
    in_on_range = np.where(
        np.isnan(onh) | np.isnan(onl),
        0.0,
        ((close <= onh) & (close >= onl)).astype(np.float32),
    )

    return pd.DataFrame({
        "pda_dist_to_pdh_atr": dist_pdh,
        "pda_dist_to_pdl_atr": dist_pdl,
        "pda_dist_to_onh_atr": dist_onh,
        "pda_dist_to_onl_atr": dist_onl,
        "pda_dist_to_rth_open_atr": dist_rth_open,
        "pda_gap_size_atr": gap,
        "pda_above_pdh": above_pdh.astype(np.float32),
        "pda_below_pdl": below_pdl.astype(np.float32),
        "pda_in_overnight_range": in_on_range.astype(np.float32),
    }, index=df_1m.index)


FEATURE_COLUMNS = (
    "pda_dist_to_pdh_atr",
    "pda_dist_to_pdl_atr",
    "pda_dist_to_onh_atr",
    "pda_dist_to_onl_atr",
    "pda_dist_to_rth_open_atr",
    "pda_gap_size_atr",
    "pda_above_pdh",
    "pda_below_pdl",
    "pda_in_overnight_range",
)
