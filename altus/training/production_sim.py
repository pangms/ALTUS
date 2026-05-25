"""Layer 3.1 — production-honest trading simulation.

Where `sim_pnl.py` evaluates Layer 1/2 signals per-bar in isolation (one
contract, no concurrency, simple percentile gate), this sim adds the L3
execution layer:

  * Grade-based position sizing
        B  (top 20%)  -> 1 MNQ
        A  (top  5%)  -> 2 MNQ
        A+ (top  1%)  -> 3 MNQ
        A++(top 0.5%) -> pyramid-only second leg, 3 MNQ
  * No-overlap concurrency by default. Take the first qualifying signal in
    each cluster and hold until its assumed exit.
  * A++ pyramiding exception: stack a second (and optionally third) position
    when (i) currently holding, (ii) ≥5min since most-recent entry, (iii) new
    signal is A++ tier, (iv) same direction as open positions, (v) stack
    depth < MAX_STACK_DEPTH.
  * TopStep telemetry (informational, never gates): per-day realized PnL,
    intraday peak-to-trough drawdown, max concurrent contracts, counter of
    days that would have tripped the broker's daily-loss/trailing-DD caps.

Why this matters: the existing per-bar sim is correct *per signal* but
inflates trades/day by 5–10× because clusters of consecutive top-percentile
bars all "trade" simultaneously, and PnL by 1–3× because it ignores sizing.
This module is the honest analog of the production execution rule and is
what we should A/B against when evaluating the live decision stack.

Exit-time approximation
-----------------------
For concurrency we need to know when each trade closes. The labeler records
`time_to_long_tp` / `time_to_short_tp` (bar index of TP hit, or H if no hit)
but not the SL-hit bar. Rather than assume an asymmetric exit time
(TP→quick, SL→full hold) — which would bias pyramid eligibility — we use a
fixed assumed hold equal to the label horizon (default 60 bars = 1h). This
is conservative: it over-counts position duration and therefore *over*-
restricts new entries, so the sim under-states (not over-states) trades/day.
A future refinement can recover the exact SL bar from the labeler.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from altus.config import (
    COMMISSION_RT_USD,
    LABEL_HORIZON_BARS,
    POINT_VALUE_USD,
    SL_POINTS,
    SLIPPAGE_RT_POINTS,
    TP_POINTS,
)


# Numeric grade encoding. 0 = no signal. Used internally for fast comparison.
GRADE_NONE = 0
GRADE_B = 1
GRADE_A = 2
GRADE_A_PLUS = 3
GRADE_A_PLUS_PLUS = 4

_GRADE_NAME = {
    GRADE_NONE: "-",
    GRADE_B: "B",
    GRADE_A: "A",
    GRADE_A_PLUS: "A+",
    GRADE_A_PLUS_PLUS: "A++",
}


@dataclass
class GradeThresholds:
    """Top-percentile cutoffs for each grade tier. Nested: A++ ⊂ A+ ⊂ A ⊂ B."""
    b_pct: float = 0.20
    a_pct: float = 0.05
    a_plus_pct: float = 0.01
    a_plus_plus_pct: float = 0.005


@dataclass
class L3Config:
    grades: GradeThresholds = field(default_factory=GradeThresholds)

    # Contracts per entry. A++ only used for pyramid second leg; the first
    # entry of a signal that happens to be A++ tier sizes as A+ (3 MNQ).
    size_b: int = 1
    size_a: int = 2
    size_a_plus: int = 3
    size_a_plus_plus_pyramid: int = 3

    max_stack_depth: int = 3
    pyramid_min_gap_min: int = 5

    # Held-position duration for the concurrency check. See module docstring.
    assumed_hold_bars: int = LABEL_HORIZON_BARS

    starting_capital_usd: float = 50_000.0

    # ----- HARD RULES -----
    # End-of-day flatten: no new entries within this many minutes of NY RTH
    # close. Any open positions are force-flattened at the close. The cutoff
    # is computed in ET (16:00) and DST-converted to UTC per timestamp.
    eod_no_entry_min: int = 20
    eod_force_flatten: bool = True

    # Consecutive-loss cooldown (per-trade, not per-day; streak resets on a win).
    # After the Nth consecutive loss closes, no new entries for `cooldown_min`
    # minutes. Schedule: (loss_count, lockout_minutes). Anything >= the last
    # entry uses the last entry's lockout. User spec: 0/10/20/30/30+.
    cooldown_schedule: tuple = ((2, 10), (3, 20), (4, 30))

    # ----- TopStep telemetry (informational only, NEVER gates entries) -----
    topstep_daily_loss_usd: float = 1_000.0
    topstep_trailing_dd_usd: float = 2_000.0


@dataclass
class L3Result:
    n_trades: int
    n_long: int
    n_short: int
    n_trades_by_grade: dict          # {"B": n, "A": n, "A+": n, "A++": n}
    n_pyramid_entries: int
    avg_position_size: float
    max_concurrent_contracts: int

    win_rate: float
    avg_r: float
    expectancy_usd: float
    total_pnl_usd: float
    sharpe: float
    sortino: float
    max_drawdown_usd: float
    max_drawdown_pct: float
    profit_factor: float
    trades_per_day: float
    monthly_pnl: pd.Series
    pct_positive_months: float
    equity_curve: pd.Series

    # TopStep telemetry
    worst_day_pnl_usd: float
    worst_intraday_dd_usd: float
    n_days_would_trip_daily_loss: int
    n_days_would_trip_trailing_dd: int

    # Hard-rule diagnostics
    n_eod_entries_blocked: int
    n_eod_force_flattened: int
    n_cooldown_entries_blocked: int
    max_consecutive_losses: int

    # Raw trade log (one row per executed entry, for inspection / A/B)
    trades: pd.DataFrame

    def summary_line(self) -> str:
        gb = self.n_trades_by_grade
        return (
            f"trades={self.n_trades} ({self.n_long}L/{self.n_short}S) "
            f"B/A/A+/A++={gb['B']}/{gb['A']}/{gb['A+']}/{gb['A++']} "
            f"pyr={self.n_pyramid_entries} "
            f"| win={self.win_rate:.3f} avgR={self.avg_r:+.3f} "
            f"| PnL=${self.total_pnl_usd:,.0f} Sharpe={self.sharpe:.2f} "
            f"DD={self.max_drawdown_pct:.1%} | TPD={self.trades_per_day:.1f} "
            f"avgSize={self.avg_position_size:.2f} maxStack={self.max_concurrent_contracts}"
        )


# ---------------------------------------------------------------------------
# Grade assignment
# ---------------------------------------------------------------------------

_PROB_FLOOR = 1e-9  # below this we treat the entry as "no signal" (cascade padding)


def _top_pct_mask(probs: np.ndarray, pct: float) -> np.ndarray:
    """Rank-based top-percentile selector. See sim_pnl.py for rationale —
    rank > quantile-threshold under isotonic calibration ties.

    Percentile is computed over ENTRIES WITH A SIGNAL (prob > _PROB_FLOOR).
    Cascade cases pad non-candidate bars with prob=0; without this floor,
    argpartition would happily mark zero-probability bars as "top K%".
    """
    n = len(probs)
    active = probs > _PROB_FLOOR
    n_active = int(active.sum())
    if n_active == 0:
        return np.zeros(n, dtype=bool)
    k = max(1, int(np.ceil(pct * n_active)))
    k = min(k, n_active)
    active_idx = np.where(active)[0]
    active_probs = probs[active_idx]
    # Top-k within the active subset, then map back to global indices.
    top_local = np.argpartition(-active_probs, k - 1)[:k]
    mask = np.zeros(n, dtype=bool)
    mask[active_idx[top_local]] = True
    return mask


def assign_grades(probs: np.ndarray, thresholds: GradeThresholds) -> np.ndarray:
    """Per-bar grade label using cascading top-pct masks.

    Returns int8 array in {0, 1, 2, 3, 4} = {none, B, A, A+, A++}.
    The highest grade wins because the nested masks overwrite in order.
    Percentiles are computed over ENTRIES WITH A SIGNAL only — cascade
    padding zeros don't get fabricated grades.
    """
    n = len(probs)
    if n == 0:
        return np.zeros(0, dtype=np.int8)
    grades = np.zeros(n, dtype=np.int8)
    grades[_top_pct_mask(probs, thresholds.b_pct)] = GRADE_B
    grades[_top_pct_mask(probs, thresholds.a_pct)] = GRADE_A
    grades[_top_pct_mask(probs, thresholds.a_plus_pct)] = GRADE_A_PLUS
    grades[_top_pct_mask(probs, thresholds.a_plus_plus_pct)] = GRADE_A_PLUS_PLUS
    return grades


# ---------------------------------------------------------------------------
# Per-signal PnL (same model as sim_pnl.py)
# ---------------------------------------------------------------------------

def _per_side_pnl_pts(
    label: np.ndarray, mfe: np.ndarray, mae: np.ndarray,
    tp_pts: float, sl_pts: float, cost_pts: float,
) -> np.ndarray:
    pnl = mfe - mae
    full_stop = (label == 0) & (mae >= sl_pts)
    full_tp = label == 1
    pnl = np.where(full_stop, -sl_pts, pnl)
    pnl = np.where(full_tp, tp_pts, pnl)
    return pnl - cost_pts


def _next_ny_rth_close_utc(ts: pd.Timestamp) -> pd.Timestamp:
    """Next NY RTH close (16:00 ET) at or after `ts`, in UTC (naive).

    Handles DST via tz-aware conversion. If `ts` already past today's 16:00 ET,
    returns tomorrow's. Weekend handling is best-effort — Globex is closed
    weekends so we don't expect entries then anyway; this still returns the
    next *calendar* 16:00 ET (which may land on Sat/Sun, harmlessly).
    """
    if ts.tzinfo is None:
        ts_utc = ts.tz_localize("UTC")
    else:
        ts_utc = ts.tz_convert("UTC")
    et = ts_utc.tz_convert("America/New_York")
    close_et = et.normalize() + pd.Timedelta(hours=16)
    if et > close_et:
        close_et = close_et + pd.Timedelta(days=1)
    return close_et.tz_convert("UTC").tz_localize(None)


def _cooldown_min_for(n_consec_losses: int, schedule: tuple) -> int:
    """Look up the cooldown (minutes) for a given consecutive-loss count.

    `schedule` is a tuple of (threshold, minutes) sorted ascending by threshold.
    Returns 0 when n is below the first threshold; otherwise returns the
    minutes for the highest threshold that n still meets or exceeds.
    """
    if n_consec_losses < schedule[0][0]:
        return 0
    cd = 0
    for threshold, minutes in schedule:
        if n_consec_losses >= threshold:
            cd = minutes
    return cd


# ---------------------------------------------------------------------------
# Main simulation
# ---------------------------------------------------------------------------

def simulate_l3(
    timestamps: np.ndarray,
    preds: dict[str, np.ndarray],
    truths: dict[str, np.ndarray],
    cfg: L3Config | None = None,
) -> L3Result:
    """Run L3.1 execution rules over per-bar signals.

    Parameters
    ----------
    timestamps : 1D np.datetime64 array, aligned to preds/truths, NOT required
                 to be sorted — we sort internally.
    preds      : dict with 'long_tp_prob', 'short_tp_prob'.
    truths     : dict with 'long_tp', 'short_tp', 'mfe_long', 'mae_long',
                 'mfe_short', 'mae_short' (same contract as sim_pnl.py).
    """
    cfg = cfg or L3Config()
    n = len(timestamps)
    if n == 0:
        return _empty_result()

    ts_idx = pd.DatetimeIndex(timestamps)
    sort_order = np.argsort(ts_idx.values)
    ts_sorted = ts_idx[sort_order]
    long_p = preds["long_tp_prob"][sort_order]
    short_p = preds["short_tp_prob"][sort_order]

    cost_pts = SLIPPAGE_RT_POINTS + (COMMISSION_RT_USD / POINT_VALUE_USD)
    long_pnl_pts = _per_side_pnl_pts(
        truths["long_tp"][sort_order], truths["mfe_long"][sort_order],
        truths["mae_long"][sort_order], TP_POINTS, SL_POINTS, cost_pts,
    )
    short_pnl_pts = _per_side_pnl_pts(
        truths["short_tp"][sort_order], truths["mfe_short"][sort_order],
        truths["mae_short"][sort_order], TP_POINTS, SL_POINTS, cost_pts,
    )

    long_grade = assign_grades(long_p, cfg.grades)
    short_grade = assign_grades(short_p, cfg.grades)

    # Conflict resolution: pick the side with the HIGHER grade. Skip only when
    # both sides land on the same grade tier (genuine model indecision). Nested
    # grade masks mean an A++ long will almost always coincide with at least a
    # B short — naive "any-overlap-kills" would gut all top signals, so we
    # rank by grade and let probability break ties at the same tier.
    has_long = long_grade > 0
    has_short = short_grade > 0
    long_higher = has_long & (long_grade > short_grade)
    short_higher = has_short & (short_grade > long_grade)
    same_tier = has_long & has_short & (long_grade == short_grade)
    # At the same tier, prefer the side with the higher probability — but only
    # if it's meaningfully higher, otherwise skip.
    PROB_TIE_MARGIN = 0.02
    long_wins_tie = same_tier & (long_p > short_p + PROB_TIE_MARGIN)
    short_wins_tie = same_tier & (short_p > long_p + PROB_TIE_MARGIN)
    take_long = (has_long & ~has_short) | long_higher | long_wins_tie
    take_short = (has_short & ~has_long) | short_higher | short_wins_tie
    cand_mask = take_long | take_short

    cand_indices = np.where(cand_mask)[0]
    if cand_indices.size == 0:
        return _empty_result()

    hold = pd.Timedelta(minutes=cfg.assumed_hold_bars)
    pyramid_gap = pd.Timedelta(minutes=cfg.pyramid_min_gap_min)
    eod_buffer = pd.Timedelta(minutes=cfg.eod_no_entry_min)

    open_pos: list[dict] = []
    trades: list[dict] = []
    n_pyramid = 0
    n_eod_blocked = 0
    n_eod_flat = 0
    n_cd_blocked = 0
    grade_counts = {"B": 0, "A": 0, "A+": 0, "A++": 0}

    # Consecutive-loss tracker. Updated each time a position's exit_ts is
    # crossed by the current candidate's ts (i.e., we "observe" the close).
    n_consec_losses = 0
    max_consec_losses = 0
    cooldown_until: pd.Timestamp | None = None

    # Concurrency event log for max-stack-contracts computation.
    contract_events: list[tuple[pd.Timestamp, int]] = []

    def _close_position(p, realized_exit_ts):
        """Register a closed position: update loss streak + emit contract event."""
        nonlocal n_consec_losses, max_consec_losses, cooldown_until
        # Use the position's per-signal PnL (proxy; see module docstring).
        if p["pnl_pts"] <= 0:
            n_consec_losses += 1
            max_consec_losses = max(max_consec_losses, n_consec_losses)
            cd = _cooldown_min_for(n_consec_losses, cfg.cooldown_schedule)
            if cd > 0:
                cooldown_until = realized_exit_ts + pd.Timedelta(minutes=cd)
        else:
            n_consec_losses = 0
            cooldown_until = None
        contract_events.append((realized_exit_ts, -p["contracts"]))

    for i in cand_indices:
        ts = ts_sorted[i]

        # "Next NY close" for the current ts — drives both the EoD-block window
        # check and any effective_exit caps for open positions in this iteration.
        ny_close_for_ts = _next_ny_rth_close_utc(ts) if cfg.eod_force_flatten else None

        # Step 1: close any positions whose exit_ts has been reached. Each
        # position uses its OWN "next NY close after entry" (cached at entry
        # time) so a post-close Asia/London entry doesn't get a stale today-cap.
        still_open = []
        for p in open_pos:
            natural_exit = p["exit_ts"]
            effective_exit = (
                min(natural_exit, p["ny_close_cap"])
                if cfg.eod_force_flatten else natural_exit
            )
            if effective_exit <= ts:
                _close_position(p, effective_exit)
                if effective_exit < natural_exit:
                    n_eod_flat += 1
            else:
                still_open.append(p)
        open_pos = still_open

        # Step 2: EoD no-entry window. Block ONLY in (ny_close - buffer, ny_close];
        # after NY close, trading resumes for the next session (Asia → London → NY).
        if cfg.eod_force_flatten and (ny_close_for_ts - eod_buffer) <= ts < ny_close_for_ts:
            n_eod_blocked += 1
            continue

        # Step 3: consecutive-loss cooldown.
        if cooldown_until is not None and ts < cooldown_until:
            n_cd_blocked += 1
            continue

        if take_long[i]:
            side = "long"
            grade = int(long_grade[i])
            pnl_pts = float(long_pnl_pts[i])
        else:
            side = "short"
            grade = int(short_grade[i])
            pnl_pts = float(short_pnl_pts[i])

        if not open_pos:
            # First leg. A++ first-leg sizes as A+ (user spec: A++ only triggers
            # the pyramid second leg; A++ qualifies as A+ for opening sizing).
            entry_grade = min(grade, GRADE_A_PLUS)
            contracts = _size_for_grade(entry_grade, cfg, is_pyramid=False)
            is_pyramid = False
        else:
            # Pyramid eligibility check.
            if grade < GRADE_A_PLUS_PLUS:
                continue
            most_recent_entry = max(p["entry_ts"] for p in open_pos)
            if (ts - most_recent_entry) < pyramid_gap:
                continue
            if len(open_pos) >= cfg.max_stack_depth:
                continue
            # Pyramid only in the direction of the existing stack.
            if open_pos[-1]["side"] != side:
                continue
            contracts = cfg.size_a_plus_plus_pyramid
            entry_grade = GRADE_A_PLUS_PLUS
            is_pyramid = True
            n_pyramid += 1

        natural_exit_ts = ts + hold
        # Cap exit at the NY close FOLLOWING this entry (cached on the position
        # so post-close entries get the right next-day close, not a stale value).
        pos_close_cap = ny_close_for_ts if cfg.eod_force_flatten else natural_exit_ts
        exit_ts_for_log = min(natural_exit_ts, pos_close_cap)
        pos = {
            "entry_ts": ts,
            "exit_ts": natural_exit_ts,            # natural — used for concurrency check
            "effective_exit_ts": exit_ts_for_log,  # what actually gets logged
            "ny_close_cap": pos_close_cap,         # used in step 1 next iteration
            "side": side,
            "contracts": contracts,
            "grade": entry_grade,
            "pnl_pts": pnl_pts,
            "is_pyramid": is_pyramid,
        }
        open_pos.append(pos)
        trades.append(pos)
        grade_counts[_GRADE_NAME[entry_grade]] += 1
        contract_events.append((ts, +contracts))

    # Drain any positions still open at end-of-data.
    for p in open_pos:
        natural_exit = p["exit_ts"]
        effective_exit = (
            min(natural_exit, p["ny_close_cap"])
            if cfg.eod_force_flatten else natural_exit
        )
        _close_position(p, effective_exit)

    n_trades = len(trades)
    if n_trades == 0:
        return _empty_result()

    trade_df = pd.DataFrame(trades)
    trade_df["pnl_usd"] = trade_df["pnl_pts"] * trade_df["contracts"] * POINT_VALUE_USD

    # ---- Headline stats ----
    n_long = int((trade_df["side"] == "long").sum())
    n_short = int((trade_df["side"] == "short").sum())
    win_mask = trade_df["pnl_pts"] > 0
    wins = trade_df.loc[win_mask, "pnl_usd"]
    losses = trade_df.loc[~win_mask, "pnl_usd"]
    win_rate = float(win_mask.mean())
    # avg_r expressed per-contract on the SL_POINTS R-unit (size-independent).
    avg_r = float(trade_df["pnl_pts"].mean() / SL_POINTS)
    expectancy_usd = float(trade_df["pnl_usd"].mean())
    total_pnl_usd = float(trade_df["pnl_usd"].sum())
    profit_factor = (
        float(wins.sum() / abs(losses.sum()))
        if len(losses) and losses.sum() != 0
        else float("inf")
    )

    # ---- Daily aggregation (anchored on entry timestamp) ----
    daily = trade_df.set_index("entry_ts")["pnl_usd"].resample("1D").sum()
    daily_active = daily[daily != 0]
    if len(daily_active) > 1:
        mean_d = daily_active.mean()
        std_d = daily_active.std()
        sharpe = float(mean_d / std_d * np.sqrt(252)) if std_d > 0 else 0.0
        neg = daily_active[daily_active < 0]
        downside = float(neg.std()) if len(neg) > 1 else 0.0
        sortino = float(mean_d / downside * np.sqrt(252)) if downside > 0 else 0.0
    else:
        sharpe = sortino = 0.0

    equity = daily.cumsum()
    peak = equity.cummax()
    dd = peak - equity
    max_dd_usd = float(dd.max())
    max_dd_pct = max_dd_usd / cfg.starting_capital_usd

    monthly = daily.resample("1ME").sum()
    pct_pos_months = float((monthly > 0).mean()) if len(monthly) else 0.0

    n_days_traded = int((daily_active != 0).sum())
    trades_per_day = n_trades / max(n_days_traded, 1)
    avg_position_size = float(trade_df["contracts"].mean())

    # ---- Concurrency: max contracts ever simultaneously held ----
    # Sort so that CLOSES happen before OPENS at the same timestamp — matches
    # the open_pos cleanup convention (`exit_ts > ts` treats exits at the
    # current bar as already closed). With delta < 0 for closes and delta > 0
    # for opens, sorting by (ts, delta) puts negatives before positives.
    contract_events.sort(key=lambda e: (e[0], e[1]))
    running = 0
    max_concurrent = 0
    for _, delta in contract_events:
        running += delta
        max_concurrent = max(max_concurrent, running)

    # ---- TopStep telemetry (informational) ----
    worst_day = float(daily.min()) if len(daily) else 0.0
    n_trip_daily = int((daily < -cfg.topstep_daily_loss_usd).sum())
    # Trailing DD: peak-to-trough on the equity curve at end-of-day granularity.
    # Days where dd from running peak exceeds the configured trailing cap.
    n_trip_trailing = int((dd > cfg.topstep_trailing_dd_usd).sum())
    worst_intraday_dd = _worst_intraday_dd(trade_df)

    return L3Result(
        n_trades=n_trades,
        n_long=n_long,
        n_short=n_short,
        n_trades_by_grade=grade_counts,
        n_pyramid_entries=n_pyramid,
        avg_position_size=avg_position_size,
        max_concurrent_contracts=max_concurrent,
        win_rate=win_rate,
        avg_r=avg_r,
        expectancy_usd=expectancy_usd,
        total_pnl_usd=total_pnl_usd,
        sharpe=sharpe,
        sortino=sortino,
        max_drawdown_usd=max_dd_usd,
        max_drawdown_pct=max_dd_pct,
        profit_factor=profit_factor,
        trades_per_day=trades_per_day,
        monthly_pnl=monthly,
        pct_positive_months=pct_pos_months,
        equity_curve=equity,
        worst_day_pnl_usd=worst_day,
        worst_intraday_dd_usd=worst_intraday_dd,
        n_days_would_trip_daily_loss=n_trip_daily,
        n_days_would_trip_trailing_dd=n_trip_trailing,
        n_eod_entries_blocked=n_eod_blocked,
        n_eod_force_flattened=n_eod_flat,
        n_cooldown_entries_blocked=n_cd_blocked,
        max_consecutive_losses=max_consec_losses,
        trades=trade_df,
    )


def _size_for_grade(grade: int, cfg: L3Config, is_pyramid: bool) -> int:
    if is_pyramid:
        return cfg.size_a_plus_plus_pyramid
    if grade == GRADE_A_PLUS:
        return cfg.size_a_plus
    if grade == GRADE_A:
        return cfg.size_a
    if grade == GRADE_B:
        return cfg.size_b
    raise ValueError(f"unexpected first-entry grade: {grade}")


def _worst_intraday_dd(trade_df: pd.DataFrame) -> float:
    """Largest peak-to-trough drawdown within a single trading day, in USD.

    Approximation: order trades within each day by entry_ts and compute the
    running equity drawdown. Doesn't account for unrealized intra-trade
    excursion — uses realized PnL at the entry timestamp for simplicity
    (consistent with how `daily` is aggregated).
    """
    if trade_df.empty:
        return 0.0
    df = trade_df.sort_values("entry_ts").copy()
    df["day"] = df["entry_ts"].dt.normalize()
    worst = 0.0
    for _, day_df in df.groupby("day"):
        cum = day_df["pnl_usd"].cumsum().to_numpy()
        peak = np.maximum.accumulate(np.concatenate([[0.0], cum]))[1:]
        dd = peak - cum
        if dd.size:
            worst = max(worst, float(dd.max()))
    return worst


def _empty_result() -> L3Result:
    return L3Result(
        n_trades=0, n_long=0, n_short=0,
        n_trades_by_grade={"B": 0, "A": 0, "A+": 0, "A++": 0},
        n_pyramid_entries=0, avg_position_size=0.0, max_concurrent_contracts=0,
        win_rate=float("nan"), avg_r=0.0, expectancy_usd=0.0, total_pnl_usd=0.0,
        sharpe=0.0, sortino=0.0, max_drawdown_usd=0.0, max_drawdown_pct=0.0,
        profit_factor=0.0, trades_per_day=0.0,
        monthly_pnl=pd.Series(dtype=float), pct_positive_months=0.0,
        equity_curve=pd.Series(dtype=float),
        worst_day_pnl_usd=0.0, worst_intraday_dd_usd=0.0,
        n_days_would_trip_daily_loss=0, n_days_would_trip_trailing_dd=0,
        n_eod_entries_blocked=0, n_eod_force_flattened=0,
        n_cooldown_entries_blocked=0, max_consecutive_losses=0,
        trades=pd.DataFrame(),
    )
