"""Layer-1 standalone trading simulation.

We use a *deliberately dumb* trading rule so we can isolate whether Layer 1
has real edge before Layer 2 (meta-labeling) starts amplifying it. Rule:

    if   P(long_tp)  > T_enter AND P(short_tp) < T_avoid: go long
    elif P(short_tp) > T_enter AND P(long_tp)  < T_avoid: go short
    else: flat

Each entered trade is sized 1 contract, exits on TP (+30pt), SL (-30pt), or
H-bar timeout (at the close of the H-th bar). Costs (commission + slippage)
are deducted per round-trip.

Because the labeler already recorded the realized outcome for every entry
(long_tp/short_tp and the resolution-bar MFE/MAE), we can compute the PnL of
this rule directly from predictions + labels — no separate event-driven
backtester needed. This is exactly what we want for fast iteration.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from altus.config import (
    COMMISSION_RT_USD,
    POINT_VALUE_USD,
    SL_POINTS,
    SLIPPAGE_RT_POINTS,
    TP_POINTS,
)


@dataclass
class SimConfig:
    """Trading rule config. Two modes:

      * Absolute thresholds (mode='absolute'): enter if P > enter_threshold and
        the opposite-side P < avoid_threshold. Requires calibrated probabilities.
      * Percentile thresholds (mode='percentile'): trade the top `enter_percentile`%
        of signals per side by rank. Robust to miscalibration AND isotonic
        calibration ties.
    """
    mode: str = "absolute"
    enter_threshold: float = 0.55
    avoid_threshold: float = 0.50
    enter_percentile: float = 0.10   # top 10% of signals
    avoid_percentile: float = 0.50   # (legacy; unused in new rank-based logic)
    timeout_assume_close: bool = True
    # Starting capital baseline for drawdown % — TopStep Combine default.
    # Without this, drawdown % is computed against the running equity peak, which
    # produces nonsensical numbers (>1000%) for losing strategies where the peak
    # is tiny early before everything falls.
    starting_capital_usd: float = 50_000.0


@dataclass
class SimResult:
    n_trades: int
    n_long: int
    n_short: int
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

    def summary_line(self) -> str:
        return (
            f"trades={self.n_trades} ({self.n_long}L/{self.n_short}S) "
            f"| win={self.win_rate:.3f} avgR={self.avg_r:+.3f} "
            f"| PnL=${self.total_pnl_usd:,.0f} Sharpe={self.sharpe:.2f} "
            f"DD={self.max_drawdown_pct:.1%} | TPD={self.trades_per_day:.1f}"
        )


def simulate_trading(
    timestamps: np.ndarray,
    preds: dict[str, np.ndarray],
    truths: dict[str, np.ndarray],
    cfg: SimConfig | None = None,
) -> SimResult:
    """Run the dumb trading rule over the OOS samples and return PnL stats.

    Parameters
    ----------
    timestamps : 1D array of entry timestamps (np.datetime64), aligned to preds/truths.
    preds      : dict with 'long_tp_prob', 'short_tp_prob' arrays.
    truths     : dict with 'long_tp', 'short_tp', and the four MFE/MAE arrays.
    """
    cfg = cfg or SimConfig()
    n = len(timestamps)
    long_p = preds["long_tp_prob"]
    short_p = preds["short_tp_prob"]

    if cfg.mode == "absolute":
        take_long = (long_p > cfg.enter_threshold) & (short_p < cfg.avoid_threshold)
        take_short = (short_p > cfg.enter_threshold) & (long_p < cfg.avoid_threshold)
        # Conflict resolution: if both true (shouldn't happen with proper thresholds), take higher P
        both = take_long & take_short
        if both.any():
            take_long = take_long & (~both | (long_p >= short_p))
            take_short = take_short & (~both | (short_p > long_p))
    elif cfg.mode == "percentile":
        # Rank-based top-K selection (NOT quantile threshold).
        #
        # Why: after isotonic calibration, the probability vector contains many ties
        # (isotonic produces a step function — entire buckets collapse onto a single
        # calibrated value). A `prob >= np.quantile(prob, 0.99)` test then matches a
        # whole tied bucket — could be 10% of samples instead of 1%. The rank-based
        # approach takes exactly N samples by descending order, guaranteed.
        n_take = max(1, int(np.ceil(cfg.enter_percentile * n)))
        # argpartition is O(n) vs full sort O(n log n) — faster for large n
        long_top_idx = np.argpartition(-long_p, n_take)[:n_take]
        short_top_idx = np.argpartition(-short_p, n_take)[:n_take]
        long_topk = np.zeros(n, dtype=bool)
        short_topk = np.zeros(n, dtype=bool)
        long_topk[long_top_idx] = True
        short_topk[short_top_idx] = True
        # Conflict avoidance: skip bars that are in BOTH top-K (model is confused / no clean signal)
        conflict = long_topk & short_topk
        take_long = long_topk & ~conflict
        take_short = short_topk & ~conflict
    else:
        raise ValueError(f"unknown sim mode: {cfg.mode}")

    # Realized point PnL per trade:
    #  - if TP hit: +TP_POINTS
    #  - if SL hit (or timeout with adverse): -SL_POINTS or realized MFE-MAE
    # We approximate: if label==1 -> +TP_POINTS; if label==0 ->
    #   if MAE >= SL_POINTS (full stop) -> -SL_POINTS
    #   else (timeout with no stop) -> use MFE - MAE as net (proxy for unrealized at close)
    cost_pts_per_trade = SLIPPAGE_RT_POINTS + (COMMISSION_RT_USD / POINT_VALUE_USD)

    def _pnl_for_side(label, mfe, mae, tp_pts, sl_pts):
        # Default: timeout-no-stop outcomes
        pnl = (mfe - mae)
        full_stop = (label == 0) & (mae >= sl_pts)
        full_tp = (label == 1)
        pnl = np.where(full_stop, -sl_pts, pnl)
        pnl = np.where(full_tp, tp_pts, pnl)
        return pnl - cost_pts_per_trade

    long_pnl_pts = _pnl_for_side(
        truths["long_tp"], truths["mfe_long"], truths["mae_long"], TP_POINTS, SL_POINTS
    )
    short_pnl_pts = _pnl_for_side(
        truths["short_tp"], truths["mfe_short"], truths["mae_short"], TP_POINTS, SL_POINTS
    )

    trade_pnl_pts = np.zeros(n, dtype=np.float32)
    trade_pnl_pts[take_long] = long_pnl_pts[take_long]
    trade_pnl_pts[take_short] = short_pnl_pts[take_short]
    took = take_long | take_short

    trade_pnl_usd = trade_pnl_pts * POINT_VALUE_USD
    n_trades = int(took.sum())
    n_long = int(take_long.sum())
    n_short = int(take_short.sum())

    if n_trades == 0:
        return SimResult(
            n_trades=0, n_long=0, n_short=0, win_rate=float("nan"), avg_r=0.0,
            expectancy_usd=0.0, total_pnl_usd=0.0, sharpe=0.0, sortino=0.0,
            max_drawdown_usd=0.0, max_drawdown_pct=0.0, profit_factor=0.0,
            trades_per_day=0.0, monthly_pnl=pd.Series(dtype=float),
            pct_positive_months=0.0, equity_curve=pd.Series(dtype=float),
        )

    win_mask = trade_pnl_pts > 0
    wins = trade_pnl_usd[took & win_mask]
    losses = trade_pnl_usd[took & ~win_mask]
    win_rate = float(took[win_mask].sum() / n_trades) if n_trades else 0.0
    avg_r = float(trade_pnl_pts[took].mean() / SL_POINTS)
    expectancy_usd = float(trade_pnl_usd[took].mean())
    total_pnl_usd = float(trade_pnl_usd[took].sum())
    profit_factor = float(wins.sum() / abs(losses.sum())) if losses.size and losses.sum() != 0 else float("inf")

    # Daily PnL → Sharpe/Sortino. Anchor on the trade timestamps so sparse days don't break it.
    ts = pd.DatetimeIndex(timestamps)
    daily = pd.Series(trade_pnl_usd, index=ts).resample("1D").sum()
    daily_active = daily[daily != 0]
    if len(daily_active) > 1:
        mean_d = daily_active.mean()
        std_d = daily_active.std()
        sharpe = float(mean_d / std_d * np.sqrt(252)) if std_d > 0 else 0.0
        neg = daily_active[daily_active < 0]
        downside = float(neg.std()) if len(neg) > 1 else 0.0
        sortino = float(mean_d / downside * np.sqrt(252)) if downside > 0 else 0.0
    else:
        sharpe = 0.0
        sortino = 0.0

    # Equity curve & drawdown.
    # DD% is computed against starting capital (not running peak) — the running-peak
    # version produces nonsensical numbers for losing strategies (peak is tiny if
    # reached early then everything falls, so dd/peak explodes past 100%).
    equity = daily.cumsum()
    peak = equity.cummax()
    dd = peak - equity
    max_dd_usd = float(dd.max())
    max_dd_pct = max_dd_usd / cfg.starting_capital_usd

    # Monthly stability
    monthly = daily.resample("1ME").sum()
    pct_pos_months = float((monthly > 0).mean()) if len(monthly) else 0.0

    # Trades per day (only days we traded at all)
    n_days_traded = int((daily_active != 0).sum())
    trades_per_day = n_trades / max(n_days_traded, 1)

    return SimResult(
        n_trades=n_trades,
        n_long=n_long,
        n_short=n_short,
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
    )


def sweep_thresholds(
    timestamps: np.ndarray,
    preds: dict[str, np.ndarray],
    truths: dict[str, np.ndarray],
    enter_grid: np.ndarray | None = None,
    avoid_grid: np.ndarray | None = None,
) -> pd.DataFrame:
    """Grid-sweep enter/avoid thresholds and return PnL summary per cell."""
    if enter_grid is None:
        enter_grid = np.arange(0.50, 0.71, 0.02)
    if avoid_grid is None:
        avoid_grid = np.arange(0.30, 0.51, 0.05)
    rows = []
    for et in enter_grid:
        for at in avoid_grid:
            res = simulate_trading(
                timestamps, preds, truths,
                cfg=SimConfig(enter_threshold=float(et), avoid_threshold=float(at)),
            )
            rows.append({
                "enter_threshold": et,
                "avoid_threshold": at,
                "n_trades": res.n_trades,
                "trades_per_day": res.trades_per_day,
                "win_rate": res.win_rate,
                "expectancy_usd": res.expectancy_usd,
                "total_pnl_usd": res.total_pnl_usd,
                "sharpe": res.sharpe,
                "max_dd_pct": res.max_drawdown_pct,
            })
    return pd.DataFrame(rows)
