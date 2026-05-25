"""Smoke test for L3.1 production sim.

Loads a cached L1 prediction fold, runs both the baseline per-bar sim
(altus.training.sim_pnl) and the L3.1 production sim
(altus.training.production_sim) on the same predictions, and prints a
side-by-side comparison.

What we expect to see:
  * TPD drops sharply (5-10x lower) under L3.1 because clustered top-pct
    signals collapse into single trades.
  * Total PnL changes — magnitude depends on grade distribution × sizing;
    direction depends on whether stacking was over- or under-stating PnL.
  * Max concurrent contracts ≤ MAX_STACK_DEPTH (3 by default).
  * Pyramid entries are a small fraction of trades (A++ is top 0.5% of an
    already-restricted set).

Usage:
    python3 scripts/smoke_test_l3.py \\
        --preds artifacts/tcn_runs/cloud_full_vol+trend+anomaly_20260524_105013/tcn_fold0_val_preds.npz
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from altus.training.production_sim import L3Config, simulate_l3
from altus.training.sim_pnl import SimConfig, simulate_trading


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--preds", required=True, help=".npz file from train_cloud.py")
    ap.add_argument("--bar-minutes", type=int, default=1,
                    help="Bar period for synthetic timestamps (default 1m)")
    args = ap.parse_args()

    d = np.load(args.preds, allow_pickle=True)
    preds = {
        "long_tp_prob": d["val_preds_long_tp_prob"],
        "short_tp_prob": d["val_preds_short_tp_prob"],
    }
    truths = {
        "long_tp": d["val_truths_long_tp"],
        "short_tp": d["val_truths_short_tp"],
        "mfe_long": d["val_truths_mfe_long"],
        "mae_long": d["val_truths_mae_long"],
        "mfe_short": d["val_truths_mfe_short"],
        "mae_short": d["val_truths_mae_short"],
    }
    positions = d["val_positions"]

    # Synthesize timestamps: each position = N bar-minutes from an arbitrary
    # epoch. Exact dates don't matter; only relative spacing matters for
    # concurrency and the 5-min pyramid rule.
    base = pd.Timestamp("2024-01-01")
    timestamps = (base + pd.to_timedelta(positions * args.bar_minutes, unit="m")).values

    n = len(timestamps)
    n_days = (positions.max() - positions.min()) * args.bar_minutes / (60 * 24)
    print(f"Loaded {n:,} val samples spanning ~{n_days:.0f} bar-days "
          f"(positions {positions.min()}-{positions.max()})")
    print()

    # ---- Baseline: per-bar sim, top-1% percentile (sim_pnl default-ish) ----
    print("=" * 78)
    print("BASELINE (sim_pnl, percentile top-1%, no concurrency, 1 contract):")
    print("=" * 78)
    baseline_cfg = SimConfig(mode="percentile", enter_percentile=0.01)
    baseline = simulate_trading(timestamps, preds, truths, cfg=baseline_cfg)
    print(baseline.summary_line())
    print()

    # ---- L3.1: grade-based sizing + no-overlap + A++ pyramiding ----
    print("=" * 78)
    print("L3.1 (production_sim, grade-sized, no-overlap + A++ pyramid):")
    print("=" * 78)
    l3_cfg = L3Config()
    l3 = simulate_l3(timestamps, preds, truths, cfg=l3_cfg)
    print(l3.summary_line())
    print()
    print(f"  Grade thresholds: B={l3_cfg.grades.b_pct:.1%}  A={l3_cfg.grades.a_pct:.1%}  "
          f"A+={l3_cfg.grades.a_plus_pct:.1%}  A++={l3_cfg.grades.a_plus_plus_pct:.2%}")
    print(f"  Sizing: B={l3_cfg.size_b}  A={l3_cfg.size_a}  A+={l3_cfg.size_a_plus}  "
          f"A++pyr={l3_cfg.size_a_plus_plus_pyramid}  max_stack={l3_cfg.max_stack_depth}")
    print()
    print("  TopStep telemetry (informational):")
    print(f"    worst_day_pnl = ${l3.worst_day_pnl_usd:,.0f}")
    print(f"    worst_intraday_dd = ${l3.worst_intraday_dd_usd:,.0f}")
    print(f"    days that would have tripped ${l3_cfg.topstep_daily_loss_usd:,.0f} daily-loss: "
          f"{l3.n_days_would_trip_daily_loss}")
    print(f"    days that would have tripped ${l3_cfg.topstep_trailing_dd_usd:,.0f} trailing-DD: "
          f"{l3.n_days_would_trip_trailing_dd}")
    print()
    print("  Hard-rule diagnostics:")
    print(f"    EoD entries blocked (within {l3_cfg.eod_no_entry_min}min of NY close): "
          f"{l3.n_eod_entries_blocked}")
    print(f"    EoD forced-flatten closures: {l3.n_eod_force_flattened}")
    print(f"    cooldown entries blocked (consec-loss schedule {l3_cfg.cooldown_schedule}): "
          f"{l3.n_cooldown_entries_blocked}")
    print(f"    max consecutive losses observed: {l3.max_consecutive_losses}")
    print()

    # ---- Side-by-side ----
    print("=" * 78)
    print("DELTA (L3.1 vs baseline):")
    print("=" * 78)
    print(f"  trades:     {baseline.n_trades:>6}  ->  {l3.n_trades:>6}  "
          f"({l3.n_trades / max(baseline.n_trades,1):.2f}x)")
    print(f"  trades/day: {baseline.trades_per_day:>6.1f}  ->  {l3.trades_per_day:>6.1f}  "
          f"({l3.trades_per_day / max(baseline.trades_per_day, 1e-9):.2f}x)")
    print(f"  total PnL:  ${baseline.total_pnl_usd:>10,.0f}  ->  ${l3.total_pnl_usd:>10,.0f}")
    print(f"  Sharpe:     {baseline.sharpe:>6.2f}  ->  {l3.sharpe:>6.2f}")
    print(f"  max DD %:   {baseline.max_drawdown_pct:>6.1%}  ->  {l3.max_drawdown_pct:>6.1%}")


if __name__ == "__main__":
    main()
