"""Asymmetric R:R sweep on saved val_preds.

For each of the 4 trained TCN variants, recompute trade outcomes + PnL under
multiple (TP_points, SL_points) configurations using the stored MFE/MAE
arrays. This tells us whether the current L1 quality is profitable under
asymmetric R:R *without* retraining anything.

Why this is potentially answer-changing:
  At current top-1% WR 0.508 with 30/30 R:R the math is breakeven-to-losing.
  Same WR under 40/20 R:R (2:1) gives ~$10/trade gross profit. If the model's
  ranking holds when we change barriers (it largely does — MFE/MAE preserve
  the underlying move character), asymmetric R:R can make a "barely losing"
  system profitable WITHOUT needing better ML.

Methodology:
  - MFE/MAE are max favorable/adverse excursion over the full label horizon.
  - For each candidate trade, recompute outcome under new (TP, SL):
      * win if MFE >= TP AND MAE < SL     → both barriers permit TP first
      * loss if MFE < TP AND MAE >= SL     → SL hit before TP could
      * timeout if both excursions stayed inside both barriers
      * ambiguous if both MFE >= TP AND MAE >= SL → use conservative
        assumption: assume SL hit first (loss). Pessimistic but safe.

  Then rank trades by model probability, take top-K%, compute PnL:
      PnL_per_trade = wins × TP - losses × SL - cost × total_trades

  Cost = COMMISSION_RT_USD + SLIPPAGE_RT_POINTS × POINT_VALUE_USD per trade.

Run:
    python3 scripts/eval_rr_asymmetry.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from altus.config import COMMISSION_RT_USD, POINT_VALUE_USD, SLIPPAGE_RT_POINTS


ARTIFACTS = Path(__file__).resolve().parents[1] / "artifacts" / "tcn_runs"

VARIANTS = [
    ("01_baseline", "cloud_full_vol+trend+anomaly_20260524_105013"),
    ("02_phaseE",   "cloud_full_vol+trend+anomaly+round+mtf+absorp+pvd+extension+vreg+sanat+creg+lasym+rhythm+facc+surprise_20260524_111257"),
    ("03_phaseF",   "cloud_full_vol+trend+anomaly+bocpd_20260524_114401"),
    ("04_phaseEF",  "cloud_full_vol+trend+anomaly+round+mtf+absorp+pvd+extension+vreg+sanat+creg+lasym+rhythm+facc+surprise+bocpd_20260524_123154"),
]

RR_CONFIGS = [
    # (label, TP_points, SL_points)
    ("30/30",  30.0, 30.0),    # baseline (current)
    ("25/30",  25.0, 30.0),    # slight defensive bias
    ("20/30",  20.0, 30.0),    # tight take-profit
    ("40/20",  40.0, 20.0),    # 2:1 asymmetric
    ("40/30",  40.0, 30.0),    # 1.33:1
    ("50/30",  50.0, 30.0),    # 1.67:1
    ("60/20",  60.0, 20.0),    # 3:1 aggressive
    ("60/30",  60.0, 30.0),    # 2:1 wide
]

PERCENTILES = [0.01, 0.05, 0.10, 0.20]

# OOS lockbox length in trading days (for TPD computation)
OOS_TRADING_DAYS = 80  # ~4 months × 20 trading days/month


def recompute_outcomes(mfe: np.ndarray, mae: np.ndarray, tp: float, sl: float) -> tuple[np.ndarray, np.ndarray]:
    """Recompute (win, loss) bool arrays under new (TP, SL) thresholds.

    Returns (is_win, is_loss). The remaining trades (neither win nor loss) are
    timeouts and contribute zero PnL (we close at entry — slightly optimistic).
    Ambiguous cases (both barriers triggered) are assigned to LOSS as a
    conservative safety assumption.
    """
    reached_tp = mfe >= tp
    reached_sl = mae >= sl
    is_win = reached_tp & ~reached_sl
    is_loss_clean = reached_sl & ~reached_tp
    is_ambiguous = reached_tp & reached_sl
    # Conservative: ambiguous = loss
    is_loss = is_loss_clean | is_ambiguous
    return is_win, is_loss


def pnl_per_trade(is_win: np.ndarray, is_loss: np.ndarray, tp: float, sl: float, cost_per_trade: float) -> float:
    """Average PnL per trade in USD."""
    n = len(is_win)
    if n == 0:
        return 0.0
    gross = is_win.astype(np.float64).sum() * tp - is_loss.astype(np.float64).sum() * sl
    gross_usd = gross * POINT_VALUE_USD
    total_cost = n * cost_per_trade
    return float((gross_usd - total_cost) / n)


def load_variant(dir_path: Path) -> dict:
    """Load and concatenate all 3 folds' val preds + truths."""
    folds = []
    for f in range(3):
        p = dir_path / f"tcn_fold{f}_val_preds.npz"
        if not p.exists():
            return None
        d = dict(np.load(p))
        folds.append(d)
    # Only concatenate keys present in ALL folds (skip fusion_embedding if some
    # folds have been re-extracted with embeddings and others haven't).
    common_keys = set(folds[0].keys())
    for f in folds[1:]:
        common_keys &= set(f.keys())
    out = {}
    for k in common_keys:
        if folds[0][k].ndim == 0:  # scalar like 'fold'
            continue
        try:
            out[k] = np.concatenate([f[k] for f in folds])
        except ValueError:
            # Shape mismatch (e.g., embeddings present in some folds only) — skip
            continue
    return out


def main():
    cost = COMMISSION_RT_USD + SLIPPAGE_RT_POINTS * POINT_VALUE_USD
    print(f"Cost per trade: ${cost:.2f}  (commission ${COMMISSION_RT_USD:.2f} + slippage {SLIPPAGE_RT_POINTS}pt × ${POINT_VALUE_USD}/pt)")
    print()

    for label, dirname in VARIANTS:
        dir_path = ARTIFACTS / dirname
        if not dir_path.exists():
            print(f"  [{label}] MISSING: {dir_path}")
            continue
        data = load_variant(dir_path)
        if data is None:
            print(f"  [{label}] INCOMPLETE: not all 3 folds present")
            continue

        p_long = data["val_preds_long_tp_prob"]
        p_short = data["val_preds_short_tp_prob"]
        mfe_long = data["val_truths_mfe_long"]
        mae_long = data["val_truths_mae_long"]
        mfe_short = data["val_truths_mfe_short"]
        mae_short = data["val_truths_mae_short"]
        n_samples = len(p_long)

        print(f"\n{'=' * 100}")
        print(f"VARIANT: {label}   (n={n_samples:,} val samples across 3 folds)")
        print(f"{'=' * 100}")
        print(f"{'R:R':<10} {'Pct':<6} {'#trades':<9} {'#L/#S':<13} {'WR':<8} {'PnL/trade':<11} {'Total PnL':<14} {'TPD':<6}")
        print("-" * 100)

        for rr_label, tp, sl in RR_CONFIGS:
            long_win, long_loss = recompute_outcomes(mfe_long, mae_long, tp, sl)
            short_win, short_loss = recompute_outcomes(mfe_short, mae_short, tp, sl)

            for pct in PERCENTILES:
                k = int(np.ceil(n_samples * pct))
                if k < 10:
                    continue

                # Pick the top-K long-side and top-K short-side candidates by prob
                top_long_idx = np.argpartition(p_long, -k)[-k:]
                top_short_idx = np.argpartition(p_short, -k)[-k:]

                lw = long_win[top_long_idx].sum()
                ll = long_loss[top_long_idx].sum()
                sw = short_win[top_short_idx].sum()
                sl_count = short_loss[top_short_idx].sum()

                total_trades = 2 * k
                total_wins = lw + sw
                total_losses = ll + sl_count
                wr = total_wins / max(total_trades, 1)

                gross_pnl_usd = (total_wins * tp - total_losses * sl) * POINT_VALUE_USD
                total_pnl = gross_pnl_usd - total_trades * cost
                ppt = total_pnl / max(total_trades, 1)
                tpd = total_trades / OOS_TRADING_DAYS

                print(
                    f"{rr_label:<10} {pct*100:<5.0f}% "
                    f"{total_trades:<9,} "
                    f"{int(lw)}W/{int(ll)}L / {int(sw)}W/{int(sl_count)}L".ljust(14)
                    + f" {wr:<8.3f} "
                    + f"${ppt:<+10.2f} "
                    + f"${total_pnl:<+12,.0f} "
                    + f"{tpd:<6.1f}"
                )
            print()

    print()
    print(f"NOTES:")
    print(f"  - Ambiguous cases (MFE >= TP AND MAE >= SL) counted as LOSS — pessimistic estimate.")
    print(f"  - Cost ${cost:.2f}/trade includes commission + ${SLIPPAGE_RT_POINTS}pt slippage.")
    print(f"  - TPD = trades per (assumed {OOS_TRADING_DAYS}) OOS trading days.")
    print(f"  - 'Total PnL' is over the 4-month OOS lockbox. Annualize × ~3 for ballpark yearly.")


if __name__ == "__main__":
    main()
