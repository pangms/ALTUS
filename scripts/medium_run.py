"""Medium-scale training run for ALTUS Layer 1.

Goal: get honest first numbers against the acceptance criteria.

  * 1 year of MNQ data (configurable below)
  * Full model size: d_model=96, 3 blocks per branch, seq_len=240
  * Single walk-forward fold inside dev + small OOS lockbox at tail
  * Both variants (Mamba and xLSTM)
  * Post-hoc isotonic calibration fit on a held-out tail of train
  * Percentile-threshold trading sim at K in {1%, 5%, 10%, 20%}
  * Artifacts saved to artifacts/medium_<variant>_<timestamp>/

Target wall time: 2-4 hours on MPS. Headless-friendly output (no tqdm spam).
"""
from __future__ import annotations

import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Force line-buffered output so a tail -f on the logfile shows progress live.
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

from altus.config import ARTIFACT_DIR, AcceptanceCriteria, TrainConfig
from altus.data import load_mnq
from altus.features import build_features
from altus.labels import filter_labels_to_index, triple_barrier_labels
from altus.models.hybrid import build_hybrid
from altus.splits import purged_walk_forward
from altus.training import evaluate_predictions, train_model
from altus.training.calibration import calibrate_predictions
from altus.training.dataset import ALTUSDataset
from altus.training.sim_pnl import SimConfig, simulate_trading


# Configurable run parameters.
# These are calibrated to fit 9GB MPS memory AND finish in 2-4 hours.
# The pure-PyTorch selective scan stores (B, L, D_inner, N) intermediates for
# backward, so memory scales as B*L*(2*D_model)*d_state*n_blocks. Wall time is
# dominated by the Python-loop scan; reducing batch size doesn't help speed.
RUN_START = "2024-04-01"
RUN_END = "2025-04-01"           # 1 year
SEQ_LEN = 120                    # 2hr of 1m context — plenty for short-TF specialist
D_MODEL = 64
TCN_BLOCKS = 2
MAMBA_BLOCKS = 2
XLSTM_BLOCKS = 2
BATCH_SIZE = 128
EPOCHS = 8
EARLY_STOP_PATIENCE = 3
LEARNING_RATE = 1e-3
OOS_MONTHS = 1                   # small OOS for medium run; full run uses 6
FOLDS = 1                        # single fold for medium run; full uses 5
CAL_HOLDOUT_FRAC = 0.10          # last 10% of train held out for calibration fit


def banner(msg: str) -> None:
    print("\n" + "=" * 78 + f"\n{msg}\n" + "=" * 78, flush=True)


def _truths_at(labels, positions: np.ndarray) -> dict:
    return {
        "long_tp": labels.long_tp[positions].astype(np.int8),
        "short_tp": labels.short_tp[positions].astype(np.int8),
        "mfe_long": labels.mfe_long[positions],
        "mae_long": labels.mae_long[positions],
        "mfe_short": labels.mfe_short[positions],
        "mae_short": labels.mae_short[positions],
    }


def _run_percentile_sweep(timestamps, preds, truths, ks=(0.01, 0.05, 0.10, 0.20)) -> dict:
    out = {}
    for k in ks:
        cfg = SimConfig(mode="percentile", enter_percentile=k, avoid_percentile=0.5)
        sim = simulate_trading(timestamps, preds, truths, cfg=cfg)
        out[f"top{int(k*100)}pct"] = {
            "n_trades": sim.n_trades,
            "win_rate": sim.win_rate,
            "trades_per_day": sim.trades_per_day,
            "expectancy_usd": sim.expectancy_usd,
            "total_pnl_usd": sim.total_pnl_usd,
            "sharpe": sim.sharpe,
            "sortino": sim.sortino,
            "max_drawdown_pct": sim.max_drawdown_pct,
            "profit_factor": sim.profit_factor,
            "pct_positive_months": sim.pct_positive_months,
        }
        print(f"    K={int(k*100):>3}%: {sim.summary_line()}")
    return out


def _serialize_metrics(m) -> dict:
    return {
        "n": m.n,
        "base_rate": m.base_rate,
        "auc": m.auc,
        "pr_auc": m.pr_auc,
        "brier": m.brier,
        "brier_baseline": m.brier_baseline,
        "brier_improvement": m.brier_improvement,
        "ic": m.ic,
        "top_decile_winrate": m.top_decile_winrate,
        "top_5pct_winrate": m.top_5pct_winrate,
        "top_1pct_winrate": m.top_1pct_winrate,
        "mfe_rmse": m.mfe_rmse,
        "mae_rmse": m.mae_rmse,
        "mean_auc": m.mean_auc(),
    }


def main():
    t0 = time.time()
    run_id = time.strftime("%Y%m%d_%H%M%S")
    print(f"medium run id: {run_id}", flush=True)

    banner("1. Loading MNQ")
    df = load_mnq(start=RUN_START, end=RUN_END)
    print(f"loaded MNQ: {len(df):,} bars, {df.index[0]} -> {df.index[-1]}")

    banner("2. Features")
    feats = build_features(df)
    n_feat = feats.shape[1]
    print(f"features: {feats.shape[0]:,} rows × {n_feat} cols")

    banner("3. Labels")
    labels = triple_barrier_labels(df)
    labels = filter_labels_to_index(labels, feats.index)
    print(f"labels: {len(labels.index):,} samples")
    print(f"  base rate long_tp:  {labels.long_tp.mean():.3f}")
    print(f"  base rate short_tp: {labels.short_tp.mean():.3f}")

    banner("4. Splits")
    splits = purged_walk_forward(labels.index, n_folds=FOLDS, oos_months=OOS_MONTHS)
    fold = splits.folds[0]
    print(f"fold 0: train={len(fold.train_idx):,}, val={len(fold.val_idx):,}, "
          f"oos={len(splits.oos_idx):,}, embargo={splits.embargo_bars}")

    # Carve a calibration-fit slice from the tail of the train set.
    n_cal = int(len(fold.train_idx) * CAL_HOLDOUT_FRAC)
    cal_fit_idx = fold.train_idx[-n_cal:]
    train_idx_minus_cal = fold.train_idx[:-n_cal]
    print(f"calibration-fit slice: last {n_cal:,} samples of train")

    artifacts_dir = ARTIFACT_DIR / f"medium_{run_id}"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    all_results = {}
    for variant in ("mamba", "xlstm"):
        banner(f"5. Training: ModernTCN + {variant.upper()}")
        v_t0 = time.time()

        train_ds = ALTUSDataset(feats, labels, train_idx_minus_cal, seq_len=SEQ_LEN)
        cal_fit_ds = ALTUSDataset(feats, labels, cal_fit_idx, seq_len=SEQ_LEN)
        val_ds = ALTUSDataset(feats, labels, fold.val_idx, seq_len=SEQ_LEN)
        oos_ds = ALTUSDataset(feats, labels, splits.oos_idx, seq_len=SEQ_LEN) if len(splits.oos_idx) else None
        print(f"  train: {len(train_ds):,} | cal-fit: {len(cal_fit_ds):,} | "
              f"val: {len(val_ds):,} | oos: {0 if oos_ds is None else len(oos_ds):,}")

        model = build_hybrid(
            n_features=n_feat,
            long_context=variant,
            d_model=D_MODEL,
            seq_len=SEQ_LEN,
            tcn_n_blocks=TCN_BLOCKS,
            mamba_n_blocks=MAMBA_BLOCKS,
            xlstm_n_blocks=XLSTM_BLOCKS,
        )
        n_params = sum(p.numel() for p in model.parameters())
        print(f"  model params: {n_params:,}")

        cfg = TrainConfig(
            batch_size=BATCH_SIZE,
            n_epochs=EPOCHS,
            lr=LEARNING_RATE,
            early_stop_patience=EARLY_STOP_PATIENCE,
        )
        result = train_model(model, train_ds, val_ds, cfg=cfg, verbose=True, show_progress=False)
        print(f"  trained in {(time.time() - v_t0)/60:.1f} min, best epoch {result.best_epoch}, "
              f"best mean AUC {result.best_val_metric:.4f}")

        # Save checkpoint
        ckpt_path = artifacts_dir / f"{variant}_best.pt"
        torch.save(model.state_dict(), ckpt_path)
        print(f"  saved checkpoint: {ckpt_path}")

        # Get predictions on cal-fit slice for calibration
        from torch.utils.data import DataLoader
        from altus.training.dataset import collate
        from altus.training.train import _predict, _select_device
        device = _select_device(cfg.device)
        cal_loader = DataLoader(cal_fit_ds, batch_size=cfg.batch_size, shuffle=False, collate_fn=collate)
        cal_preds, cal_truths = _predict(model, cal_loader, device)

        # Apply calibration to val and OOS
        val_preds_cal = calibrate_predictions(cal_preds, cal_truths, result.val_preds, method="isotonic")
        val_metrics_cal = evaluate_predictions(val_preds_cal, result.val_truths)

        oos_preds_raw = None
        oos_preds_cal = None
        oos_metrics_raw = None
        oos_metrics_cal = None
        oos_truths = None
        oos_kept_positions = None
        if oos_ds is not None and len(oos_ds) > 0:
            oos_loader = DataLoader(oos_ds, batch_size=cfg.batch_size, shuffle=False, collate_fn=collate)
            oos_preds_raw, oos_truths = _predict(model, oos_loader, device)
            oos_metrics_raw = evaluate_predictions(oos_preds_raw, oos_truths)
            oos_preds_cal = calibrate_predictions(cal_preds, cal_truths, oos_preds_raw, method="isotonic")
            oos_metrics_cal = evaluate_predictions(oos_preds_cal, oos_truths)
            oos_kept_positions = oos_ds.sample_positions

        banner(f"6. Eval: {variant.upper()}")
        print(f"  VAL  raw : {result.val_metrics.summary_line()}")
        print(f"  VAL  cal : {val_metrics_cal.summary_line()}")
        if oos_metrics_raw is not None:
            print(f"  OOS  raw : {oos_metrics_raw.summary_line()}")
            print(f"  OOS  cal : {oos_metrics_cal.summary_line()}")

        # Trading sim on calibrated VAL
        val_kept_pos = val_ds.sample_positions
        val_ts = labels.index.to_numpy()[val_kept_pos]
        val_truths_for_sim = _truths_at(labels, val_kept_pos)
        print("  VAL percentile sweep (calibrated):")
        val_sweep = _run_percentile_sweep(val_ts, val_preds_cal, val_truths_for_sim)

        oos_sweep = None
        if oos_metrics_cal is not None and oos_kept_positions is not None:
            oos_ts = labels.index.to_numpy()[oos_kept_positions]
            oos_truths_for_sim = _truths_at(labels, oos_kept_positions)
            print("  OOS percentile sweep (calibrated):")
            oos_sweep = _run_percentile_sweep(oos_ts, oos_preds_cal, oos_truths_for_sim)

        all_results[variant] = {
            "best_epoch": result.best_epoch,
            "training_minutes": (time.time() - v_t0) / 60,
            "history": result.history,
            "val_raw": _serialize_metrics(result.val_metrics),
            "val_cal": _serialize_metrics(val_metrics_cal),
            "oos_raw": _serialize_metrics(oos_metrics_raw) if oos_metrics_raw else None,
            "oos_cal": _serialize_metrics(oos_metrics_cal) if oos_metrics_cal else None,
            "val_sim_sweep": val_sweep,
            "oos_sim_sweep": oos_sweep,
            "n_params": n_params,
        }

    banner("FINAL SUMMARY")
    crit = AcceptanceCriteria()
    print(f"acceptance criteria: AUC>={crit.min_auc_per_side}, "
          f"topDecileWR>={crit.top_decile_min_winrate}, "
          f"Sharpe>={crit.min_sharpe}, maxDD<{crit.max_drawdown_pct:.0%}, "
          f"+months>={crit.min_pct_positive_months:.0%}")
    for variant, r in all_results.items():
        m = r["oos_cal"] or r["val_cal"]
        if m is None:
            continue
        which = "OOS" if r["oos_cal"] else "VAL"
        print(f"\n{variant.upper()} ({which}, calibrated):")
        print(f"  AUC long={m['auc'].get('long_tp', float('nan')):.4f}  "
              f"short={m['auc'].get('short_tp', float('nan')):.4f}  "
              f"mean={m['mean_auc']:.4f}")
        print(f"  Brier improvement: long={m['brier_improvement'].get('long_tp', float('nan')):+.3f}  "
              f"short={m['brier_improvement'].get('short_tp', float('nan')):+.3f}")
        print(f"  Top-decile win rate: long={m['top_decile_winrate'].get('long_tp', float('nan')):.3f}  "
              f"short={m['top_decile_winrate'].get('short_tp', float('nan')):.3f}")
        # Pick the K=5% sim row as representative
        sweep = r["oos_sim_sweep"] or r["val_sim_sweep"]
        if sweep:
            for k_label in ("top1pct", "top5pct", "top10pct", "top20pct"):
                s = sweep[k_label]
                print(f"  sim {k_label:>9}: trades={s['n_trades']:,} "
                      f"win={s['win_rate']:.3f}  PnL=${s['total_pnl_usd']:,.0f}  "
                      f"Sharpe={s['sharpe']:.2f}  DD={s['max_drawdown_pct']:.1%}  "
                      f"TPD={s['trades_per_day']:.1f}")

    # Save full metrics JSON
    metrics_path = artifacts_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nmetrics saved: {metrics_path}")
    print(f"checkpoints in: {artifacts_dir}")
    print(f"\nDONE in {(time.time() - t0)/60:.1f} min")


if __name__ == "__main__":
    main()
