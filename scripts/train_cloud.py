"""Cloud-GPU training run for ALTUS Layer 1 — full ambitious configs.

Auto-detects CUDA and scales up. On a 4090 / A6000 / A100 this finishes in
30-90 minutes for the 1-year run, or 2-4 hours for the full 5-year run.

Two stages:
  STAGE 1 (default): 1 year of MNQ, single fold + 1 month OOS. Quick sanity
  pass at full model size. ~30-60 min on RTX 4090.

  STAGE 2: pass `--full` for 5 years, 5-fold walk-forward, 6 month OOS. The
  real acceptance test. ~2-4 hours on RTX 4090.

Headless-friendly: no tqdm spam, line-buffered stdout, all artifacts saved
to artifacts/cloud_<timestamp>/.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
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
from altus.training.dataset import ALTUSDataset, collate
from altus.training.sim_pnl import SimConfig, simulate_trading
from altus.training.train import _predict, _select_device


# Full configs — sized for CUDA. The sequential scan is fast on CUDA so we
# can afford the original ambitious model.
FULL_CFG = dict(
    seq_len=240,
    d_model=96,
    tcn_n_blocks=3,
    mamba_n_blocks=3,
    xlstm_n_blocks=3,
    batch_size=256,
    epochs=20,
    early_stop_patience=4,
    lr=1e-3,
    cal_holdout_frac=0.10,
)


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


def _serialize_metrics(m) -> dict | None:
    if m is None:
        return None
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", action="store_true",
                        help="Full 5yr data + 5-fold walk-forward + 6mo OOS (vs 1yr/1fold/1mo default)")
    parser.add_argument("--variants", default="tcn",
                        help="Comma-separated long-context branches to train: tcn | mamba | xlstm.\n"
                             "  tcn   = TCN-only (no long-context branch; fully parallel, fast on CUDA)\n"
                             "  mamba = TCN + selective SSM (slow without mamba-ssm package)\n"
                             "  xlstm = TCN + xLSTM (slow without optimized kernels)\n"
                             "Default is 'tcn' for fast iteration; use 'mamba,xlstm' once kernels are wired up.")
    args = parser.parse_args()

    t0 = time.time()
    run_id = time.strftime("%Y%m%d_%H%M%S")

    # ----- Device check ----------------------------------------------------
    banner("0. Device check")
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"CUDA: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("WARNING: running on MPS. Full configs will be slow (~hours).")
        print("         This script is designed for CUDA. Consider scripts/medium_run.py for MPS.")
    else:
        device = torch.device("cpu")
        print("WARNING: no GPU detected — running on CPU. This will take many hours.")

    # ----- Data range + split config --------------------------------------
    if args.full:
        run_start, run_end = "2021-01-01", "2026-03-31"  # full history
        n_folds, oos_months = 5, 6
        run_tag = "full"
    else:
        run_start, run_end = "2024-04-01", "2025-04-01"  # 1 year
        n_folds, oos_months = 1, 1
        run_tag = "quick"

    print(f"\nrun_id={run_id}  tag={run_tag}  data=[{run_start}, {run_end}]  "
          f"folds={n_folds}  oos_months={oos_months}")
    print(f"variants: {args.variants}")

    artifacts_dir = ARTIFACT_DIR / f"cloud_{run_tag}_{run_id}"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    print(f"artifacts: {artifacts_dir}")

    # ----- Load + features + labels ---------------------------------------
    banner("1. Loading MNQ")
    df = load_mnq(start=run_start, end=run_end)
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
    splits = purged_walk_forward(labels.index, n_folds=n_folds, oos_months=oos_months)
    for f in splits.folds:
        print(f"  fold {f.fold}: train={len(f.train_idx):,}  val={len(f.val_idx):,}")
    print(f"  oos lockbox: {len(splits.oos_idx):,}  embargo={splits.embargo_bars}")

    # ----- Train each variant on each fold --------------------------------
    all_results = {}
    for variant in args.variants.split(","):
        variant = variant.strip()
        if variant not in ("mamba", "xlstm", "tcn"):
            print(f"skipping unknown variant: {variant}")
            continue
        # 'tcn' selects the no-long-context (TCN-only) build in HybridLayer1
        long_context_for_model = "none" if variant == "tcn" else variant

        all_results[variant] = {"folds": []}
        for fold_idx, fold in enumerate(splits.folds):
            banner(f"5.{fold_idx}. Training {variant.upper()} — fold {fold.fold}")
            f_t0 = time.time()

            # Calibration-fit slice from tail of this fold's train.
            n_cal = int(len(fold.train_idx) * FULL_CFG["cal_holdout_frac"])
            cal_fit_idx = fold.train_idx[-n_cal:]
            train_idx_minus_cal = fold.train_idx[:-n_cal]

            train_ds = ALTUSDataset(feats, labels, train_idx_minus_cal, seq_len=FULL_CFG["seq_len"])
            cal_fit_ds = ALTUSDataset(feats, labels, cal_fit_idx, seq_len=FULL_CFG["seq_len"])
            val_ds = ALTUSDataset(feats, labels, fold.val_idx, seq_len=FULL_CFG["seq_len"])
            print(f"  train: {len(train_ds):,} | cal-fit: {len(cal_fit_ds):,} | val: {len(val_ds):,}")

            model = build_hybrid(
                n_features=n_feat,
                long_context=long_context_for_model,
                d_model=FULL_CFG["d_model"],
                seq_len=FULL_CFG["seq_len"],
                tcn_n_blocks=FULL_CFG["tcn_n_blocks"],
                mamba_n_blocks=FULL_CFG["mamba_n_blocks"],
                xlstm_n_blocks=FULL_CFG["xlstm_n_blocks"],
            )
            n_params = sum(p.numel() for p in model.parameters())
            print(f"  model params: {n_params:,}")

            cfg = TrainConfig(
                batch_size=FULL_CFG["batch_size"],
                n_epochs=FULL_CFG["epochs"],
                lr=FULL_CFG["lr"],
                early_stop_patience=FULL_CFG["early_stop_patience"],
                device=str(device).split(":")[0],
            )
            result = train_model(model, train_ds, val_ds, cfg=cfg, verbose=True, show_progress=True)
            print(f"  trained in {(time.time() - f_t0)/60:.1f} min, "
                  f"best epoch {result.best_epoch}, best mean AUC {result.best_val_metric:.4f}")

            # Save checkpoint
            ckpt_path = artifacts_dir / f"{variant}_fold{fold.fold}_best.pt"
            torch.save(model.state_dict(), ckpt_path)

            # Calibration on cal-fit slice
            cal_loader = DataLoader(cal_fit_ds, batch_size=cfg.batch_size, shuffle=False, collate_fn=collate)
            cal_preds, cal_truths = _predict(model, cal_loader, device)
            val_preds_cal = calibrate_predictions(cal_preds, cal_truths, result.val_preds, method="isotonic")
            val_metrics_cal = evaluate_predictions(val_preds_cal, result.val_truths)
            print(f"  VAL raw: {result.val_metrics.summary_line()}")
            print(f"  VAL cal: {val_metrics_cal.summary_line()}")

            # Per-fold val percentile sweep
            val_kept = val_ds.sample_positions
            val_ts = labels.index.to_numpy()[val_kept]
            val_truths_for_sim = _truths_at(labels, val_kept)
            print("  VAL percentile sweep (calibrated):")
            val_sweep = _run_percentile_sweep(val_ts, val_preds_cal, val_truths_for_sim)

            all_results[variant]["folds"].append({
                "fold": fold.fold,
                "best_epoch": result.best_epoch,
                "training_minutes": (time.time() - f_t0) / 60,
                "history": result.history,
                "val_raw": _serialize_metrics(result.val_metrics),
                "val_cal": _serialize_metrics(val_metrics_cal),
                "val_sim_sweep": val_sweep,
                "n_params": n_params,
                "checkpoint": str(ckpt_path),
            })

        # ----- OOS lockbox eval: ensemble of folds vs the lockbox ---------
        if len(splits.oos_idx) > 0 and len(splits.folds) > 0:
            banner(f"6. {variant.upper()} OOS lockbox eval (last {oos_months}mo, never touched in training)")
            # Use the last fold's calibrated predictor as the deployment model.
            # (For the full run with 5 folds, this approximates time-of-deployment.)
            last_fold = splits.folds[-1]
            n_cal = int(len(last_fold.train_idx) * FULL_CFG["cal_holdout_frac"])
            cal_fit_idx = last_fold.train_idx[-n_cal:]
            cal_fit_ds = ALTUSDataset(feats, labels, cal_fit_idx, seq_len=FULL_CFG["seq_len"])
            cal_loader = DataLoader(cal_fit_ds, batch_size=FULL_CFG["batch_size"], shuffle=False, collate_fn=collate)

            # Re-load best weights from the last fold
            last_ckpt = artifacts_dir / f"{variant}_fold{last_fold.fold}_best.pt"
            model = build_hybrid(
                n_features=n_feat, long_context=long_context_for_model,
                d_model=FULL_CFG["d_model"], seq_len=FULL_CFG["seq_len"],
                tcn_n_blocks=FULL_CFG["tcn_n_blocks"],
                mamba_n_blocks=FULL_CFG["mamba_n_blocks"],
                xlstm_n_blocks=FULL_CFG["xlstm_n_blocks"],
            ).to(device)
            model.load_state_dict(torch.load(last_ckpt, map_location=device))
            cal_preds, cal_truths = _predict(model, cal_loader, device)

            oos_ds = ALTUSDataset(feats, labels, splits.oos_idx, seq_len=FULL_CFG["seq_len"])
            oos_loader = DataLoader(oos_ds, batch_size=FULL_CFG["batch_size"], shuffle=False, collate_fn=collate)
            oos_preds_raw, oos_truths = _predict(model, oos_loader, device)
            oos_metrics_raw = evaluate_predictions(oos_preds_raw, oos_truths)
            oos_preds_cal = calibrate_predictions(cal_preds, cal_truths, oos_preds_raw, method="isotonic")
            oos_metrics_cal = evaluate_predictions(oos_preds_cal, oos_truths)
            print(f"  OOS raw: {oos_metrics_raw.summary_line()}")
            print(f"  OOS cal: {oos_metrics_cal.summary_line()}")

            oos_kept = oos_ds.sample_positions
            oos_ts = labels.index.to_numpy()[oos_kept]
            oos_truths_for_sim = _truths_at(labels, oos_kept)
            print("  OOS percentile sweep (calibrated):")
            oos_sweep = _run_percentile_sweep(oos_ts, oos_preds_cal, oos_truths_for_sim)

            all_results[variant]["oos"] = {
                "raw": _serialize_metrics(oos_metrics_raw),
                "cal": _serialize_metrics(oos_metrics_cal),
                "sim_sweep": oos_sweep,
            }

    # ----- Final summary against acceptance criteria ----------------------
    banner("FINAL SUMMARY")
    crit = AcceptanceCriteria()
    print(f"acceptance: AUC>={crit.min_auc_per_side}, topDecileWR>={crit.top_decile_min_winrate}, "
          f"Sharpe>={crit.min_sharpe}, maxDD<{crit.max_drawdown_pct:.0%}, "
          f"+months>={crit.min_pct_positive_months:.0%}")
    for variant, r in all_results.items():
        which = "OOS" if r.get("oos") else "VAL (no OOS in this run)"
        m = r.get("oos", {}).get("cal") if r.get("oos") else r["folds"][-1]["val_cal"]
        sweep = (r.get("oos", {}).get("sim_sweep") if r.get("oos") else r["folds"][-1]["val_sim_sweep"])
        if m is None:
            continue
        print(f"\n{variant.upper()} ({which}, calibrated):")
        print(f"  AUC long={m['auc'].get('long_tp', float('nan')):.4f}  "
              f"short={m['auc'].get('short_tp', float('nan')):.4f}  mean={m['mean_auc']:.4f}")
        print(f"  Brier improvement: long={m['brier_improvement'].get('long_tp', float('nan')):+.3f}  "
              f"short={m['brier_improvement'].get('short_tp', float('nan')):+.3f}")
        print(f"  Top-decile win rate: long={m['top_decile_winrate'].get('long_tp', float('nan')):.3f}  "
              f"short={m['top_decile_winrate'].get('short_tp', float('nan')):.3f}")
        if sweep:
            for k_label in ("top1pct", "top5pct", "top10pct", "top20pct"):
                s = sweep[k_label]
                print(f"  sim {k_label:>9}: trades={s['n_trades']:,}  win={s['win_rate']:.3f}  "
                      f"PnL=${s['total_pnl_usd']:,.0f}  Sharpe={s['sharpe']:.2f}  "
                      f"DD={s['max_drawdown_pct']:.1%}  TPD={s['trades_per_day']:.1f}")

    metrics_path = artifacts_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nmetrics saved: {metrics_path}")
    print(f"checkpoints in: {artifacts_dir}")
    print(f"\nDONE in {(time.time() - t0)/60:.1f} min")


if __name__ == "__main__":
    main()
