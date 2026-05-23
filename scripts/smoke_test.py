"""End-to-end smoke test for ALTUS Layer 1.

Goal: validate the data → features → labels → splits → model → metrics pipeline
on a small slice of MNQ. We're NOT trying to get final numbers here. We want to
see:
  1. Everything runs without errors on MPS
  2. Hybrid AUC > 0.5 (i.e., the model is actually learning *something*)
  3. The naive momentum baseline is meaningfully worse than the hybrid

Two model variants are smoked: ModernTCN+Mamba and ModernTCN+xLSTM.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from altus.config import TrainConfig
from altus.data import load_mnq
from altus.features import build_features, feature_column_count
from altus.labels import filter_labels_to_index, triple_barrier_labels
from altus.models.baseline_momentum import MomentumConfig, momentum_signal
from altus.models.hybrid import build_hybrid
from altus.splits import purged_walk_forward
from altus.training import evaluate_predictions, train_model
from altus.training.dataset import ALTUSDataset
from altus.training.sim_pnl import simulate_trading


# Smoke-only knobs (kept small so the run finishes in minutes).
SMOKE_START = "2024-06-01"
SMOKE_END = "2024-09-01"
SMOKE_SEQ_LEN = 96
SMOKE_D_MODEL = 48
SMOKE_BATCH_SIZE = 128
SMOKE_EPOCHS = 2
SMOKE_OOS_MONTHS = 0
SMOKE_FOLDS = 1


def banner(msg: str) -> None:
    print("\n" + "=" * 72 + f"\n{msg}\n" + "=" * 72)


def _truths_at(labels, positions: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "long_tp": labels.long_tp[positions].astype(np.int8),
        "short_tp": labels.short_tp[positions].astype(np.int8),
        "mfe_long": labels.mfe_long[positions],
        "mae_long": labels.mae_long[positions],
        "mfe_short": labels.mfe_short[positions],
        "mae_short": labels.mae_short[positions],
    }


def main():
    t0 = time.time()
    banner("1. Loading MNQ slice")
    df = load_mnq(start=SMOKE_START, end=SMOKE_END)
    print(f"loaded MNQ: {len(df):,} bars, {df.index[0]} -> {df.index[-1]}")

    banner("2. Building multi-timeframe features")
    feats = build_features(df)
    n_feat = feats.shape[1]
    print(f"feature matrix: {feats.shape[0]:,} rows × {n_feat} cols")
    print(f"warmup dropped: {len(df) - len(feats):,} rows")

    banner("3. Triple-barrier labeling")
    labels = triple_barrier_labels(df)
    # Drop labels whose timestamp doesn't exist in features (feature warmup region).
    labels = filter_labels_to_index(labels, feats.index)
    print(f"labels: {len(labels.index):,} samples (aligned to features)")
    print(f"  base rate long_tp:  {labels.long_tp.mean():.3f}")
    print(f"  base rate short_tp: {labels.short_tp.mean():.3f}")
    print(f"  mean MFE long: {labels.mfe_long.mean():.2f} pts, "
          f"MAE long: {labels.mae_long.mean():.2f} pts")

    banner("4. Naive momentum baseline (sanity check)")
    mom = momentum_signal(df, MomentumConfig()).loc[labels.index]
    mom_preds = {
        "long_tp_prob": mom["long_tp_pred"].to_numpy(),
        "short_tp_prob": mom["short_tp_pred"].to_numpy(),
        "mfe_long": np.full(len(labels.index), labels.mfe_long.mean(), dtype=np.float32),
        "mae_long": np.full(len(labels.index), labels.mae_long.mean(), dtype=np.float32),
        "mfe_short": np.full(len(labels.index), labels.mfe_short.mean(), dtype=np.float32),
        "mae_short": np.full(len(labels.index), labels.mae_short.mean(), dtype=np.float32),
    }
    mom_truths = _truths_at(labels, np.arange(len(labels.index)))
    mom_metrics = evaluate_predictions(mom_preds, mom_truths)
    print(f"momentum: {mom_metrics.summary_line()}")

    banner("5. Building splits")
    splits = purged_walk_forward(
        timestamps=labels.index,
        n_folds=SMOKE_FOLDS,
        oos_months=SMOKE_OOS_MONTHS,
    )
    fold = splits.folds[0]
    print(f"fold 0: train={len(fold.train_idx):,}, val={len(fold.val_idx):,}, "
          f"embargo={splits.embargo_bars}")

    results = {}
    for variant in ("mamba", "xlstm"):
        banner(f"6. Training hybrid: ModernTCN + {variant.upper()}")

        train_ds = ALTUSDataset(feats, labels, fold.train_idx, seq_len=SMOKE_SEQ_LEN)
        val_ds = ALTUSDataset(feats, labels, fold.val_idx, seq_len=SMOKE_SEQ_LEN)
        print(f"  train: {len(train_ds):,} samples (after seq-window filter)")
        print(f"  val:   {len(val_ds):,} samples")

        model = build_hybrid(
            n_features=n_feat,
            long_context=variant,
            d_model=SMOKE_D_MODEL,
            seq_len=SMOKE_SEQ_LEN,
            tcn_n_blocks=2,
            mamba_n_blocks=1,
            xlstm_n_blocks=1,
            fusion_hidden=96,
        )
        n_params = sum(p.numel() for p in model.parameters())
        print(f"  model params: {n_params:,}")

        cfg = TrainConfig(
            batch_size=SMOKE_BATCH_SIZE,
            n_epochs=SMOKE_EPOCHS,
            lr=1e-3,
            early_stop_patience=3,
        )
        result = train_model(model, train_ds, val_ds, cfg=cfg, verbose=True)
        print(f"\n  best epoch: {result.best_epoch}, best mean AUC: {result.best_val_metric:.4f}")
        print(f"  val: {result.val_metrics.summary_line()}")

        # Trading sim aligned to val_ds.sample_positions (same order predictions arrived in).
        sim_truths = _truths_at(labels, val_ds.sample_positions)
        sim_ts = labels.index.to_numpy()[val_ds.sample_positions]
        sim = simulate_trading(sim_ts, result.val_preds, sim_truths)
        print(f"  sim: {sim.summary_line()}")
        results[variant] = (result, sim)

    banner("SUMMARY")
    print(f"momentum    : {mom_metrics.summary_line()}")
    for variant, (res, sim) in results.items():
        tag = f"hybrid-{variant:5s}"
        print(f"{tag}: {res.val_metrics.summary_line()}")
        print(f"{' ' * len(tag)}: {sim.summary_line()}")
    print(f"\nDONE in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
