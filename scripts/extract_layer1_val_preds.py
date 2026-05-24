"""Extract Layer 1 val-fold predictions from existing checkpoints.

Use when you have Layer 1 checkpoints from a cloud run but the val_preds.npz
files weren't saved (i.e., the run was before scripts/train_cloud.py was
updated to save predictions). Runs inference locally on MPS/CPU using the
saved weights, produces the same npz format train_layer2.py expects.

Usage:
    python3 scripts/extract_layer1_val_preds.py \\
        --run-dir artifacts/artifacts/cloud_full_vol+trend+anomaly_20260524_031354 \\
        --families vol,trend,anomaly --use-revin
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True,
                        help="Path to checkpoint folder (containing tcn_fold*_best.pt)")
    parser.add_argument("--families", default="vol,trend,anomaly",
                        help="Structural families used in the original cloud run")
    parser.add_argument("--use-revin", action="store_true",
                        help="Set this if the original run used --use-revin")
    parser.add_argument("--variant", default="tcn",
                        help="Layer 1 variant (default tcn)")
    parser.add_argument("--data-start", default="2023-04-01")
    parser.add_argument("--data-end", default="2026-03-31")
    parser.add_argument("--n-folds", type=int, default=3)
    parser.add_argument("--oos-months", type=int, default=4)
    parser.add_argument("--device", default="mps",
                        help="cpu/mps/cuda — Layer 1 inference is fast even on CPU")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--save-embeddings", action="store_true",
                        help="Also save the 192-D L1 fusion embedding per val sample. "
                             "Adds ~30MB per fold (float16-compressed). Used by Layer 2.")
    args = parser.parse_args()

    from altus.config import TrainConfig
    from altus.data import load_mnq
    from altus.features import StructuralSpec, build_features
    from altus.labels import filter_labels_to_index, triple_barrier_labels
    from altus.models.hybrid import build_hybrid
    from altus.splits import purged_walk_forward
    from altus.training.dataset import ALTUSDataset, collate
    from altus.training.train import _predict, _select_device

    run_dir = Path(args.run_dir).resolve()
    if not run_dir.exists():
        print(f"ERROR: run dir not found: {run_dir}")
        sys.exit(1)
    print(f"Run dir: {run_dir}")
    print(f"Config: families={args.families}, use_revin={args.use_revin}, "
          f"folds={args.n_folds}, oos_months={args.oos_months}")

    # ---- Load data + features (must match cloud run config) ---------------
    t0 = time.time()
    print(f"\n[1/4] Loading MNQ {args.data_start} -> {args.data_end}")
    df_mnq = load_mnq(start=args.data_start, end=args.data_end)
    print(f"      Loaded {len(df_mnq):,} bars")

    print(f"\n[2/4] Building features (this is the slow part — ~1 min on Mac)")
    spec = StructuralSpec.from_string(args.families)
    feats = build_features(df_mnq, structural_spec=spec)
    n_feat = feats.shape[1]
    print(f"      Features: {feats.shape}")

    labels = triple_barrier_labels(df_mnq)
    labels = filter_labels_to_index(labels, feats.index)
    print(f"      Labels: {len(labels.index):,}")

    print(f"\n[3/4] Rebuilding splits (same purged walk-forward as cloud run)")
    splits = purged_walk_forward(labels.index, n_folds=args.n_folds, oos_months=args.oos_months)
    for f in splits.folds:
        print(f"      fold {f.fold}: train={len(f.train_idx):,}, val={len(f.val_idx):,}")

    # ---- Per-fold inference ----------------------------------------------
    device = _select_device(args.device)
    cfg = TrainConfig(batch_size=args.batch_size)
    print(f"\n[4/4] Running inference on each fold's val set (device={device})")

    SEQ_LEN = 240  # matches FULL_CFG in train_cloud.py

    for fold in splits.folds:
        ckpt_path = run_dir / f"{args.variant}_fold{fold.fold}_best.pt"
        if not ckpt_path.exists():
            print(f"  WARNING: missing checkpoint {ckpt_path.name}, skipping fold {fold.fold}")
            continue

        # Build model with matching architecture; load weights
        long_ctx = "none" if args.variant == "tcn" else args.variant
        model = build_hybrid(
            n_features=n_feat,
            long_context=long_ctx,
            d_model=96, seq_len=SEQ_LEN,
            tcn_n_blocks=3, mamba_n_blocks=3, xlstm_n_blocks=3,
            use_revin=args.use_revin,
        )
        state = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(state)
        model = model.to(device).eval()

        # Build val dataset + loader
        val_ds = ALTUSDataset(feats, labels, fold.val_idx, seq_len=SEQ_LEN)
        val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate)

        t_fold = time.time()
        val_preds, val_truths = _predict(model, val_loader, device, return_embeddings=args.save_embeddings)
        elapsed = time.time() - t_fold

        # Save in same format scripts/train_cloud.py now saves.
        # Embeddings are float16 to keep file sizes manageable (half precision is
        # plenty for a frozen feature vector used by a downstream MLP).
        save_dict = {
            **{f"val_preds_{k}": (v.astype(np.float16) if k == "fusion_embedding" else v.astype(np.float32))
               for k, v in val_preds.items()},
            **{f"val_truths_{k}": v for k, v in val_truths.items()},
            "val_positions": val_ds.sample_positions.astype(np.int64),
            "fold": np.int32(fold.fold),
        }
        out_path = run_dir / f"{args.variant}_fold{fold.fold}_val_preds.npz"
        np.savez_compressed(out_path, **save_dict)
        emb_note = f" (+embedding {val_preds['fusion_embedding'].shape})" if "fusion_embedding" in val_preds else ""
        print(f"      fold {fold.fold}: {len(val_ds):,} samples in {elapsed:.1f}s "
              f"-> {out_path.name} ({out_path.stat().st_size/1e6:.1f} MB){emb_note}")

    print(f"\nAll done in {time.time() - t0:.1f}s. val_preds.npz files saved alongside checkpoints.")
    print(f"\nNext step:")
    print(f"  python3 scripts/train_layer2.py --run-dir {run_dir}")


if __name__ == "__main__":
    main()
