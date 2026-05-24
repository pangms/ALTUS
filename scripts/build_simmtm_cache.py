"""Build the SimMTM embeddings cache for ALTUS L1 training.

Phase K of the architecture. Given a pretrained SimMTM encoder (from
scripts/pretrain_simmtm.py), run inference on every bar's window and save
the per-bar embedding to artifacts/simmtm_embeddings.parquet. The simmtm
feature family (altus/features/families/simmtm.py) loads this at training
time, exposing the embeddings as additional L1 features.

This is the analog of build_kronos_cache.py — heavy GPU op done once.

Defaults: 3yr of MNQ, decimation=10 (compute every 10 bars + ffill),
seq_len=240. Estimated runtime: 20-40min on RTX 5090.

Usage:
    python3 scripts/build_simmtm_cache.py \\
        --encoder artifacts/simmtm_encoder.pt \\
        --data-start 2023-04-01 --data-end 2026-03-31 \\
        --decimation 10
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--encoder", default="artifacts/simmtm_encoder.pt",
                        help="Pretrained encoder checkpoint")
    parser.add_argument("--data-start", default="2023-04-01")
    parser.add_argument("--data-end", default="2026-03-31")
    parser.add_argument("--decimation", type=int, default=10,
                        help="Compute embedding every Nth bar; ffill between. "
                             "10 = ~10x speedup with minimal info loss for SSL features.")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--output", default="artifacts/simmtm_embeddings.parquet")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--disable-cudnn", action="store_true", default=True,
                        help="Disable cuDNN — same RunPod workaround as pretrain_simmtm.py")
    args = parser.parse_args()

    if args.disable_cudnn:
        torch.backends.cudnn.enabled = False
        print("cuDNN: DISABLED (workaround for RunPod version mismatch)")

    from altus.data import load_mnq
    from altus.models.simmtm import SimMTMEncoder

    enc_path = Path(args.encoder)
    if not enc_path.exists():
        sys.exit(f"encoder checkpoint not found: {enc_path}\nRun pretrain_simmtm.py first.")

    device = torch.device(args.device if (
        args.device == "cpu"
        or (args.device == "cuda" and torch.cuda.is_available())
        or (args.device == "mps" and torch.backends.mps.is_available())
    ) else "cpu")
    print(f"Device: {device}")

    t0 = time.time()
    print(f"\n[1/4] Loading encoder from {enc_path}")
    ckpt = torch.load(enc_path, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    print(f"      Encoder config: {cfg}")
    feature_cols = cfg["feature_cols"]
    seq_len = cfg["seq_len"]
    encoder = SimMTMEncoder(
        n_features_in=cfg["n_features_in"],
        d_model=cfg["d_model"],
        n_blocks=cfg["n_blocks"],
        mask_ratio=cfg["mask_ratio"],
    ).to(device)
    encoder.load_state_dict(ckpt["encoder_state"])
    encoder.eval()
    print(f"      Encoder loaded (epoch {ckpt['epoch']}, loss {ckpt['loss']:.4f})")

    print(f"\n[2/4] Loading MNQ {args.data_start} -> {args.data_end}")
    df = load_mnq(start=args.data_start, end=args.data_end)
    print(f"      Loaded {len(df):,} bars")

    print(f"\n[3/4] Running encoder on every {args.decimation}th bar (seq_len={seq_len})")
    x_arr = df[list(feature_cols)].to_numpy(dtype=np.float32)
    n_bars = len(x_arr)
    if n_bars < seq_len:
        sys.exit(f"need at least {seq_len} bars; got {n_bars}")

    # Indices where we compute the embedding (decimated)
    compute_idx = np.arange(seq_len - 1, n_bars, args.decimation)
    print(f"      Will compute embeddings at {len(compute_idx):,} of {n_bars:,} bars")

    embeddings = np.zeros((n_bars, cfg["d_model"]), dtype=np.float32)
    last_emb = None

    # Process in batches
    with torch.no_grad():
        for batch_start in range(0, len(compute_idx), args.batch_size):
            batch_indices = compute_idx[batch_start : batch_start + args.batch_size]
            # Build a batch of windows
            windows = np.zeros((len(batch_indices), seq_len, len(feature_cols)), dtype=np.float32)
            for i, idx in enumerate(batch_indices):
                w = x_arr[idx - seq_len + 1 : idx + 1]
                # Same per-window z-score as pretraining
                mu = w.mean(axis=0, keepdims=True)
                sigma = w.std(axis=0, keepdims=True) + 1e-6
                windows[i] = (w - mu) / sigma
            x_t = torch.from_numpy(windows).to(device)
            emb = encoder.forward_embedding(x_t).cpu().numpy()  # (B, d_model)

            # Place the embeddings at their compute indices
            for i, idx in enumerate(batch_indices):
                embeddings[idx] = emb[i]

            if batch_start % (10 * args.batch_size) == 0:
                pct = 100.0 * (batch_start + len(batch_indices)) / len(compute_idx)
                elapsed = (time.time() - t0) / 60.0
                print(f"    {pct:5.1f}% done | elapsed={elapsed:.1f}min")

    # Forward-fill embeddings to bars in between compute_idx
    print(f"\n[4/4] Forward-filling embeddings between compute points")
    # Mark non-computed positions as NaN, then ffill
    emb_df = pd.DataFrame(embeddings, index=df.index, columns=[f"simmtm_{i:03d}" for i in range(cfg["d_model"])])
    # Bars before first compute_idx are zeros (which is neutral after pretraining z-score)
    first_compute = int(compute_idx[0])
    # Set everything before first_compute to 0; ffill the rest
    mask = np.zeros(n_bars, dtype=bool)
    mask[compute_idx] = True
    emb_df.loc[~mask] = np.nan
    emb_df = emb_df.ffill().fillna(0.0)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    emb_df.to_parquet(out_path)
    print(f"      Saved cache: {out_path}  shape={emb_df.shape}  size={out_path.stat().st_size / 1e6:.1f}MB")

    total = (time.time() - t0) / 60.0
    print(f"\nDONE in {total:.1f} min.")
    print(f"\nNext step: the simmtm feature family will load this cache automatically.")
    print(f"Enable in training: --families <existing>,simmtm")


if __name__ == "__main__":
    main()
