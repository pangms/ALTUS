"""Pretrain the SimMTM encoder via masked-bar reconstruction.

Phase K of the ALTUS architecture. Self-supervised — no labels needed.
Trained on MNQ history; encoder weights saved for later cache generation
(see scripts/build_simmtm_cache.py).

The pretraining objective is straightforward: take a window of bars,
randomly mask 50% of timesteps, reconstruct the masked values. This
forces the encoder to learn meaningful representations of bar context
without needing supervised labels.

OOS contamination guard (2026-05-24 audit fix):
The pretraining window MUST end before the OOS lockbox begins, otherwise
the encoder memorizes statistical structure of the bars it will later
embed for OOS evaluation. We compute `effective_end` from `--data-end`
minus OOS_LOCKBOX_MONTHS + a 1-day embargo, and assert the user hasn't
manually overridden it past the safe cutoff.

Run on a CUDA pod (CPU works for tiny experiments but is slow on full data):
    python3 scripts/pretrain_simmtm.py \\
        --data-start 2021-01-01 --data-end 2026-03-31 \\
        --seq-len 240 --batch-size 256 --n-epochs 30 \\
        --output artifacts/simmtm_encoder.pt

The script will automatically clip the effective end to exclude the OOS
lockbox; pass --pretrain-end explicitly to override (NOT recommended
unless you're knowingly reusing this script for non-OOS purposes).

Estimated time on 5090: ~30-90 min for the full 5yr dataset with 30 epochs.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


SIMMTM_FEATURE_COLS = ("open", "high", "low", "close", "volume")
N_SIMMTM_FEATURES = len(SIMMTM_FEATURE_COLS)


class SimMTMPretrainDataset(Dataset):
    """Random-window dataset over MNQ bars.

    Each __getitem__ returns a (L, F) window sampled from a uniformly-random
    position in the available data. The features are normalized per-bar
    (z-scored within the window) so the masked-reconstruction task is on
    standardized values.
    """

    def __init__(self, df, seq_len: int, n_samples_per_epoch: int = 50_000):
        # Pre-extract numpy block for fast indexing
        self._x = df[list(SIMMTM_FEATURE_COLS)].to_numpy(dtype=np.float32)
        self._n_bars = len(self._x)
        self.seq_len = seq_len
        self.n_samples_per_epoch = n_samples_per_epoch
        if self._n_bars < seq_len + 10:
            raise ValueError(f"need at least {seq_len + 10} bars; got {self._n_bars}")

    def __len__(self) -> int:
        return self.n_samples_per_epoch

    def __getitem__(self, idx: int) -> torch.Tensor:
        # Random window start
        start = np.random.randint(0, self._n_bars - self.seq_len)
        window = self._x[start : start + self.seq_len].copy()  # (L, F)
        # Per-window z-score (causal — uses the full window)
        mu = window.mean(axis=0, keepdims=True)
        sigma = window.std(axis=0, keepdims=True) + 1e-6
        window = (window - mu) / sigma
        return torch.from_numpy(window.astype(np.float32))


def collate_simple(batch):
    return torch.stack(batch, dim=0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-start", default="2021-01-01")
    parser.add_argument("--data-end", default="2026-03-31",
                        help="Last available bar in the source data — used to compute "
                             "the OOS lockbox cutoff. NOT the pretraining cutoff itself.")
    parser.add_argument("--pretrain-end", default=None,
                        help="OVERRIDE: explicit last bar to pretrain on. If unset, "
                             "computed as data-end minus OOS_LOCKBOX_MONTHS minus 1d "
                             "embargo to keep the OOS lockbox unseen by the encoder.")
    parser.add_argument("--seq-len", type=int, default=240)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--n-epochs", type=int, default=30)
    parser.add_argument("--n-samples-per-epoch", type=int, default=50_000)
    parser.add_argument("--d-model", type=int, default=96)
    parser.add_argument("--n-blocks", type=int, default=3)
    parser.add_argument("--mask-ratio", type=float, default=0.5)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--output", default="artifacts/simmtm_encoder.pt")
    parser.add_argument("--device", default="cuda",
                        help="cuda/mps/cpu — CUDA strongly recommended for full runs")
    parser.add_argument("--disable-cudnn", action="store_true", default=True,
                        help="Disable cuDNN. Workaround for RunPod images where the system "
                             "cuDNN is newer than torch 2.4.1's bundled version and triggers "
                             "CUDNN_STATUS_NOT_INITIALIZED. Plain CUDA kernels are used "
                             "instead — slightly slower but functional. Defaults to True "
                             "because we keep hitting this; pass --no-disable-cudnn to test "
                             "without it on a different environment.")
    args = parser.parse_args()

    # Workaround: disable cuDNN to avoid version-mismatch errors on RunPod
    if args.disable_cudnn:
        torch.backends.cudnn.enabled = False
        print("cuDNN: DISABLED (workaround for RunPod version mismatch)")

    from altus.config import OOS_LOCKBOX_MONTHS, PRIMARY_WINDOW_MIN
    from altus.data import build_rolling_ohlcv, load_mnq
    from altus.models.simmtm import SimMTMPretrainModel, masked_reconstruction_loss

    # ---- OOS contamination guard --------------------------------------------
    # Compute the safe pretraining cutoff: data_end - OOS_LOCKBOX_MONTHS - 1d.
    # If the user explicitly passed --pretrain-end, honor it BUT assert it's
    # not past the safe cutoff (with a small grace day for tz/calendar slop).
    import pandas as pd
    data_end_ts = pd.Timestamp(args.data_end)
    safe_cutoff = (data_end_ts
                   - pd.DateOffset(months=OOS_LOCKBOX_MONTHS)
                   - pd.Timedelta(days=1))
    if args.pretrain_end is None:
        effective_end = safe_cutoff
        print(f"OOS guard: pretraining capped at {effective_end.date()} "
              f"(data_end={data_end_ts.date()} - {OOS_LOCKBOX_MONTHS}mo lockbox - 1d embargo)")
    else:
        effective_end = pd.Timestamp(args.pretrain_end)
        if effective_end > safe_cutoff:
            sys.exit(
                f"REFUSING TO PRETRAIN: --pretrain-end {effective_end.date()} is past "
                f"the OOS-safe cutoff {safe_cutoff.date()}. The encoder would memorize "
                f"OOS bars, contaminating downstream evaluation. Pass --pretrain-end "
                f"on or before {safe_cutoff.date()} or remove the override."
            )
        print(f"OOS guard: user override accepted, pretraining to {effective_end.date()}")
    effective_end_str = effective_end.strftime("%Y-%m-%d")

    device = torch.device(args.device if (
        args.device == "cpu"
        or (args.device == "cuda" and torch.cuda.is_available())
        or (args.device == "mps" and torch.backends.mps.is_available())
    ) else "cpu")
    print(f"Device: {device}")

    t0 = time.time()
    print(f"\n[1/3] Loading MNQ {args.data_start} -> {effective_end_str}")
    df = load_mnq(start=args.data_start, end=effective_end_str)
    print(f"      Loaded {len(df):,} 1m bars (OOS bars EXCLUDED from pretraining)")

    # Apply the rolling-primary transform so the encoder learns on the same
    # candle representation the L1 pipeline will feed it at inference time.
    # If PRIMARY_WINDOW_MIN=1, this is a no-op copy.
    if PRIMARY_WINDOW_MIN > 1:
        df = build_rolling_ohlcv(df, PRIMARY_WINDOW_MIN).dropna(how="any")
        print(f"      Aggregated to rolling-{PRIMARY_WINDOW_MIN}m primary bars: "
              f"{len(df):,} rows after warmup trim")

    print(f"\n[2/3] Building model + dataset")
    model = SimMTMPretrainModel(
        n_features_in=N_SIMMTM_FEATURES,
        d_model=args.d_model,
        n_blocks=args.n_blocks,
        mask_ratio=args.mask_ratio,
    ).to(device)
    print(f"      Model params: {sum(p.numel() for p in model.parameters()):,}")

    ds = SimMTMPretrainDataset(df, seq_len=args.seq_len, n_samples_per_epoch=args.n_samples_per_epoch)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=0, collate_fn=collate_simple, drop_last=True)
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=args.n_epochs)

    print(f"\n[3/3] Pretraining ({args.n_epochs} epochs, {args.n_samples_per_epoch:,} samples/epoch)")
    best_loss = float("inf")
    for epoch in range(args.n_epochs):
        model.train()
        running = 0.0
        n = 0
        for batch in loader:
            batch = batch.to(device)
            optim.zero_grad()
            out = model(batch)
            loss = masked_reconstruction_loss(out["x_orig"], out["x_recon"], out["mask"])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            running += loss.item() * batch.shape[0]
            n += batch.shape[0]
        sched.step()
        avg = running / max(n, 1)
        elapsed = (time.time() - t0) / 60.0
        print(f"  epoch {epoch+1:3d}/{args.n_epochs} | loss={avg:.4f} | elapsed={elapsed:.1f}min")
        if avg < best_loss:
            best_loss = avg
            # Save encoder weights only (not the recon head — we throw it away at inference)
            out_path = Path(args.output)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                "encoder_state": model.encoder.state_dict(),
                "config": {
                    "n_features_in": N_SIMMTM_FEATURES,
                    "d_model": args.d_model,
                    "n_blocks": args.n_blocks,
                    "mask_ratio": args.mask_ratio,
                    "seq_len": args.seq_len,
                    "feature_cols": list(SIMMTM_FEATURE_COLS),
                },
                "epoch": epoch + 1,
                "loss": avg,
            }, out_path)
            print(f"    saved best encoder: {out_path}")

    total = (time.time() - t0) / 60.0
    print(f"\nDONE in {total:.1f} min. Best loss: {best_loss:.4f}")
    print(f"Encoder saved to: {args.output}")
    print(f"\nNext step: run scripts/build_simmtm_cache.py to generate per-bar embeddings.")


if __name__ == "__main__":
    main()
