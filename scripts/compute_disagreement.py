"""Phase L: Multi-encoder disagreement feature.

Computes per-bar inter-encoder disagreement from two trained variants' val
predictions (e.g., TCN-only vs TCN+Mamba). The disagreement signal serves L2
meta-labeling: when encoders agree, that's a confidence amplifier; when they
disagree, the engine should be cautious. Addresses Q30 in the philosophical
framework.

Inputs: two val_preds.npz files (long_tp_prob + short_tp_prob arrays).
Output: a .npz with disagreement features (per-bar arrays).

Computed features:
  • disagree_long_abs      |p_long_A - p_long_B|         absolute disagreement on long
  • disagree_short_abs     |p_short_A - p_short_B|       absolute disagreement on short
  • disagree_long_kl       KL(p_A || p_B) for long head  divergence in distribution sense
  • disagree_short_kl      same for short head
  • disagree_total         sum of the 4 above              composite uncertainty signal
  • direction_concordance  +1 if both models agree on long-vs-short bias, else -1

Usage:
    python3 scripts/compute_disagreement.py \\
        --preds-a artifacts/cloud_full_<...>_tcn_<...>/tcn_fold0_val_preds.npz \\
        --preds-b artifacts/cloud_full_<...>_mamba_<...>/mamba_fold0_val_preds.npz \\
        --output  artifacts/disagreement_fold0.npz

L2 training picks the file up via an --include-disagreement flag (added in a
separate L2 wiring change). This script just emits the raw signal.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np


EPS = 1e-9


def _kl_bernoulli(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """KL divergence between two Bernoulli distributions: D_KL(p || q)."""
    p = np.clip(p, EPS, 1.0 - EPS)
    q = np.clip(q, EPS, 1.0 - EPS)
    return p * np.log(p / q) + (1.0 - p) * np.log((1.0 - p) / (1.0 - q))


def compute_disagreement(preds_a: dict[str, np.ndarray], preds_b: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    pl_a = preds_a["long_tp_prob"]
    pl_b = preds_b["long_tp_prob"]
    ps_a = preds_a["short_tp_prob"]
    ps_b = preds_b["short_tp_prob"]

    if pl_a.shape != pl_b.shape or ps_a.shape != ps_b.shape:
        raise ValueError(
            f"prediction array shape mismatch: A long={pl_a.shape} short={ps_a.shape}, "
            f"B long={pl_b.shape} short={ps_b.shape}"
        )

    disagree_long_abs = np.abs(pl_a - pl_b).astype(np.float32)
    disagree_short_abs = np.abs(ps_a - ps_b).astype(np.float32)
    disagree_long_kl = ((_kl_bernoulli(pl_a, pl_b) + _kl_bernoulli(pl_b, pl_a)) / 2.0).astype(np.float32)
    disagree_short_kl = ((_kl_bernoulli(ps_a, ps_b) + _kl_bernoulli(ps_b, ps_a)) / 2.0).astype(np.float32)
    disagree_total = (disagree_long_abs + disagree_short_abs + disagree_long_kl + disagree_short_kl).astype(np.float32)

    # Direction concordance: do both models agree on which side has higher prob?
    bias_a = np.sign(pl_a - ps_a)
    bias_b = np.sign(pl_b - ps_b)
    concordance = (bias_a * bias_b).astype(np.float32)  # +1 agree, -1 disagree, 0 indifferent

    return {
        "disagree_long_abs": disagree_long_abs,
        "disagree_short_abs": disagree_short_abs,
        "disagree_long_kl": disagree_long_kl,
        "disagree_short_kl": disagree_short_kl,
        "disagree_total": disagree_total,
        "direction_concordance": concordance,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preds-a", required=True, help="First val_preds.npz file")
    parser.add_argument("--preds-b", required=True, help="Second val_preds.npz file (different encoder)")
    parser.add_argument("--output", required=True, help="Output .npz path")
    args = parser.parse_args()

    a_path = Path(args.preds_a)
    b_path = Path(args.preds_b)
    out_path = Path(args.output)

    if not a_path.exists():
        sys.exit(f"missing --preds-a: {a_path}")
    if not b_path.exists():
        sys.exit(f"missing --preds-b: {b_path}")

    print(f"loading A: {a_path}")
    preds_a = dict(np.load(a_path))
    print(f"  arrays: {list(preds_a.keys())}, long_tp_prob shape: {preds_a['long_tp_prob'].shape}")
    print(f"loading B: {b_path}")
    preds_b = dict(np.load(b_path))
    print(f"  arrays: {list(preds_b.keys())}, long_tp_prob shape: {preds_b['long_tp_prob'].shape}")

    print("\ncomputing disagreement features...")
    disagree = compute_disagreement(preds_a, preds_b)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, **disagree)
    print(f"\nsaved disagreement features: {out_path}")
    print(f"  features: {list(disagree.keys())}")
    print(f"  n samples: {len(disagree['disagree_total']):,}")
    print(f"  mean disagree_total: {disagree['disagree_total'].mean():.4f}")
    print(f"  direction concordance: {(disagree['direction_concordance'] > 0).mean() * 100:.1f}% agree")


if __name__ == "__main__":
    main()
