"""Production-stack smoke test: L1 → L2 → L3.1.

Loads a Layer 1 cloud-run artifact directory, retrains Layer 2 in-process
(L2 is a small MLP — ~30s on MPS/CPU), and runs the L3.1 production sim
against BOTH:
  1. L1-only signals (per-bar L1 calibrated prob)        — current sim_pnl baseline
  2. L1+L2 cascade  (L2 calibrated prob, gated to L1's top-K candidates)

Why retrain L2 inside the script:
  train_layer2.py saves only model state_dict() — NOT the isotonic
  calibrator, conformal gate, or input feature mean/std. Those are
  rebuilt from a calibration split at train time. Without saving them
  we cannot reproduce calibrated probabilities standalone, so the
  cleanest fix is to re-fit L2 in-process. (A follow-up should patch
  train_layer2.py to persist the full pipeline.)

Usage:
    python3 scripts/smoke_test_l3_l2.py \\
        --run-dir artifacts/tcn_runs/cloud_full_<families>_<ts>

The script reports two L3.1 summary blocks side by side so the L2 lift
(if any) is visible end-to-end through the execution layer.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from altus.config import (
    LAYER1_V2_STRUCTURAL_FAMILIES,
    Layer2TrainConfig,
)
from altus.data import load_mnq
from altus.features.pipeline import build_features
from altus.features.structural import StructuralSpec
from altus.labels.triple_barrier import LabelOutput, filter_labels_to_index, triple_barrier_labels
from altus.models.layer2 import LAYER2_INPUT_FEATURES, build_layer2_input
from altus.training.layer2_train import select_candidates, train_layer2
from altus.training.production_sim import L3Config, simulate_l3


def _load_layer1_val_preds(run_dir: Path, variant: str = "tcn") -> dict:
    """Load + concat all per-fold val predictions, mirroring train_layer2.py."""
    npz_files = sorted(run_dir.glob(f"{variant}_fold*_val_preds.npz"))
    if not npz_files:
        raise FileNotFoundError(
            f"No '{variant}_fold*_val_preds.npz' in {run_dir}.\n"
            "Check the run dir or re-run scripts/train_cloud.py."
        )
    out: dict = {}
    positions: list[np.ndarray] = []
    for f in npz_files:
        data = np.load(f, allow_pickle=False)
        positions.append(data["val_positions"])
        for k in data.files:
            if k in ("val_positions", "fold"):
                continue
            arr = data[k]
            if k == "val_preds_fusion_embedding":
                arr = arr.astype(np.float32)
            out.setdefault(k, []).append(arr)
    merged = {k: np.concatenate(v) for k, v in out.items()}
    merged["val_positions"] = np.concatenate(positions)
    return merged


def _print_block(title: str, result, l3_cfg: L3Config) -> None:
    print("=" * 78)
    print(title)
    print("=" * 78)
    print(result.summary_line())
    print()
    print("  TopStep telemetry (informational):")
    print(f"    worst_day_pnl   = ${result.worst_day_pnl_usd:,.0f}")
    print(f"    worst_intra_dd  = ${result.worst_intraday_dd_usd:,.0f}")
    print(f"    days that would trip ${l3_cfg.topstep_daily_loss_usd:,.0f} daily-loss: "
          f"{result.n_days_would_trip_daily_loss}")
    print(f"    days that would trip ${l3_cfg.topstep_trailing_dd_usd:,.0f} trail-DD:  "
          f"{result.n_days_would_trip_trailing_dd}")
    print()
    print("  Hard-rule diagnostics:")
    print(f"    EoD entries blocked (within {l3_cfg.eod_no_entry_min}min of NY close): "
          f"{result.n_eod_entries_blocked}")
    print(f"    EoD forced-flatten closures:  {result.n_eod_force_flattened}")
    print(f"    cooldown entries blocked:     {result.n_cooldown_entries_blocked}")
    print(f"    max consecutive losses:       {result.max_consecutive_losses}")
    print()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", required=True,
                    help="L1 cloud-run dir containing tcn_fold*_val_preds.npz")
    ap.add_argument("--variant", default="tcn")
    ap.add_argument("--top-k", type=float, default=0.20,
                    help="Top-K%% of L1 signals to treat as L2 candidates (default 0.20)")
    ap.add_argument("--data-start", default="2023-04-01")
    ap.add_argument("--data-end", default="2026-03-31")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    print(f"Loading L1 val preds from {run_dir}")
    l1 = _load_layer1_val_preds(run_dir, args.variant)
    n = len(l1["val_positions"])
    print(f"  {n:,} val samples across {len(sorted(run_dir.glob(f'{args.variant}_fold*_val_preds.npz')))} folds")

    # ---- Rebuild features + labels (must match the L1 cloud-run config) ----
    print(f"\nReloading MNQ + features ({args.data_start} -> {args.data_end})")
    df_mnq = load_mnq(start=args.data_start, end=args.data_end)
    spec = StructuralSpec.from_string(LAYER1_V2_STRUCTURAL_FAMILIES)
    feats = build_features(df_mnq, structural_spec=spec)
    labels = triple_barrier_labels(df_mnq)
    labels = filter_labels_to_index(labels, feats.index)

    # ---- Order L1 outputs by val_position so timestamps align ----
    sort_order = np.argsort(l1["val_positions"])
    val_positions = l1["val_positions"][sort_order]
    l1_outputs = {
        "long_tp_prob": l1["val_preds_long_tp_prob"][sort_order],
        "short_tp_prob": l1["val_preds_short_tp_prob"][sort_order],
        "mfe_long": l1["val_preds_mfe_long"][sort_order],
        "mae_long": l1["val_preds_mae_long"][sort_order],
        "mfe_short": l1["val_preds_mfe_short"][sort_order],
        "mae_short": l1["val_preds_mae_short"][sort_order],
    }
    l1_emb = (
        l1["val_preds_fusion_embedding"][sort_order]
        if "val_preds_fusion_embedding" in l1 else None
    )
    if l1_emb is not None:
        print(f"  Loaded L1 fusion embeddings: {l1_emb.shape} (L2 will use them)")

    # Subset labels to val positions (we only train/test on what L1 actually scored).
    labels_val = LabelOutput(
        index=labels.index[val_positions],
        long_tp=labels.long_tp[val_positions],
        short_tp=labels.short_tp[val_positions],
        mfe_long=labels.mfe_long[val_positions],
        mae_long=labels.mae_long[val_positions],
        mfe_short=labels.mfe_short[val_positions],
        mae_short=labels.mae_short[val_positions],
        time_to_long_tp=labels.time_to_long_tp[val_positions],
        time_to_short_tp=labels.time_to_short_tp[val_positions],
        entry_price=labels.entry_price[val_positions],
        inflection_label=labels.inflection_label[val_positions]
            if hasattr(labels, "inflection_label") else
            np.zeros(len(val_positions), dtype=np.int8),
        tp_points=labels.tp_points[val_positions],
        sl_points=labels.sl_points[val_positions],
    )

    # ---- Build L1-only "truths" dict for simulate_l3 (uses REAL labels) ----
    truths = {
        "long_tp": labels_val.long_tp.astype(np.float32),
        "short_tp": labels_val.short_tp.astype(np.float32),
        "mfe_long": labels_val.mfe_long,
        "mae_long": labels_val.mae_long,
        "mfe_short": labels_val.mfe_short,
        "mae_short": labels_val.mae_short,
        "tp_points": labels_val.tp_points,
        "sl_points": labels_val.sl_points,
    }
    timestamps = labels_val.index.values  # real UTC timestamps from the label set

    l3_cfg = L3Config()

    # ---- Run 1: L1-only L3 ----
    preds_l1 = {
        "long_tp_prob": l1_outputs["long_tp_prob"],
        "short_tp_prob": l1_outputs["short_tp_prob"],
    }
    print("\nRunning L3.1 on L1-only signals...")
    res_l1 = simulate_l3(timestamps, preds_l1, truths, cfg=l3_cfg)

    # ---- Build L1 candidates → train L2 → calibrated probs ----
    print("\nSelecting L1 candidates + training L2...")
    cfg_l2 = Layer2TrainConfig(candidate_top_k=args.top_k)
    cands = select_candidates(l1_outputs, labels_val, labels_val.index, cfg_l2)
    print(f"  Candidates (top {args.top_k:.0%} of L1): {len(cands.indices):,}  "
          f"({int((cands.direction > 0).sum())}L / {int((cands.direction < 0).sum())}S)")

    X_full = build_layer2_input(cands.layer1_outputs, feats, cands.bar_index)
    y_full = cands.meta_label

    # Chronological 80/20 train/val split (same as train_layer2.py).
    split = int(len(X_full) * 0.80)
    X_train, X_val = X_full.iloc[:split], X_full.iloc[split:]
    y_train, y_val = y_full[:split], y_full[split:]
    emb_train = emb_val = None
    if l1_emb is not None:
        emb_for_cands = l1_emb[cands.indices]
        emb_train = emb_for_cands[:split]
        emb_val = emb_for_cands[split:]

    t0 = time.time()
    result = train_layer2(X_train, y_train, X_val, y_val, cfg=cfg_l2,
                          device=args.device, verbose=False,
                          emb_train=emb_train, emb_val=emb_val)
    print(f"  L2 trained in {time.time()-t0:.1f}s | val_auc_cal={result.val_meta_metrics['val_auc_cal']:.4f}")

    # ---- Project L2 calibrated probs back to per-bar long/short arrays ----
    # L2's val slice corresponds to candidate positions [split:]. Each such
    # candidate has a fixed direction (long or short) and a calibrated prob.
    # For L3, we want per-bar long_prob/short_prob where:
    #   - non-candidate bars → 0 on both sides (L1 already screened them out)
    #   - candidate bars      → calibrated prob on the L1-favored side; 0 other
    val_cand_local_idx = np.arange(split, len(cands.indices))
    val_cand_global_idx = cands.indices[val_cand_local_idx]   # into labels_val.index
    val_cand_dir = cands.direction[val_cand_local_idx]
    val_cand_p = result.val_probs_calibrated.astype(np.float32)

    n_val = len(labels_val.index)
    long_prob_l2 = np.zeros(n_val, dtype=np.float32)
    short_prob_l2 = np.zeros(n_val, dtype=np.float32)
    long_mask = val_cand_dir > 0
    long_prob_l2[val_cand_global_idx[long_mask]] = val_cand_p[long_mask]
    short_prob_l2[val_cand_global_idx[~long_mask]] = val_cand_p[~long_mask]

    # IMPORTANT: only run the cascade sim on the val-slice slice in chronological
    # time so L2's training slice doesn't contaminate the eval. The earliest
    # val-candidate timestamp marks the cutoff.
    val_start_ts = labels_val.index[val_cand_global_idx.min()]
    eval_mask = labels_val.index >= val_start_ts
    print(f"  L1+L2 eval slice: {eval_mask.sum():,} bars from {val_start_ts}")

    preds_l2 = {
        "long_tp_prob": long_prob_l2[eval_mask],
        "short_tp_prob": short_prob_l2[eval_mask],
    }
    truths_eval = {k: v[eval_mask] for k, v in truths.items()}
    timestamps_eval = timestamps[eval_mask]

    print("\nRunning L3.1 on L1+L2 cascade...")
    res_l2 = simulate_l3(timestamps_eval, preds_l2, truths_eval, cfg=l3_cfg)

    # ---- For an apples-to-apples L1-only comparison, also evaluate L1 on the
    # same val-slice window so the time spans match. ----
    preds_l1_eval = {
        "long_tp_prob": l1_outputs["long_tp_prob"][eval_mask],
        "short_tp_prob": l1_outputs["short_tp_prob"][eval_mask],
    }
    res_l1_evalwin = simulate_l3(timestamps_eval, preds_l1_eval, truths_eval, cfg=l3_cfg)

    # ---- Report ----
    print()
    _print_block("L1-ONLY through L3.1 (FULL L1 val slice)", res_l1, l3_cfg)
    _print_block("L1-ONLY through L3.1 (matched L2 eval window)", res_l1_evalwin, l3_cfg)
    _print_block("L1+L2 CASCADE through L3.1 (L2 eval window)", res_l2, l3_cfg)

    # ---- Delta ----
    print("=" * 78)
    print("DELTA (L1+L2 vs L1-only on the matched window):")
    print("=" * 78)
    print(f"  trades:     {res_l1_evalwin.n_trades:>6}  ->  {res_l2.n_trades:>6}")
    print(f"  trades/day: {res_l1_evalwin.trades_per_day:>6.1f}  ->  {res_l2.trades_per_day:>6.1f}")
    print(f"  win_rate:   {res_l1_evalwin.win_rate:>6.3f}  ->  {res_l2.win_rate:>6.3f}")
    print(f"  total PnL:  ${res_l1_evalwin.total_pnl_usd:>10,.0f}  ->  ${res_l2.total_pnl_usd:>10,.0f}")
    print(f"  Sharpe:     {res_l1_evalwin.sharpe:>6.2f}  ->  {res_l2.sharpe:>6.2f}")
    print(f"  max DD %:   {res_l1_evalwin.max_drawdown_pct:>6.1%}  ->  {res_l2.max_drawdown_pct:>6.1%}")
    print()
    print(f"  L2 lift (matched window):")
    print(f"    PnL delta:    ${res_l2.total_pnl_usd - res_l1_evalwin.total_pnl_usd:+,.0f}")
    print(f"    Sharpe delta: {res_l2.sharpe - res_l1_evalwin.sharpe:+.2f}")
    print(f"    WR delta:     {res_l2.win_rate - res_l1_evalwin.win_rate:+.3f}")


if __name__ == "__main__":
    main()
