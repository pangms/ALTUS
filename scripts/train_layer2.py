"""Train Layer 2 (meta-labeling) on top of saved Layer 1 predictions.

Inputs (produced by scripts/train_cloud.py):
  artifacts/<run_dir>/<variant>_fold{0..N}_val_preds.npz
      — one per fold; contains Layer 1's val predictions + truths

What this script does:
  1. Aggregates Layer 1 val predictions across all folds (Layer 1 NEVER saw
     these bars at training time, so it's clean meta-labeling data)
  2. Selects top-K% candidates per the Layer2TrainConfig
  3. Builds Layer 2 input features
  4. Train/val split chronologically (the recent slice is val)
  5. Trains Layer 2 meta-labeler with isotonic calibration + conformal wrap
  6. Reports cascade metrics: Layer 1 alone vs Layer 1+Layer 2

Run from ALTUS root after a successful cloud training run:
    python3 scripts/train_layer2.py --run-dir artifacts/cloud_full_vol+trend+anomaly_<timestamp>

Optional flags:
    --threshold 0.55       # L2 trade-gate threshold
    --top-k 0.20           # top-K% Layer 1 candidates to consider
    --device mps           # cpu/mps; Layer 2 is tiny so either is fast
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _load_layer1_val_preds(run_dir: Path, variant: str = "tcn") -> dict:
    """Load + concat all per-fold val predictions from a cloud run directory.

    Auto-includes Layer 1 fusion embeddings if present (saved by
    extract_layer1_val_preds.py --save-embeddings).
    """
    npz_files = sorted(run_dir.glob(f"{variant}_fold*_val_preds.npz"))
    if not npz_files:
        raise FileNotFoundError(
            f"No val-pred files matching '{variant}_fold*_val_preds.npz' in {run_dir}.\n"
            "Re-run scripts/train_cloud.py or scripts/extract_layer1_val_preds.py."
        )
    out: dict[str, list[np.ndarray]] = {}
    positions: list[np.ndarray] = []
    for f in npz_files:
        data = np.load(f, allow_pickle=False)
        positions.append(data["val_positions"])
        for k in data.files:
            if k in ("val_positions", "fold"):
                continue
            arr = data[k]
            # Embeddings are saved as float16 to save disk space; up-cast on load
            if k == "val_preds_fusion_embedding":
                arr = arr.astype(np.float32)
            out.setdefault(k, []).append(arr)
    merged = {k: np.concatenate(v) for k, v in out.items()}
    merged["val_positions"] = np.concatenate(positions)
    return merged


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True,
                        help="Path to a cloud run dir containing fold*_val_preds.npz files")
    parser.add_argument("--variant", default="tcn", help="Layer 1 variant (default tcn)")
    parser.add_argument("--threshold", type=float, default=0.55,
                        help="L2 trade-gate calibrated-probability threshold (default 0.55)")
    parser.add_argument("--top-k", type=float, default=0.20,
                        help="Top-K%% of Layer 1 signals to consider as candidates (default 0.20)")
    parser.add_argument("--device", default="cpu",
                        help="torch device for the (small) Layer 2 model (default cpu)")
    parser.add_argument("--data-start", default="2023-04-01",
                        help="Start of MNQ data to use (must match Layer 1 cloud run)")
    parser.add_argument("--data-end", default="2026-03-31",
                        help="End of MNQ data to use (must match Layer 1 cloud run)")
    args = parser.parse_args()

    from altus.config import ARTIFACT_DIR, LAYER1_V2_STRUCTURAL_FAMILIES, Layer2TrainConfig
    from altus.data import load_mnq
    from altus.features import StructuralSpec, build_features
    from altus.labels import filter_labels_to_index, triple_barrier_labels
    from altus.models.layer2 import LAYER2_INPUT_FEATURES, build_layer2_input
    from altus.training.layer2_train import (
        evaluate_cascade, select_candidates, train_layer2,
    )

    run_dir = Path(args.run_dir)
    print(f"Loading Layer 1 predictions from {run_dir}")
    l1 = _load_layer1_val_preds(run_dir, variant=args.variant)
    n = len(l1["val_positions"])
    print(f"  Loaded {n:,} val predictions across all folds")

    # ---- Reconstruct features + labels (must match cloud run config) ------
    print(f"\nLoading MNQ + features (range {args.data_start} -> {args.data_end})")
    df_mnq = load_mnq(start=args.data_start, end=args.data_end)
    spec = StructuralSpec.from_string(LAYER1_V2_STRUCTURAL_FAMILIES)
    feats = build_features(df_mnq, structural_spec=spec)
    labels = triple_barrier_labels(df_mnq)
    labels = filter_labels_to_index(labels, feats.index)
    print(f"  Features: {feats.shape}, Labels: {len(labels.index):,}")

    # ---- Reconstruct Layer 1 outputs at label positions -------------------
    # val_positions are integer indices into labels.index from each fold's val_idx
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
    # C-tier predictive heads — added 2026-05-27 to close the Disconnection #1
    # gap (audit). Without these, L2's MLP can't see the magnitude/path/clearance
    # forecasts the predictive pivot exists to deliver. Optional — the older
    # checkpoints may not have these in their npz files.
    if "val_preds_return_H15" in l1:
        l1_outputs["return_H15_pred"] = l1["val_preds_return_H15"][sort_order]
    if "val_preds_return_H60" in l1:
        l1_outputs["return_H60_pred"] = l1["val_preds_return_H60"][sort_order]
    if "val_preds_clears_level_prob" in l1:
        l1_outputs["clears_level_prob"] = l1["val_preds_clears_level_prob"][sort_order]
    if "val_preds_inflection_prob" in l1:
        l1_outputs["inflection_prob"] = l1["val_preds_inflection_prob"][sort_order]
    if "val_preds_path_shape_probs" in l1:
        # Shape (N, 3) — slice into per-class probabilities for the tabular L2 input
        psp = l1["val_preds_path_shape_probs"][sort_order]
        if psp.ndim == 2 and psp.shape[1] >= 3:
            l1_outputs["path_shape_p0"] = psp[:, 0]   # continuation
            l1_outputs["path_shape_p1"] = psp[:, 1]   # revert
            l1_outputs["path_shape_p2"] = psp[:, 2]   # chop
    # Optional: L1 fusion embedding (192-D per sample) — present if extract was
    # run with --save-embeddings
    l1_embeddings = None
    if "val_preds_fusion_embedding" in l1:
        l1_embeddings = l1["val_preds_fusion_embedding"][sort_order]
        print(f"  Loaded L1 fusion embeddings: shape {l1_embeddings.shape} (Layer 2 will use them)")
    else:
        print(f"  (No L1 fusion embeddings in npz; running Layer 2 on hand-crafted features only)")

    # Subset labels to those val positions only
    from altus.labels.triple_barrier import LabelOutput
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
        inflection_label=labels.inflection_label[val_positions],
        tp_points=labels.tp_points[val_positions],
        sl_points=labels.sl_points[val_positions],
        return_H15=labels.return_H15[val_positions],
        return_H60=labels.return_H60[val_positions],
        path_shape_class=labels.path_shape_class[val_positions],
        clears_up_first=labels.clears_up_first[val_positions],
    )

    # ---- Select Layer 1 candidates ----------------------------------------
    cfg = Layer2TrainConfig(candidate_top_k=args.top_k)
    cands = select_candidates(l1_outputs, labels_val, labels_val.index, cfg)
    print(f"\nLayer 1 candidates (top {args.top_k:.0%}): {len(cands.indices):,}")
    print(f"  Meta-label base rate (did L1's chosen side win): {cands.meta_label.mean():.3f}")
    print(f"  Long candidates : {int((cands.direction > 0).sum()):,}")
    print(f"  Short candidates: {int((cands.direction < 0).sum()):,}")

    # ---- Build Layer 2 input features for those candidates ----------------
    X_full = build_layer2_input(cands.layer1_outputs, feats, cands.bar_index)
    y_full = cands.meta_label
    print(f"\nLayer 2 input matrix: {X_full.shape}")
    print(f"  Feature columns: {len(LAYER2_INPUT_FEATURES)}")

    # ---- Chronological train/val split (last 20% as val) ------------------
    n_total = len(X_full)
    split_idx = int(n_total * 0.80)
    X_train, X_val = X_full.iloc[:split_idx], X_full.iloc[split_idx:]
    y_train, y_val = y_full[:split_idx], y_full[split_idx:]
    cands_val_indices = np.arange(split_idx, n_total)
    print(f"\nTrain {len(X_train):,} → Val {len(X_val):,} (chronological)")

    # Also split embeddings if present. Embeddings are indexed by the same
    # candidate ordering, so the same split indices apply.
    emb_train = emb_val = None
    if l1_embeddings is not None:
        # First subset to candidate indices (Layer 2 uses only top-K signals)
        emb_for_candidates = l1_embeddings[cands.indices]
        emb_train = emb_for_candidates[:split_idx]
        emb_val = emb_for_candidates[split_idx:]
        print(f"  Embeddings: train {emb_train.shape}, val {emb_val.shape}")

    # ---- Train Layer 2 ----------------------------------------------------
    print("\nTraining Layer 2 meta-labeler...")
    t0 = time.time()
    result = train_layer2(X_train, y_train, X_val, y_val, cfg=cfg, device=args.device,
                          verbose=True, emb_train=emb_train, emb_val=emb_val)
    print(f"  Trained in {time.time() - t0:.1f}s")
    print(f"  val_auc raw: {result.val_meta_metrics['val_auc_raw']:.4f}")
    print(f"  val_auc cal: {result.val_meta_metrics['val_auc_cal']:.4f}")
    print(f"  val_brier raw: {result.val_meta_metrics['val_brier_raw']:.4f}")
    print(f"  val_brier cal: {result.val_meta_metrics['val_brier_cal']:.4f}")
    print(f"  val_base_rate: {result.val_meta_metrics['val_base_rate']:.3f}")

    # ---- Cascade evaluation: L1 alone vs L1+L2 ----------------------------
    # Use only the val-slice candidates (Layer 2 didn't train on these)
    from altus.training.layer2_train import Candidates
    val_cands = Candidates(
        indices=cands.indices[cands_val_indices],
        layer1_outputs={k: v[cands_val_indices] for k, v in cands.layer1_outputs.items()},
        direction=cands.direction[cands_val_indices],
        meta_label=cands.meta_label[cands_val_indices],
        realized_pnl_pts=cands.realized_pnl_pts[cands_val_indices],
        bar_index=cands.bar_index[cands_val_indices],
    )

    # L2 output distribution diagnostic
    p = result.val_probs_calibrated
    print("\n" + "=" * 72)
    print(" L2 calibrated probability distribution (val slice)")
    print("=" * 72)
    print(f"  min={p.min():.3f}  p25={np.percentile(p,25):.3f}  median={np.median(p):.3f}  "
          f"p75={np.percentile(p,75):.3f}  max={p.max():.3f}  mean={p.mean():.3f}")
    print(f"  fraction >=0.50: {(p>=0.50).mean():.3f}    >=0.55: {(p>=0.55).mean():.3f}    "
          f">=0.60: {(p>=0.60).mean():.3f}")

    print("\n" + "=" * 72)
    print(" CASCADE EVAL — PERCENTILE MODE (rank by L2 score, take top K%)")
    print(" This is the right measure when base rate < 0.5 (always the case for triple-barrier)")
    print("=" * 72)
    for k in (0.01, 0.05, 0.10, 0.20, 0.50):
        cascade = evaluate_cascade(
            val_cands, result.val_probs_calibrated,
            mode="percentile", top_k_fraction=k,
        )
        print(f"  top {int(k*100):>2}%: {cascade.summary_line()}")

    print("\n" + "=" * 72)
    print(" CASCADE EVAL — THRESHOLD MODE (absolute calibrated probability)")
    print(" (likely retains 0 trades if base rate is low — diagnostic only)")
    print("=" * 72)
    for thresh in (0.45, 0.48, 0.50, 0.55, 0.60):
        cascade = evaluate_cascade(val_cands, result.val_probs_calibrated, l2_threshold=thresh)
        print(f"  threshold={thresh:.2f}: {cascade.summary_line()}")

    print("\nConformal-gated cascade (lower-bound of 90% interval >= threshold):")
    for thresh in (0.45, 0.50, 0.55):
        cascade = evaluate_cascade(
            val_cands, result.val_probs_calibrated,
            l2_threshold=thresh, use_conformal=True, conformal_gate=result.conformal,
        )
        print(f"  threshold={thresh:.2f}: {cascade.summary_line()}")

    # ---- L3.1 production-honest cascade sim --------------------------------
    # Project L2 calibrated probs back to per-bar long/short arrays, then run
    # the L3 sim. This is the headline number — what actually lands in PnL
    # under no-overlap + grade sizing + EoD flatten + cooldown.
    print("\n" + "=" * 72)
    print(" L3.1 production sim — L1+L2 cascade (val slice)")
    print("=" * 72)
    from altus.training.production_sim import L3Config, simulate_l3
    n_full = len(labels_val.index)
    long_prob_l2 = np.zeros(n_full, dtype=np.float32)
    short_prob_l2 = np.zeros(n_full, dtype=np.float32)
    val_cal_p = result.val_probs_calibrated.astype(np.float32)
    val_cand_global = val_cands.indices                    # positions into labels_val
    long_mask = val_cands.direction > 0
    long_prob_l2[val_cand_global[long_mask]] = val_cal_p[long_mask]
    short_prob_l2[val_cand_global[~long_mask]] = val_cal_p[~long_mask]
    val_start_ts = labels_val.index[val_cand_global.min()]
    eval_mask = labels_val.index >= val_start_ts
    truths_eval = {
        "long_tp": labels_val.long_tp[eval_mask].astype(np.float32),
        "short_tp": labels_val.short_tp[eval_mask].astype(np.float32),
        "mfe_long": labels_val.mfe_long[eval_mask],
        "mae_long": labels_val.mae_long[eval_mask],
        "mfe_short": labels_val.mfe_short[eval_mask],
        "mae_short": labels_val.mae_short[eval_mask],
    }
    l3_res = simulate_l3(
        labels_val.index[eval_mask].values,
        {"long_tp_prob": long_prob_l2[eval_mask],
         "short_tp_prob": short_prob_l2[eval_mask]},
        truths_eval,
        cfg=L3Config(),
    )
    print(f"  {l3_res.summary_line()}")
    print(f"  worst_day=${l3_res.worst_day_pnl_usd:,.0f}  "
          f"days_trip_trail_dd={l3_res.n_days_would_trip_trailing_dd}  "
          f"max_consec_losses={l3_res.max_consecutive_losses}")

    # ---- Setup-aware L3 cascade (with L2 router + per-setup execution) -----
    # Apply the predictive-framework router on top of the L2 cascade. Reads
    # setup features from the structural feature matrix at each candidate bar,
    # arbitrates the primary setup, applies modulators, and uses setup-
    # conditional stops/targets/holds in the sim.
    setup_feature_cols = [c for c in feats.columns if any(
        c.startswith(p + "_") for p in ("sfs", "sfa", "sld", "orb", "svwap", "spb", "scomp", "seod")
    )]
    if setup_feature_cols:
        print("\n" + "=" * 72)
        print(" L3.1 production sim — SETUP-AWARE cascade (val slice)")
        print(" Uses L2 router + per-setup execution params per SETUPS.md")
        print("=" * 72)

        from altus.training.l2_router import SetupCandidate, route_one_bar, BASELINE_WR
        from altus.training.production_sim import compute_setup_aware_barriers

        # Extract setup features for each val bar
        feats_at_val = feats.reindex(labels_val.index[eval_mask])

        # -------- Setup-conditional WR diagnostic ---------------------------
        # The first sanity check on "do setups carry signal":
        #   unconditional WR vs setup-active WR (per setup, per side).
        # If lift < 1pp across most setups, the setup library is decoration —
        # it identifies configurations the market doesn't actually resolve
        # asymmetrically. If lift > 3pp on the predictive A-tier setups
        # (sfs/sfa/sld), the library has earned its place.
        #
        # We use the realized label outcome directly (long_tp / short_tp) as
        # the WR target — not L2's calibrated prob — so this is a property
        # of the feature library, not the model.
        print("\n" + "=" * 72)
        print(" SETUP-CONDITIONAL WR DIAGNOSTIC (val slice, realized labels)")
        print(" Tests whether each setup-active condition shifts WR vs unconditional")
        print("=" * 72)
        long_tp_eval = labels_val.long_tp[eval_mask].astype(np.float32)
        short_tp_eval = labels_val.short_tp[eval_mask].astype(np.float32)
        uncond_long_wr = float(long_tp_eval.mean())
        uncond_short_wr = float(short_tp_eval.mean())
        print(f"  unconditional WR: long={uncond_long_wr:.4f}  short={uncond_short_wr:.4f}  "
              f"n={len(long_tp_eval):,}")
        for setup_id in ("sfs", "sfa", "sld", "orb", "svwap", "spb", "scomp", "seod"):
            active_col = f"{setup_id}_active"
            dir_col = f"{setup_id}_direction"
            if active_col not in feats_at_val.columns:
                continue
            active = (feats_at_val[active_col].to_numpy(dtype=np.float32) >= 0.5)
            direction = feats_at_val[dir_col].to_numpy(dtype=np.float32) if dir_col in feats_at_val.columns else np.zeros(len(active))
            is_long = active & (direction > 0.5)
            is_short = active & (direction < -0.5)
            n_long, n_short = int(is_long.sum()), int(is_short.sum())
            wr_long = float(long_tp_eval[is_long].mean()) if n_long > 0 else float("nan")
            wr_short = float(short_tp_eval[is_short].mean()) if n_short > 0 else float("nan")
            lift_long = (wr_long - uncond_long_wr) * 100.0 if n_long > 0 else float("nan")
            lift_short = (wr_short - uncond_short_wr) * 100.0 if n_short > 0 else float("nan")
            # Verdict flag
            verdict = ""
            if n_long >= 30 and lift_long > 3.0:
                verdict += " LONG✓"
            elif n_long >= 30 and lift_long < -1.0:
                verdict += " LONG✗"
            if n_short >= 30 and lift_short > 3.0:
                verdict += " SHORT✓"
            elif n_short >= 30 and lift_short < -1.0:
                verdict += " SHORT✗"
            print(f"  {setup_id:>5}: nL={n_long:>5} wr={wr_long:.3f} lift={lift_long:+5.1f}pp  "
                  f"|  nS={n_short:>5} wr={wr_short:.3f} lift={lift_short:+5.1f}pp{verdict}")

        def _setup_candidates_for_bar(row) -> list[SetupCandidate]:
            cands = []
            for setup_id in ("sfs", "sfa", "sld", "orb", "svwap", "spb", "scomp", "seod"):
                active_col = f"{setup_id}_active"
                strength_col = f"{setup_id}_strength"
                dir_col = f"{setup_id}_direction"
                if active_col in row.index and float(row[active_col]) >= 0.5:
                    cands.append(SetupCandidate(
                        setup_id=setup_id,
                        active=float(row[active_col]),
                        strength=float(row.get(strength_col, 0.5)),
                        direction=int(round(float(row.get(dir_col, 0)))),
                    ))
            return cands

        # Per-bar setup_id (primary winner from FULL router cascade) — None
        # when no setup OR when the router abstains (modulator-shrunk WR below
        # breakeven, conformal lower bound below threshold, ambiguous conflict).
        # Disconnection-3 fix (2026-05-26): exercises Stages 1-4 of the L2 router
        # (arbitrate_setups + L2-prob base WR + apply_modulators + gate_decision)
        # instead of arbitrate_setups alone — so the modulator features earn
        # their place AND the conformal gate actually gates.
        n_eval = int(eval_mask.sum())
        setup_ids_per_bar = [None] * n_eval
        router_directions = np.zeros(n_eval, dtype=np.int8)
        router_adjusted_wr = np.zeros(n_eval, dtype=np.float32)
        router_sizing = np.zeros(n_eval, dtype=np.float32)
        abstain_reasons: dict[str, int] = {}

        # Conformal half-width — read directly from the calibrated gate's
        # _q_lo. Used to build a per-bar lower bound (prob - half_width).
        try:
            conformal_hw = float(getattr(result.conformal, "_q_lo", None) or 0.05)
        except Exception:
            conformal_hw = 0.05

        long_prob_eval = long_prob_l2[eval_mask]
        short_prob_eval = short_prob_l2[eval_mask]

        def _base_wr_predictor_factory(bar_idx: int):
            """Returns a closure: (setup_id, ctx) -> base_wr.
            Uses L2's calibrated prob for the setup's direction at this bar;
            falls back to BASELINE_WR if the bar isn't an L2 candidate."""
            lp = float(long_prob_eval[bar_idx])
            sp = float(short_prob_eval[bar_idx])
            def _predict(setup_id, ctx):
                if setup_id is None:
                    # no-setup fallback — conservative
                    return float(max(lp, sp)) * 0.95
                # Pull direction from candidates
                dir_for_setup = 0
                for c in ctx.get("candidates", []):
                    if c.setup_id == setup_id:
                        dir_for_setup = c.direction
                        break
                base_l2 = lp if dir_for_setup > 0 else (sp if dir_for_setup < 0 else 0.0)
                if base_l2 > 0.0:
                    return base_l2
                # Bar isn't an L2 candidate — use setup baseline (priors only).
                return float(BASELINE_WR.get(setup_id, 0.50))
            return _predict

        for i in range(n_eval):
            row = feats_at_val.iloc[i] if i < len(feats_at_val) else None
            if row is None or row.isna().all():
                continue
            cands = _setup_candidates_for_bar(row)
            if not cands:
                continue

            # Build modulator context from L2-input feature columns.
            # HTF agreement (the surfer-bridge nudge): prefer the setup-DIRECTIONAL
            # agreement (primary setup's direction × multi-TF swell) from the
            # setup_htf_context family. This is the audit fix — the old generic
            # `trend_alignment` scalar wasn't setup-aware, so a counter-trend
            # setup got the same nudge as an aligned one. Fall back to the
            # generic scalar for variants that don't include the surfer bridge.
            if "shtf_primary_agree" in row.index:
                htf_agreement = float(row.get("shtf_primary_agree", 0.0) or 0.0)
            else:
                htf_agreement = float(row.get("trend_alignment", 0.0) or 0.0)
            # Model confidence: 1 - normalized 3-class entropy proxy.
            ent = float(row.get("derived_entropy", 0.0) or 0.0)
            model_conf = float(np.clip(1.0 - (ent / math.log(3.0)), 0.0, 1.0)) if ent > 0 else 0.7
            xa_div = bool(int(round(float(row.get("cac_divergence_active", 0.0) or 0.0))))
            # Conformal LB at this bar — approximate with max-side prob - hw
            max_side = max(float(long_prob_eval[i]), float(short_prob_eval[i]))
            conf_lb = max(0.0, max_side - conformal_hw) if max_side > 0 else 0.0

            decision = route_one_bar(
                cands,
                base_wr_predictor=_base_wr_predictor_factory(i),
                htf_agreement=htf_agreement,
                model_confidence=model_conf,
                recent_similar_wr=None,
                drift_score=0.0,
                cross_asset_divergence=xa_div,
                conformal_lower_bound=conf_lb,
            )
            router_adjusted_wr[i] = decision.adjusted_wr
            router_sizing[i] = decision.sizing_factor

            if decision.trade and decision.setup_id is not None:
                setup_ids_per_bar[i] = decision.setup_id
                router_directions[i] = decision.direction
            else:
                reason = decision.abstain_reason or "unknown"
                abstain_reasons[reason] = abstain_reasons.get(reason, 0) + 1

        # Router diagnostics
        n_trade = sum(1 for s in setup_ids_per_bar if s is not None)
        print(f"  router decisions: trade={n_trade}/{n_eval} "
              f"({100.0*n_trade/max(n_eval,1):.2f}%)")
        if abstain_reasons:
            print("  abstain reasons:")
            for reason, count in sorted(abstain_reasons.items(), key=lambda x: -x[1]):
                print(f"    {reason}: {count}")

        # Compute setup-aware barriers
        atr_at_val = np.maximum(
            ((feats_at_val.get("vol_realized_30m") or pd.Series(np.ones(n_eval))).to_numpy(dtype=np.float32)
             if "vol_realized_30m" in feats_at_val.columns else
             np.ones(n_eval, dtype=np.float32)) * 30.0,  # rough conversion
            1.0,
        )
        # Better: use ATR from labels' tp_points (which IS atr*k for vol-scaled)
        atr_at_val = labels_val.tp_points[eval_mask] / 1.5  # k=1.5

        cfg = L3Config()
        tp_eff, sl_eff, _ = compute_setup_aware_barriers(
            setup_ids_per_bar, atr_at_val,
            cfg=cfg,
            default_tp=labels_val.tp_points[eval_mask],
            default_sl=labels_val.sl_points[eval_mask],
        )

        truths_setup_aware = dict(truths_eval)
        truths_setup_aware["tp_points"] = tp_eff
        truths_setup_aware["sl_points"] = sl_eff

        l3_setup = simulate_l3(
            labels_val.index[eval_mask].values,
            {"long_tp_prob": long_prob_l2[eval_mask],
             "short_tp_prob": short_prob_l2[eval_mask]},
            truths_setup_aware,
            cfg=cfg,
        )
        print(f"  {l3_setup.summary_line()}")
        n_setup_trades = sum(1 for s in setup_ids_per_bar if s is not None)
        print(f"  bars with active setup: {n_setup_trades:,} / {n_eval:,}")

        # ---- Bootstrap setup-performance tracker from this run's outcomes ----
        # For each bar that had an active setup, compute the realized R-multiple
        # using setup-aware execution + label outcomes, then populate the
        # SetupPerformanceTracker. Persisted to JSON for next iteration's B1
        # conditional-WR feature.
        from altus.training.setup_performance import SetupPerformanceTracker
        # Regime ID proxy: use bocpd_age_60m bucket if available, else 0
        if "bocpd_age_60m" in feats_at_val.columns:
            bocpd_age = feats_at_val["bocpd_age_60m"].to_numpy()
            regime_ids = np.digitize(bocpd_age, bins=[30, 100, 300]).astype(np.int64)
        else:
            regime_ids = np.zeros(n_eval, dtype=np.int64)

        # Compute realized R per bar from actual mfe/mae vs setup barriers
        eval_long_tp = labels_val.long_tp[eval_mask].astype(np.float32)
        eval_short_tp = labels_val.short_tp[eval_mask].astype(np.float32)
        eval_mfe_long = labels_val.mfe_long[eval_mask]
        eval_mae_long = labels_val.mae_long[eval_mask]
        eval_mfe_short = labels_val.mfe_short[eval_mask]
        eval_mae_short = labels_val.mae_short[eval_mask]
        realized_r = np.zeros(n_eval, dtype=np.float32)
        for i in range(n_eval):
            sid = setup_ids_per_bar[i]
            if sid is None:
                continue
            stop = sl_eff[i]
            target = tp_eff[i]
            if stop <= 0:
                continue
            # Determine direction from setup features
            row = feats_at_val.iloc[i]
            dir_col = f"{sid}_direction"
            direction = int(round(float(row.get(dir_col, 0)))) if dir_col in row.index else 0
            if direction == 0:
                continue
            if direction > 0:
                mfe = float(eval_mfe_long[i]); mae = float(eval_mae_long[i])
            else:
                mfe = float(eval_mfe_short[i]); mae = float(eval_mae_short[i])
            # Did target hit before stop?
            if mfe >= target and mae < stop:
                r = target / stop
            elif mae >= stop:
                r = -1.0
            else:
                # Timeout: rough R from mfe-mae
                r = (mfe - mae) / stop
            realized_r[i] = float(np.clip(r, -3.0, 3.0))

        tracker_path = ARTIFACT_DIR / "setup_performance_tracker.json"
        tracker = SetupPerformanceTracker(persistence_path=tracker_path)
        n_bootstrapped = tracker.bootstrap_from_predictions(
            setup_ids_per_bar,
            regime_ids,
            realized_r,
            labels_val.index[eval_mask].values,
        )
        print(f"  bootstrapped tracker: {n_bootstrapped} trades recorded, "
              f"saved to {tracker_path}")
        # Print cell summary
        summary = tracker.summary()
        if summary:
            print("  per-cell performance:")
            for cell, stats in sorted(summary.items()):
                print(f"    {cell}: n={stats['n']} wr={stats['wr']:.3f} r_mean={stats['r_mean']:+.3f}")
    else:
        print("  (skipping setup-aware cascade — no setup features in feature matrix)")

    # ---- Save trained Layer 2 + outputs -----------------------------------
    out_dir = ARTIFACT_DIR / f"layer2_{int(time.time())}"
    out_dir.mkdir(parents=True, exist_ok=True)
    import torch
    torch.save(result.model.state_dict(), out_dir / "layer2_model.pt")
    with open(out_dir / "metadata.json", "w") as f:
        json.dump({
            "feature_names": list(result.feature_names),
            "val_meta_metrics": result.val_meta_metrics,
            "history": result.history,
            "args": vars(args),
        }, f, indent=2, default=str)
    print(f"\nLayer 2 model + metadata saved to: {out_dir}")


if __name__ == "__main__":
    main()
