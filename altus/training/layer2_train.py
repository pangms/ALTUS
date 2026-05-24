"""Layer 2 training pipeline.

Workflow:
  1. Load Layer 1 trained checkpoint
  2. Run Layer 1 inference on a dataset (val or OOS) to extract its 6 outputs
  3. Identify candidate signals (top-K% of Layer 1's signals by max probability)
  4. Build Layer 2 input features for those candidates
  5. Determine the meta-label: did Layer 1's winning side actually win?
  6. Train Layer 2 to predict that meta-label
  7. Calibrate (isotonic) + conformal-wrap the trained Layer 2

Layer 2 training data comes from VALIDATION-set Layer 1 predictions, never
training-set. Otherwise Layer 2 would be trained to score predictions Layer 1
saw at training time — leakage that would inflate eval metrics dramatically.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.isotonic import IsotonicRegression
from torch.utils.data import DataLoader, TensorDataset

from altus.config import Layer2Config, Layer2TrainConfig
from altus.labels.triple_barrier import LabelOutput
from altus.models.layer2 import (
    LAYER2_INPUT_FEATURES,
    Layer2MetaLabeler,
    build_layer2,
    build_layer2_input,
    derive_signal_features,
)
from altus.training.conformal import ConformalGate


log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Candidate selection
# ---------------------------------------------------------------------------

@dataclass
class Candidates:
    """A set of Layer 1 candidate signals + their resolved outcomes."""
    indices: np.ndarray              # positions into the source labeled set
    layer1_outputs: dict[str, np.ndarray]
    direction: np.ndarray            # +1 long, -1 short — which side L1 favored
    meta_label: np.ndarray           # 1 if L1's winning side actually won; else 0
    realized_pnl_pts: np.ndarray     # realized point P&L assuming we took the trade
    bar_index: pd.DatetimeIndex      # timestamps for the candidates


def select_candidates(
    layer1_outputs: dict[str, np.ndarray],
    labels: LabelOutput,
    label_index: pd.DatetimeIndex,
    cfg: Layer2TrainConfig,
) -> Candidates:
    """Pick the Layer 1 candidate signals we'll ask Layer 2 to score.

    'Candidate' = a bar where Layer 1's max-side probability is in the top K%
    of the distribution (matches our percentile-threshold trading sim).
    """
    long_p = np.asarray(layer1_outputs["long_tp_prob"], dtype=np.float64)
    short_p = np.asarray(layer1_outputs["short_tp_prob"], dtype=np.float64)
    n = len(long_p)

    # Direction: which side did Layer 1 favor more?
    direction = np.where(long_p >= short_p, 1, -1)
    max_p = np.maximum(long_p, short_p)

    if cfg.candidate_mode == "top_k_percent":
        k = max(1, int(np.ceil(cfg.candidate_top_k * n)))
        cand_idx = np.argpartition(-max_p, k)[:k]
        cand_idx = np.sort(cand_idx)
    elif cfg.candidate_mode == "all_bars":
        cand_idx = np.arange(n, dtype=np.int64)
    else:
        raise ValueError(f"unknown candidate_mode: {cfg.candidate_mode}")

    # Meta-label: did Layer 1's chosen side actually win?
    long_won = labels.long_tp[cand_idx].astype(np.int8)
    short_won = labels.short_tp[cand_idx].astype(np.int8)
    cand_dir = direction[cand_idx]
    meta_label = np.where(cand_dir > 0, long_won, short_won).astype(np.int8)

    # Realized PnL in points (used for evaluation only — not training)
    mfeL = labels.mfe_long[cand_idx]
    maeL = labels.mae_long[cand_idx]
    mfeS = labels.mfe_short[cand_idx]
    maeS = labels.mae_short[cand_idx]
    from altus.config import SL_POINTS, TP_POINTS
    long_pnl = np.where(long_won == 1, TP_POINTS,
                        np.where(maeL >= SL_POINTS, -SL_POINTS, mfeL - maeL))
    short_pnl = np.where(short_won == 1, TP_POINTS,
                         np.where(maeS >= SL_POINTS, -SL_POINTS, mfeS - maeS))
    realized_pnl_pts = np.where(cand_dir > 0, long_pnl, short_pnl).astype(np.float32)

    # Subset Layer 1 outputs to just the candidates
    cand_l1 = {k: np.asarray(v)[cand_idx] for k, v in layer1_outputs.items()}

    return Candidates(
        indices=cand_idx,
        layer1_outputs=cand_l1,
        direction=cand_dir.astype(np.int8),
        meta_label=meta_label,
        realized_pnl_pts=realized_pnl_pts,
        bar_index=label_index[cand_idx],
    )


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

@dataclass
class Layer2TrainResult:
    model: Layer2MetaLabeler
    feature_names: tuple[str, ...]
    history: list[dict]
    val_probs_raw: np.ndarray
    val_probs_calibrated: np.ndarray
    val_labels: np.ndarray
    val_meta_metrics: dict[str, float]
    isotonic: IsotonicRegression
    conformal: ConformalGate


def train_layer2(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    cfg: Layer2TrainConfig | None = None,
    device: str = "cpu",
    verbose: bool = True,
) -> Layer2TrainResult:
    """Train a Layer 2 meta-labeler on (X_train, y_train), validate on (X_val, y_val).

    X_*: DataFrames with columns LAYER2_INPUT_FEATURES (in that order).
    y_*: 1-D int8 arrays of 0/1 meta-labels (was the L1 trade actually profitable).

    Returns the trained model + isotonic calibration + conformal gate, all
    ready for downstream cascade evaluation.
    """
    cfg = cfg or Layer2TrainConfig()
    dev = torch.device(device)
    torch.manual_seed(1337)
    np.random.seed(1337)

    feature_names = tuple(X_train.columns)
    input_dim = X_train.shape[1]

    # ---- Split a calibration tail off the training set --------------------
    n_train = len(X_train)
    n_cal = int(n_train * cfg.cal_holdout_frac)
    X_fit = X_train.iloc[: n_train - n_cal]
    y_fit = y_train[: n_train - n_cal]
    X_cal = X_train.iloc[n_train - n_cal :]
    y_cal = y_train[n_train - n_cal :]

    # ---- Standardize inputs (per-feature z-score from fit set) ------------
    feat_mean = X_fit.mean().to_numpy(dtype=np.float32)
    feat_std = X_fit.std().to_numpy(dtype=np.float32).clip(min=1e-6)

    def _to_tensor(df_or_arr) -> torch.Tensor:
        arr = df_or_arr.to_numpy(dtype=np.float32) if hasattr(df_or_arr, "to_numpy") else np.asarray(df_or_arr, dtype=np.float32)
        arr = (arr - feat_mean) / feat_std
        return torch.from_numpy(arr).to(dev)

    Xf_t = _to_tensor(X_fit)
    yf_t = torch.from_numpy(y_fit.astype(np.float32)).to(dev)
    Xc_t = _to_tensor(X_cal)
    yc_t = torch.from_numpy(y_cal.astype(np.float32)).to(dev)
    Xv_t = _to_tensor(X_val)

    # ---- Build model + optimizer ------------------------------------------
    model = build_layer2(input_dim=input_dim,
                         hidden_dim=Layer2Config().hidden_dim,
                         n_hidden_layers=Layer2Config().n_hidden_layers,
                         dropout=Layer2Config().dropout).to(dev)
    n_params = sum(p.numel() for p in model.parameters())
    if verbose:
        log.info(f"Layer 2: {n_params:,} params, train={len(X_fit):,}, "
                 f"cal={len(X_cal):,}, val={len(X_val):,}, input_dim={input_dim}")

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.n_epochs)
    bce = nn.BCEWithLogitsLoss()

    # ---- Training loop ----------------------------------------------------
    loader = DataLoader(TensorDataset(Xf_t, yf_t), batch_size=cfg.batch_size,
                        shuffle=True, drop_last=False)
    best_val_auc = -float("inf")
    best_state = None
    patience = cfg.early_stop_patience
    history: list[dict] = []

    for epoch in range(cfg.n_epochs):
        model.train()
        epoch_loss = 0.0
        for xb, yb in loader:
            yb_smooth = yb * (1 - 2 * cfg.label_smoothing) + cfg.label_smoothing
            opt.zero_grad()
            logits = model(xb)
            loss = bce(logits, yb_smooth)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()
            epoch_loss += float(loss.detach()) * xb.size(0)
        sched.step()
        epoch_loss /= len(Xf_t)

        # Validation
        model.eval()
        with torch.no_grad():
            val_probs = torch.sigmoid(model(Xv_t)).cpu().numpy()
        # Compute val AUC (simple)
        try:
            from sklearn.metrics import roc_auc_score
            if len(np.unique(y_val)) >= 2:
                val_auc = float(roc_auc_score(y_val, val_probs))
            else:
                val_auc = float("nan")
        except Exception:
            val_auc = float("nan")

        history.append({"epoch": epoch + 1, "train_loss": epoch_loss, "val_auc": val_auc})
        if verbose:
            log.info(f"[L2 epoch {epoch+1}] train_loss={epoch_loss:.4f}  val_auc={val_auc:.4f}")

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience = cfg.early_stop_patience
        else:
            patience -= 1
            if patience <= 0:
                if verbose:
                    log.info(f"[L2 early-stop] no improvement for {cfg.early_stop_patience} epochs")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    # ---- Final raw predictions on cal + val -------------------------------
    model.eval()
    with torch.no_grad():
        cal_probs_raw = torch.sigmoid(model(Xc_t)).cpu().numpy()
        val_probs_raw = torch.sigmoid(model(Xv_t)).cpu().numpy()

    # ---- Isotonic calibration fit on the cal slice ------------------------
    iso = IsotonicRegression(y_min=1e-4, y_max=1 - 1e-4, out_of_bounds="clip")
    iso.fit(cal_probs_raw, y_cal.astype(np.float64))
    val_probs_cal = iso.transform(val_probs_raw)

    # ---- Conformal gate fit on the cal slice (calibrated probs) -----------
    conformal = ConformalGate(alpha=0.10).calibrate(iso.transform(cal_probs_raw), y_cal)

    # ---- Quick val metrics summary ----------------------------------------
    from sklearn.metrics import brier_score_loss, roc_auc_score
    val_meta_metrics = {
        "val_auc_raw": float(roc_auc_score(y_val, val_probs_raw)) if len(np.unique(y_val)) >= 2 else float("nan"),
        "val_auc_cal": float(roc_auc_score(y_val, val_probs_cal)) if len(np.unique(y_val)) >= 2 else float("nan"),
        "val_brier_raw": float(brier_score_loss(y_val, val_probs_raw)),
        "val_brier_cal": float(brier_score_loss(y_val, val_probs_cal)),
        "val_base_rate": float(y_val.mean()),
    }

    return Layer2TrainResult(
        model=model,
        feature_names=feature_names,
        history=history,
        val_probs_raw=val_probs_raw,
        val_probs_calibrated=val_probs_cal,
        val_labels=y_val,
        val_meta_metrics=val_meta_metrics,
        isotonic=iso,
        conformal=conformal,
    )


# ---------------------------------------------------------------------------
# Cascade evaluation: Layer 1 + Layer 2 together
# ---------------------------------------------------------------------------

@dataclass
class CascadeResult:
    n_candidates_l1: int
    n_filtered_l2: int
    retention_pct: float
    win_rate_l1_only: float
    win_rate_l1_l2: float
    pnl_pts_l1_only: float
    pnl_pts_l1_l2: float
    avg_realized_r_l1_only: float
    avg_realized_r_l1_l2: float

    def summary_line(self) -> str:
        return (
            f"L1 candidates: {self.n_candidates_l1:,} → L2 kept: {self.n_filtered_l2:,} "
            f"({self.retention_pct:.1%})  | "
            f"WR L1 alone: {self.win_rate_l1_only:.3f} → L1+L2: {self.win_rate_l1_l2:.3f} "
            f"(Δ {self.win_rate_l1_l2 - self.win_rate_l1_only:+.3f})  | "
            f"avgR L1 alone: {self.avg_realized_r_l1_only:+.3f} → L1+L2: {self.avg_realized_r_l1_l2:+.3f}"
        )


def evaluate_cascade(
    candidates: Candidates,
    l2_probs_calibrated: np.ndarray,
    l2_threshold: float = 0.55,
    use_conformal: bool = False,
    conformal_gate: ConformalGate | None = None,
) -> CascadeResult:
    """Measure how Layer 2 filtering changes the trading outcome.

    Compares:
      - Layer 1 alone: trade every candidate
      - Layer 1 + Layer 2: trade only when L2's calibrated P >= threshold
        (or when conformal lower-bound >= threshold if use_conformal=True)
    """
    from altus.config import SL_POINTS

    n_candidates = len(candidates.indices)
    if n_candidates == 0:
        return CascadeResult(0, 0, 0.0, float("nan"), float("nan"), 0.0, 0.0, 0.0, 0.0)

    if use_conformal:
        if conformal_gate is None:
            raise ValueError("use_conformal=True but conformal_gate is None")
        keep_mask = conformal_gate.trade_mask(l2_probs_calibrated, threshold=l2_threshold)
    else:
        keep_mask = l2_probs_calibrated >= l2_threshold

    n_filtered = int(keep_mask.sum())

    # L1 alone metrics
    wr_l1 = float(candidates.meta_label.mean())
    pnl_l1 = float(candidates.realized_pnl_pts.sum())
    avgr_l1 = float(candidates.realized_pnl_pts.mean() / SL_POINTS)

    # L1 + L2 metrics
    if n_filtered == 0:
        wr_l2 = float("nan")
        pnl_l2 = 0.0
        avgr_l2 = 0.0
    else:
        wr_l2 = float(candidates.meta_label[keep_mask].mean())
        pnl_l2 = float(candidates.realized_pnl_pts[keep_mask].sum())
        avgr_l2 = float(candidates.realized_pnl_pts[keep_mask].mean() / SL_POINTS)

    return CascadeResult(
        n_candidates_l1=n_candidates,
        n_filtered_l2=n_filtered,
        retention_pct=n_filtered / max(n_candidates, 1),
        win_rate_l1_only=wr_l1,
        win_rate_l1_l2=wr_l2,
        pnl_pts_l1_only=pnl_l1,
        pnl_pts_l1_l2=pnl_l2,
        avg_realized_r_l1_only=avgr_l1,
        avg_realized_r_l1_l2=avgr_l2,
    )
