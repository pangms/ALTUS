"""Evaluation metrics for ALTUS Layer 1.

Two categories:
  A) Statistical quality — AUC, PR-AUC, Brier, reliability, IC, RMSE on MFE/MAE
  B) Decision-quality — top-K precision, lift, threshold-sweep trade-frequency

Most metrics are computed per side (long, short). `evaluate_predictions` returns
a dict you can stash, log, or print.

Why we report Brier alongside AUC: AUC measures ranking — "does the model rank
TP winners above non-winners?" — but a well-ranked model can still have miscalibrated
probabilities (e.g., always predicts 0.6 when true rate is 0.4). For trading,
we threshold on probability and need that threshold to mean what we think it
means. Brier score and reliability diagrams reveal calibration directly.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
)


@dataclass
class MetricsBundle:
    n: int
    base_rate: dict[str, float]
    auc: dict[str, float]
    pr_auc: dict[str, float]
    brier: dict[str, float]
    brier_baseline: dict[str, float]
    brier_improvement: dict[str, float]
    ic: dict[str, float]
    top_decile_winrate: dict[str, float]
    top_5pct_winrate: dict[str, float]
    top_1pct_winrate: dict[str, float]
    mfe_rmse: dict[str, float]
    mae_rmse: dict[str, float]

    def summary_line(self) -> str:
        auc_l = self.auc.get("long_tp", float("nan"))
        auc_s = self.auc.get("short_tp", float("nan"))
        bri_l = self.brier_improvement.get("long_tp", float("nan"))
        bri_s = self.brier_improvement.get("short_tp", float("nan"))
        td_l = self.top_decile_winrate.get("long_tp", float("nan"))
        td_s = self.top_decile_winrate.get("short_tp", float("nan"))
        return (
            f"n={self.n} | AUC L={auc_l:.4f} S={auc_s:.4f} "
            f"| BrierImp L={bri_l:+.3f} S={bri_s:+.3f} "
            f"| TopDecileWR L={td_l:.3f} S={td_s:.3f}"
        )

    def mean_auc(self) -> float:
        vals = [v for v in self.auc.values() if not np.isnan(v)]
        return float(np.mean(vals)) if vals else float("nan")


def _safe_auc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_pred))


def _top_k_winrate(y_true: np.ndarray, y_pred: np.ndarray, k_frac: float) -> float:
    if len(y_true) == 0:
        return float("nan")
    k = max(1, int(np.ceil(k_frac * len(y_pred))))
    top_idx = np.argpartition(y_pred, -k)[-k:]
    return float(y_true[top_idx].mean())


def _ic(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Spearman rank correlation (Information Coefficient)."""
    from scipy.stats import spearmanr
    if len(y_true) < 30:
        return float("nan")
    return float(spearmanr(y_true, y_pred).correlation)


def evaluate_predictions(
    preds: dict[str, np.ndarray],
    truths: dict[str, np.ndarray],
) -> MetricsBundle:
    """Compute the full metrics bundle from arrays.

    `preds` must contain:
      long_tp_prob, short_tp_prob, mfe_long, mae_long, mfe_short, mae_short
    `truths` must contain:
      long_tp, short_tp, mfe_long, mae_long, mfe_short, mae_short
    All arrays must be aligned and 1D.
    """
    n = len(next(iter(truths.values())))
    cls_keys = ("long_tp", "short_tp")
    reg_pairs = (("mfe_long", "mae_long"), ("mfe_short", "mae_short"))

    base = {k: float(truths[k].mean()) for k in cls_keys}
    auc = {k: _safe_auc(truths[k], preds[f"{k}_prob"]) for k in cls_keys}
    pr = {k: float(average_precision_score(truths[k], preds[f"{k}_prob"])) if len(np.unique(truths[k])) > 1 else float("nan")
          for k in cls_keys}
    brier = {k: float(brier_score_loss(truths[k], preds[f"{k}_prob"])) for k in cls_keys}
    brier_base = {k: float(base[k] * (1 - base[k])) for k in cls_keys}
    brier_imp = {k: (brier_base[k] - brier[k]) / max(brier_base[k], 1e-9) for k in cls_keys}
    ic = {k: _ic(truths[k].astype(float), preds[f"{k}_prob"]) for k in cls_keys}

    top10 = {k: _top_k_winrate(truths[k], preds[f"{k}_prob"], 0.10) for k in cls_keys}
    top5 = {k: _top_k_winrate(truths[k], preds[f"{k}_prob"], 0.05) for k in cls_keys}
    top1 = {k: _top_k_winrate(truths[k], preds[f"{k}_prob"], 0.01) for k in cls_keys}

    mfe_rmse = {}
    mae_rmse = {}
    for mfe_k, mae_k in reg_pairs:
        mfe_rmse[mfe_k] = float(np.sqrt(np.mean((preds[mfe_k] - truths[mfe_k]) ** 2)))
        mae_rmse[mae_k] = float(np.sqrt(np.mean((preds[mae_k] - truths[mae_k]) ** 2)))

    return MetricsBundle(
        n=n,
        base_rate=base,
        auc=auc,
        pr_auc=pr,
        brier=brier,
        brier_baseline=brier_base,
        brier_improvement=brier_imp,
        ic=ic,
        top_decile_winrate=top10,
        top_5pct_winrate=top5,
        top_1pct_winrate=top1,
        mfe_rmse=mfe_rmse,
        mae_rmse=mae_rmse,
    )


def reliability_bins(
    y_true: np.ndarray, y_pred: np.ndarray, n_bins: int = 10
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (bin_centers, predicted_mean_per_bin, observed_rate_per_bin) for
    plotting reliability diagrams. Equal-width bins on [0, 1]."""
    edges = np.linspace(0, 1, n_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    pred_mean = np.full(n_bins, np.nan)
    obs_rate = np.full(n_bins, np.nan)
    for i in range(n_bins):
        m = (y_pred >= edges[i]) & (y_pred < edges[i + 1])
        if i == n_bins - 1:
            m = m | (y_pred == 1.0)
        if m.sum() == 0:
            continue
        pred_mean[i] = y_pred[m].mean()
        obs_rate[i] = y_true[m].mean()
    return centers, pred_mean, obs_rate
