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

from dataclasses import dataclass, field
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
    # --- Predictive-vs-pacing diagnostics (2026-05-26) ---
    # Populated by evaluate_predictions when the new heads are present in
    # preds/truths. NaN means the head wasn't trained for this run.
    predictive_diag: dict[str, float] = field(default_factory=dict)

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

    def predictive_diag_line(self) -> str:
        """One-line summary of the predictive-vs-pacing diagnostics.

        Interpretation:
          - dir_pl_ps_corr near -1 = predictive (L/S mutually exclusive)
                          near +1 = PACING (both fire on volatility)
          - ic_H15 / ic_H60 > 0.05 = real signal on return horizon
          - ps_acc > 0.38 = path_shape beats 33% chance baseline
          - clr_auc > 0.55 = clears_level above coin-flip
        """
        if not self.predictive_diag:
            return "(predictive diagnostics not available — heads not trained?)"
        d = self.predictive_diag
        def _f(key, fmt="{:+.3f}"):
            v = d.get(key, float("nan"))
            return fmt.format(v) if not np.isnan(v) else "  nan"
        return (
            f"PREDICTIVE-DIAG | "
            f"corr(P_L,P_S)={_f('dir_pl_ps_corr')} "
            f"ic_H15={_f('ic_return_H15')} ic_H60={_f('ic_return_H60')} "
            f"ps_acc={_f('path_shape_accuracy', '{:.3f}')} "
            f"clr_auc={_f('clears_level_auc', '{:.3f}')}"
        )

    def predictive_diag_verdict(self) -> str:
        """Coarse heuristic verdict on predictive vs pacing.

        Used in the FINAL SUMMARY of the sweep to flag pacing-mode failures
        before we squint at PnL.

        Tiers (2026-05-27 audit recommendation — added STRONG tier above PREDICTIVE):
        - STRONG-PREDICTIVE: ≥3/5 strong bars pass (corr<-0.5, IC>0.06, ps_acc>0.42, clr_auc>0.56)
        - PREDICTIVE: ≥3/5 marginal bars pass (corr<-0.3, IC>0.03, ps_acc>0.38, clr_auc>0.53)
        - WEAK: 1-2 marginal bars pass
        - PACING-LIKE: 0 marginal bars pass
        - PACING: corr(P_L,P_S) > +0.5 (decisive sign of vol-detector failure mode)

        Why two tiers: an IC of 0.03 on return_H15 OOS is real signal that
        survives transaction costs at scale, but it's not a "ship to live"
        bar. STRONG is the "ship" bar; PREDICTIVE is the "promising, iterate".
        """
        if not self.predictive_diag:
            return "INCONCLUSIVE — predictive heads not trained"
        d = self.predictive_diag
        corr = d.get("dir_pl_ps_corr", float("nan"))
        ic15 = d.get("ic_return_H15", float("nan"))
        ic60 = d.get("ic_return_H60", float("nan"))
        ps = d.get("path_shape_accuracy", float("nan"))
        clr = d.get("clears_level_auc", float("nan"))

        # Tell-tale of pacing mode: dir corr is positive (both sides co-move on vol)
        if not np.isnan(corr) and corr > 0.5:
            return "PACING — long/short probs co-move (volatility detector)"

        # Count diagnostics passing the MARGINAL bar
        marginal = 0
        if not np.isnan(corr) and corr < -0.3:
            marginal += 1
        if not np.isnan(ic15) and ic15 > 0.03:
            marginal += 1
        if not np.isnan(ic60) and ic60 > 0.03:
            marginal += 1
        if not np.isnan(ps) and ps > 0.38:
            marginal += 1
        if not np.isnan(clr) and clr > 0.53:
            marginal += 1

        # Count diagnostics passing the STRONG bar
        strong = 0
        if not np.isnan(corr) and corr < -0.5:
            strong += 1
        if not np.isnan(ic15) and ic15 > 0.06:
            strong += 1
        if not np.isnan(ic60) and ic60 > 0.06:
            strong += 1
        if not np.isnan(ps) and ps > 0.42:
            strong += 1
        if not np.isnan(clr) and clr > 0.56:
            strong += 1

        if strong >= 3:
            return f"STRONG-PREDICTIVE ({strong}/5 strong bars + {marginal}/5 marginal — ready for live)"
        if marginal >= 3:
            return f"PREDICTIVE ({marginal}/5 marginal — iterate, not ship-ready)"
        if marginal >= 1:
            return f"WEAK ({marginal}/5 marginal — borderline signal)"
        return "PACING-LIKE (0/5 marginal — no forward signal)"

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

    # --- Predictive-vs-pacing diagnostics --------------------------------
    # These 5 numbers separate "real forward prediction" from "calibrated
    # volatility detector" — see MetricsBundle.predictive_diag_line docstring
    # for interpretation.
    diag: dict[str, float] = {}

    # 1) Long/short prob correlation. A predictive 3-class softmax should give
    #    near-mutually-exclusive long and short probs (corr ≈ -1). A pacing
    #    volatility detector gives both rising together (corr ≈ +1).
    if "long_tp_prob" in preds and "short_tp_prob" in preds:
        pL = np.asarray(preds["long_tp_prob"], dtype=np.float64)
        pS = np.asarray(preds["short_tp_prob"], dtype=np.float64)
        if pL.std() > 1e-9 and pS.std() > 1e-9:
            diag["dir_pl_ps_corr"] = float(np.corrcoef(pL, pS)[0, 1])
        else:
            diag["dir_pl_ps_corr"] = float("nan")

    # 2) Spearman IC on return_H15 and return_H60 — > 0.05 OOS is real signal
    if "return_H15" in preds and "return_H15" in truths:
        diag["ic_return_H15"] = _ic(
            np.asarray(truths["return_H15"], dtype=np.float64),
            np.asarray(preds["return_H15"], dtype=np.float64),
        )
        diag["var_pred_return_H15"] = float(np.var(preds["return_H15"]))
    if "return_H60" in preds and "return_H60" in truths:
        diag["ic_return_H60"] = _ic(
            np.asarray(truths["return_H60"], dtype=np.float64),
            np.asarray(preds["return_H60"], dtype=np.float64),
        )
        diag["var_pred_return_H60"] = float(np.var(preds["return_H60"]))

    # 3) path_shape multi-class accuracy + per-class — > 0.38 beats the
    #    3-class chance baseline of 1/3. Also report per-class precision so
    #    we can see if the model collapsed to a single class.
    if "path_shape_probs" in preds and "path_shape_class" in truths:
        ps_probs = np.asarray(preds["path_shape_probs"])  # (N, 3)
        ps_true = np.asarray(truths["path_shape_class"], dtype=np.int64)
        if ps_probs.ndim == 2 and ps_probs.shape[1] >= 3:
            ps_pred = np.argmax(ps_probs, axis=1)
            diag["path_shape_accuracy"] = float((ps_pred == ps_true).mean())
            for cls in (0, 1, 2):
                mask = ps_true == cls
                if mask.sum() > 0:
                    diag[f"path_shape_recall_c{cls}"] = float((ps_pred[mask] == cls).mean())
                    diag[f"path_shape_base_rate_c{cls}"] = float(mask.mean())

    # 4) clears_level binary AUC — > 0.55 means the head learned something
    #    beyond marginal class rate
    if "clears_level_prob" in preds and "clears_1atr" in truths:
        diag["clears_level_auc"] = _safe_auc(
            np.asarray(truths["clears_1atr"], dtype=np.float64),
            np.asarray(preds["clears_level_prob"], dtype=np.float64),
        )
        diag["clears_level_base_rate"] = float(np.asarray(truths["clears_1atr"]).mean())

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
        predictive_diag=diag,
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
