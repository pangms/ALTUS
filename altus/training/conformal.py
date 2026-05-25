"""Conformal prediction wrapper for trade-gate confidence intervals.

Split conformal prediction gives any classifier statistically-valid
confidence intervals: "the true probability is in [lo, hi] with 1-α coverage."
For trade gating this translates directly to a meaningful rule:

    Trade only when the LOWER BOUND of the predicted probability ≥ threshold.

This is much stronger than just "P > threshold" because the lower bound
incorporates the model's uncertainty. A signal with predicted P=0.6 but a
wide [0.4, 0.8] interval doesn't trade; one with P=0.6 and a tight
[0.55, 0.65] interval does. The system is conservative when uncertain.

Method: split conformal for binary classification (Vovk et al. 2005,
modernized treatment in Angelopoulos & Bates 2023). Cheap to compute,
mathematically clean, model-agnostic.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ConformalGate:
    """Wraps a calibrated probabilistic classifier with conformal coverage.

    Usage:
        gate = ConformalGate(alpha=0.10)        # 90% coverage
        gate.calibrate(cal_probs, cal_labels)
        lo, hi = gate.predict_interval(test_probs)
        trade_mask = lo >= 0.55                  # only take high-confidence longs
    """
    alpha: float = 0.10           # miscoverage level (1-alpha = coverage)
    _q_lo: float | None = None    # calibration quantile for lower bound
    _q_hi: float | None = None    # calibration quantile for upper bound

    def calibrate(self, cal_probs: np.ndarray, cal_labels: np.ndarray) -> "ConformalGate":
        """Calibrate on (predicted_prob, true_label) pairs from a held-out set.

        Uses absolute-error nonconformity score: |p - y|. For binary y this
        is equivalent to the probability of being wrong on that example.
        """
        cal_probs = np.asarray(cal_probs, dtype=np.float64)
        cal_labels = np.asarray(cal_labels, dtype=np.float64)
        if cal_probs.shape != cal_labels.shape:
            raise ValueError(f"shape mismatch: {cal_probs.shape} vs {cal_labels.shape}")
        n = len(cal_probs)
        if n < 50:
            raise ValueError(f"need >=50 calibration samples; got {n}")

        # Conformity scores: how wrong was each prediction?
        scores = np.abs(cal_probs - cal_labels)
        # Conformal quantile with the (n+1)/n finite-sample correction. Earlier
        # versions computed `int(np.ceil((n+1)*(1-alpha))/n*n)` which is a no-op
        # under int-cast and the actual quantile used `1-alpha` — anti-conservative
        # for small n. Caught in 2026-05-24 audit.
        q_level = min(np.ceil((n + 1) * (1 - self.alpha)) / n, 1.0)
        self._q_lo = float(np.quantile(scores, q_level, method="higher"))
        self._q_hi = self._q_lo
        return self

    def predict_interval(self, probs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return (lower, upper) probability bounds for each prediction."""
        if self._q_lo is None:
            raise RuntimeError("ConformalGate not calibrated yet — call .calibrate() first")
        probs = np.asarray(probs, dtype=np.float64)
        lo = np.clip(probs - self._q_lo, 0.0, 1.0)
        hi = np.clip(probs + self._q_hi, 0.0, 1.0)
        return lo, hi

    def trade_mask(self, probs: np.ndarray, threshold: float = 0.55) -> np.ndarray:
        """Boolean mask: True for signals with lower-bound prob >= threshold.

        This is the actual trade-gate rule: 'I'm conformally sure this trade has
        at least `threshold` probability of being profitable.'
        """
        lo, _ = self.predict_interval(probs)
        return lo >= threshold
