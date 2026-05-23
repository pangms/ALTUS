"""Post-hoc probability calibration.

Two methods supported:
  * Temperature scaling: divide logits by a learned scalar T before sigmoid.
    Simple, monotonic, preserves ranking (AUC). Often "good enough."
  * Isotonic regression: nonparametric monotone mapping from raw_prob -> calibrated_prob.
    More flexible, can fix non-monotone calibration errors, but needs more data
    and can overfit on small samples.

Always fit on a HELD-OUT slice — never on training-fit predictions and never
on the test set. The standard pattern: split train -> (calibration-fit, calibration-eval),
fit on the cal-fit slice, evaluate on val/OOS.

Why bother: the BCE loss only weakly constrains absolute probability values.
A model can have great AUC (ranking) but predict 0.4 when the true rate is 0.6
across an entire bucket. Trading decisions threshold on probability, so
calibration matters for downstream PnL.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from sklearn.isotonic import IsotonicRegression


@dataclass
class TemperatureScaler:
    """Learn a single positive scalar T s.t. sigmoid(logit / T) is well-calibrated.

    Uses LBFGS on the held-out NLL — fast and reliable for one parameter.
    """
    temperature: float = 1.0

    def fit(self, logits: np.ndarray, y: np.ndarray, max_iter: int = 50) -> "TemperatureScaler":
        logits_t = torch.tensor(logits, dtype=torch.float32)
        y_t = torch.tensor(y, dtype=torch.float32)
        log_T = torch.tensor(0.0, requires_grad=True)
        opt = torch.optim.LBFGS([log_T], lr=0.1, max_iter=max_iter)
        bce = torch.nn.BCEWithLogitsLoss()

        def closure():
            opt.zero_grad()
            T = torch.exp(log_T)  # enforce positivity
            loss = bce(logits_t / T, y_t)
            loss.backward()
            return loss

        opt.step(closure)
        self.temperature = float(torch.exp(log_T).item())
        return self

    def transform(self, logits: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-logits / self.temperature))


@dataclass
class IsotonicCalibrator:
    iso: IsotonicRegression | None = None

    def fit(self, probs: np.ndarray, y: np.ndarray) -> "IsotonicCalibrator":
        self.iso = IsotonicRegression(y_min=1e-4, y_max=1 - 1e-4, out_of_bounds="clip")
        self.iso.fit(probs, y)
        return self

    def transform(self, probs: np.ndarray) -> np.ndarray:
        if self.iso is None:
            raise RuntimeError("fit before transform")
        return self.iso.transform(probs)


def calibrate_predictions(
    cal_fit_preds: dict[str, np.ndarray],
    cal_fit_truths: dict[str, np.ndarray],
    eval_preds: dict[str, np.ndarray],
    method: str = "isotonic",
) -> dict[str, np.ndarray]:
    """Calibrate long_tp_prob and short_tp_prob, leave regression heads untouched."""
    out = dict(eval_preds)  # copy
    for side in ("long_tp", "short_tp"):
        prob_key = f"{side}_prob"
        raw_fit = cal_fit_preds[prob_key]
        y_fit = cal_fit_truths[side]
        raw_eval = eval_preds[prob_key]
        if method == "isotonic":
            cal = IsotonicCalibrator().fit(raw_fit, y_fit)
            out[prob_key] = cal.transform(raw_eval)
        elif method == "temperature":
            # Need logits, not probabilities, for temperature scaling
            logits_fit = np.log(raw_fit.clip(1e-6, 1 - 1e-6) / (1 - raw_fit.clip(1e-6, 1 - 1e-6)))
            logits_eval = np.log(raw_eval.clip(1e-6, 1 - 1e-6) / (1 - raw_eval.clip(1e-6, 1 - 1e-6)))
            ts = TemperatureScaler().fit(logits_fit, y_fit.astype(np.float32))
            out[prob_key] = ts.transform(logits_eval)
        else:
            raise ValueError(f"unknown calibration method: {method}")
    return out
