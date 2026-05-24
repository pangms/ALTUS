"""Family F (Phase F): BOCPD multi-timescale regime detector.

Answers Q19 (stale regime classification) and Q20 (regime confirmed today).

WHY THIS COMPONENT EXISTS (architectural justification):
ModernTCN is a pattern recognizer — convolutional, translation-invariant. It
has no native ability to produce a calibrated discrete-state posterior with
explicit uncertainty over "what regime are we in and how confident." That's
a categorically different kind of computation (Bayesian state-space, not
pattern matching). BOCPD fills this gap as a small parallel component, fed
to L1 as features and to L2 as gating context.

METHOD: Bayesian Online Change-Point Detection (Adams & MacKay 2007).
Maintains a posterior p(r_t | x_1..t) over the "run length" r_t — the number
of bars since the last change point. At each bar we update the posterior
using a constant-hazard prior and a Student-t predictive distribution
(robust to fat-tailed financial returns). Outputs:
  - regime_age: expected run length E[r_t]
  - change_prob: P(r_t = 0) — probability current bar is a change point
  - run_entropy: H(r_t) — sharpness of regime certainty

We run BOCPD independently at three timescales (5m / 60m / 4h-equivalent
windows of 1m bars) so the engine sees regime structure at multiple horizons.

SURFER PRINCIPLE: regime outputs are FEATURES only. NEVER used as gates.
The model learns to weight (setup × regime_intersection); it must remain
free to take counter-HTF-regime trades on strong local conviction.

Features (9 total = 3 metrics × 3 timescales):
  • bocpd_age_5m, bocpd_age_60m, bocpd_age_4h       expected regime age (bars)
  • bocpd_cp_prob_5m/60m/4h                          P(change point now)
  • bocpd_entropy_5m/60m/4h                          run-length distribution entropy

CAUSALITY: BOCPD is an online algorithm — by construction, the posterior
at bar T uses only bars ≤ T. Orchestrator shift handles the final T → T-1.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


EPS = 1e-12
LOG_EPS = -1e30


def _log_student_t_pdf(x: float, mu: float, kappa: float, alpha: float, beta: float) -> float:
    """Log of Student-t posterior predictive: t_{2alpha}(mu, beta*(kappa+1)/(alpha*kappa))."""
    nu = 2.0 * alpha
    sigma2 = beta * (kappa + 1.0) / (alpha * kappa + EPS)
    if sigma2 <= 0:
        return LOG_EPS
    z = (x - mu) ** 2 / sigma2
    # log Student-t density up to a constant (constants cancel in normalization)
    # Use the form: -0.5*log(sigma2) - (nu+1)/2 * log(1 + z/nu)
    log_pdf = -0.5 * np.log(max(sigma2, EPS)) - ((nu + 1.0) / 2.0) * np.log(1.0 + z / max(nu, EPS))
    return float(log_pdf)


def _bocpd_run(
    series: np.ndarray,
    hazard: float = 1.0 / 250.0,   # prior P(change) per bar; 1/250 ≈ once per session
    mu0: float = 0.0,
    kappa0: float = 1.0,
    alpha0: float = 1.0,
    beta0: float = 1.0,
    max_run_length: int = 500,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Online BOCPD on a 1-D series. Returns (expected_age, cp_prob, entropy) per bar.

    Uses log-space message passing for numerical stability and caps the
    run-length posterior at max_run_length to bound compute (any older
    run-lengths get truncated; for our 1m-bar horizon, 500 bars is plenty).
    """
    n = len(series)
    expected_age = np.zeros(n, dtype=np.float64)
    cp_prob = np.zeros(n, dtype=np.float64)
    entropy = np.zeros(n, dtype=np.float64)

    # Posterior over run lengths in log space; index r means "current run is r bars long"
    log_posterior = np.full(max_run_length + 1, LOG_EPS, dtype=np.float64)
    log_posterior[0] = 0.0  # start with run length 0 with certainty

    # Sufficient statistics for the Normal-Inverse-Gamma posterior, one set per
    # possible run length. Initialized to the prior values.
    mu = np.full(max_run_length + 1, mu0, dtype=np.float64)
    kappa = np.full(max_run_length + 1, kappa0, dtype=np.float64)
    alpha = np.full(max_run_length + 1, alpha0, dtype=np.float64)
    beta = np.full(max_run_length + 1, beta0, dtype=np.float64)

    log_hazard = np.log(max(hazard, EPS))
    log_1m_hazard = np.log(max(1.0 - hazard, EPS))

    for t in range(n):
        x = float(series[t])
        if not np.isfinite(x):
            x = 0.0

        # 1. Compute predictive log-pdf for each run length
        log_pi = np.full(max_run_length + 1, LOG_EPS)
        for r in range(max_run_length + 1):
            if log_posterior[r] > LOG_EPS / 2:
                log_pi[r] = _log_student_t_pdf(x, mu[r], kappa[r], alpha[r], beta[r])

        # 2. Compute growth probabilities (run length increases by 1)
        log_growth = log_posterior + log_pi + log_1m_hazard
        # 3. Compute change-point probability (run length resets to 0)
        log_cp = float(np.logaddexp.reduce(log_posterior + log_pi + log_hazard))

        # 4. New posterior: r=0 gets log_cp, r=k+1 gets log_growth[k]
        new_log_posterior = np.full(max_run_length + 1, LOG_EPS)
        new_log_posterior[0] = log_cp
        new_log_posterior[1:] = log_growth[:-1]

        # 5. Normalize
        log_evidence = float(np.logaddexp.reduce(new_log_posterior))
        new_log_posterior = new_log_posterior - log_evidence

        # 6. Update sufficient statistics:
        # For existing run lengths the stats grow by absorbing x; for r=0 it's the prior.
        new_kappa = np.empty_like(kappa)
        new_alpha = np.empty_like(alpha)
        new_beta = np.empty_like(beta)
        new_mu = np.empty_like(mu)

        new_kappa[0] = kappa0
        new_alpha[0] = alpha0
        new_beta[0] = beta0
        new_mu[0] = mu0

        new_kappa[1:] = kappa[:-1] + 1.0
        new_alpha[1:] = alpha[:-1] + 0.5
        new_mu[1:] = (kappa[:-1] * mu[:-1] + x) / (kappa[:-1] + 1.0)
        new_beta[1:] = beta[:-1] + (kappa[:-1] * (x - mu[:-1]) ** 2) / (2.0 * (kappa[:-1] + 1.0))

        log_posterior = new_log_posterior
        mu, kappa, alpha, beta = new_mu, new_kappa, new_alpha, new_beta

        # 7. Read out features
        posterior = np.exp(log_posterior)
        psum = posterior.sum()
        if psum > 0:
            posterior /= psum
        rlen = np.arange(max_run_length + 1, dtype=np.float64)
        expected_age[t] = float((posterior * rlen).sum())
        cp_prob[t] = float(posterior[0])
        # Entropy in nats, normalized by log(max_rl+1) so it's in [0, 1]
        nonzero = posterior > EPS
        if nonzero.any():
            h = float(-(posterior[nonzero] * np.log(posterior[nonzero])).sum())
            entropy[t] = h / np.log(max_run_length + 1)
        else:
            entropy[t] = 0.0

    return expected_age.astype(np.float32), cp_prob.astype(np.float32), entropy.astype(np.float32)


def _aggregate_to_scale(log_ret: pd.Series, window_bars: int) -> pd.Series:
    """Aggregate 1m log returns into rolling-sum series at the target scale.

    For BOCPD at 5m scale we feed every 1m bar's 5-bar cumulative return — this
    gives the model a smoothed view consistent with that timescale without
    actually downsampling (we want a per-1m-bar output for L1 consumption).
    """
    return log_ret.rolling(window_bars, min_periods=1).sum().fillna(0.0)


def compute(df_1m: pd.DataFrame) -> pd.DataFrame:
    close = df_1m["close"]
    log_ret = np.log(close / close.shift(1)).fillna(0.0)

    # Decimate for tractability: BOCPD's per-bar cost is O(max_run_length) ≈ 500 ops.
    # On 1.8M bars that's still ~900M ops per scale × 3 scales → manageable but slow.
    # We run on a stride and ffill — regime evolves slowly, this is fine.
    n = len(close)
    stride = 5

    out: dict[str, pd.Series] = {}
    for window_bars, label in [(5, "5m"), (60, "60m"), (240, "4h")]:
        signal = _aggregate_to_scale(log_ret, window_bars)
        # Standardize so BOCPD's Gaussian-like prior fits
        sig_std = signal.rolling(2000, min_periods=100).std().replace(0, np.nan).bfill().fillna(1.0)
        z = (signal / sig_std).fillna(0.0).clip(-5, 5).to_numpy()

        # Decimated input: every `stride` bars
        z_strided = z[::stride]
        age_s, cp_s, ent_s = _bocpd_run(z_strided)

        # Forward-fill back onto the full 1m index
        full_age = np.repeat(age_s, stride)[:n]
        full_cp = np.repeat(cp_s, stride)[:n]
        full_ent = np.repeat(ent_s, stride)[:n]
        if len(full_age) < n:
            # Pad with last value
            full_age = np.concatenate([full_age, np.full(n - len(full_age), full_age[-1])])
            full_cp = np.concatenate([full_cp, np.full(n - len(full_cp), full_cp[-1])])
            full_ent = np.concatenate([full_ent, np.full(n - len(full_ent), full_ent[-1])])

        out[f"bocpd_age_{label}"] = pd.Series(full_age, index=df_1m.index).astype(np.float32)
        out[f"bocpd_cp_prob_{label}"] = pd.Series(full_cp, index=df_1m.index).astype(np.float32)
        out[f"bocpd_entropy_{label}"] = pd.Series(full_ent, index=df_1m.index).astype(np.float32)

    return pd.DataFrame(out, index=df_1m.index)


FEATURE_COLUMNS = (
    "bocpd_age_5m",
    "bocpd_cp_prob_5m",
    "bocpd_entropy_5m",
    "bocpd_age_60m",
    "bocpd_cp_prob_60m",
    "bocpd_entropy_60m",
    "bocpd_age_4h",
    "bocpd_cp_prob_4h",
    "bocpd_entropy_4h",
)
