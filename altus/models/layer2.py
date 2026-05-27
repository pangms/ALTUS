"""Layer 2 — meta-labeling network.

ROLE IN THE STACK:
  Layer 1 (price-action specialist) generates directional candidates with
  probabilities. Layer 2 sits on top and answers a different question:
  "given this Layer 1 candidate signal + the broader context, is it actually
  worth trading?" It's a learned trade-gate.

WHY THIS EXISTS:
  Layer 1's top-decile win rate is currently 0.50 ish. To hit our 70% win
  rate target, we need a way to filter Layer 1's candidates down to the
  high-quality subset. Layer 2 is exactly that filter. It's trained to
  predict whether a Layer 1 signal would have been profitable, given
  features Layer 1 didn't fully see (or didn't compress into its 6 outputs).

INPUTS (per candidate signal):
  • Layer 1's 6 outputs (P_long, P_short, MFE_long, MAE_long, MFE_short, MAE_short)
  • Derived features from those 6:
      - signal_direction: +1 if long is winning side, -1 if short
      - signal_strength : max(P_long, P_short)
      - signal_margin   : |P_long - P_short|
      - signal_entropy  : -P*log(P) - (1-P)*log(1-P) using max-side P
      - expected_R      : (max-side MFE) / (max-side MAE + eps)
  • Structural features at the same bar (vol, trend, anomaly families)
  • Time features (sin/cos hour, sin/cos day-of-week)
  Total ~30 features per signal.

OUTPUT:
  Single sigmoid logit → P(profitable_trade). Calibrated post-hoc with
  isotonic regression. Optional conformal prediction wrapper for statistical
  trade-gating guarantees.

ARCHITECTURE:
  Small MLP (2-3 layers, ~5-15K params). Deliberately tiny to avoid
  overfitting on the small candidate sample. Not a transformer, not an
  attention model, not a sequence model — just a thoughtful tabular
  classifier on rich features.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from altus.config import Layer2Config


EPS = 1e-9


# ---------------------------------------------------------------------------
# Feature derivation
# ---------------------------------------------------------------------------

LAYER1_OUTPUT_KEYS = (
    "long_tp_prob", "short_tp_prob",
    "mfe_long", "mae_long", "mfe_short", "mae_short",
)

# These are the column names Layer 2 sees per candidate. Order matters and is
# preserved end-to-end (training, inference, conformal calibration).
LAYER2_INPUT_FEATURES = (
    # --- From Layer 1's 6 outputs (raw) ---
    "l1_long_tp_prob",
    "l1_short_tp_prob",
    "l1_mfe_long",
    "l1_mae_long",
    "l1_mfe_short",
    "l1_mae_short",
    # --- Derived from L1 outputs ---
    "derived_direction",       # +1 long, -1 short
    "derived_strength",        # max(P_long, P_short)
    "derived_margin",          # |P_long - P_short|
    "derived_entropy",         # binary entropy of the winning side's prob
    "derived_expected_r",      # winning side's MFE / MAE+eps
    # --- Time context ---
    "time_hour_sin",
    "time_hour_cos",
    "time_dow_sin",
    "time_dow_cos",
    # --- Volatility regime (Phase A vol family) ---
    "vol_realized_5m",
    "vol_realized_30m",
    "vol_realized_4h",
    "vol_realized_1d",
    "vol_of_vol",
    "vol_hurst",
    "vol_percentile_60d",
    "vol_regime_score",
    # --- Multi-TF trend (Phase A trend family) ---
    "trend_4h_slope",
    "trend_4h_hurst",
    "trend_1d_slope",
    "trend_1d_hurst",
    "trend_1w_slope",
    "trend_1w_hurst",
    "trend_alignment",
    "trend_strength",
    # --- Anomaly ---
    "anomaly_mahalanobis",
    # --- BOCPD regime posterior at 3 timescales (Phase F, Q19/Q20) ---
    # Added 2026-05-24 — closes the "L2 has no regime context" gap flagged
    # in the architecture audit. Requires the bocpd feature family to be
    # included in the L1 run; absent columns are zero-filled by
    # build_layer2_input's missing-column fallback.
    "bocpd_age_5m", "bocpd_cp_prob_5m", "bocpd_entropy_5m",
    "bocpd_age_60m", "bocpd_cp_prob_60m", "bocpd_entropy_60m",
    "bocpd_age_4h", "bocpd_cp_prob_4h", "bocpd_entropy_4h",
    # --- L2 confidence modulators (2026-05-26 — FRAMEWORK.md F/G tier) ---
    # Path clearance: bidirectional booster — bigger clearance = more conviction.
    "pc_clearance_above_atr", "pc_clearance_below_atr",
    "pc_obstacle_above_strength", "pc_obstacle_below_strength",
    "pc_clearance_asymmetry", "pc_clearance_min_atr",
    # Stop pool: fuel detector — bigger pool = bigger expected magnitude.
    "sp_pool_above_size_atr", "sp_pool_below_size_atr",
    "sp_trigger_distance_above_atr", "sp_trigger_distance_below_atr",
    "sp_pool_imminent",
    # Setup confluence: multi-setup alignment.
    "scf_long_count", "scf_short_count", "scf_consensus_score", "scf_total_active",
    # Cross-asset setup confirmation: NQ/ES alignment.
    "cac_es_direction_proxy", "cac_nq_es_aligned",
    "cac_lead_lag_signed", "cac_divergence_active",
    # Vol regime sweet-spot per setup (one feature per setup + average).
    "vss_match_sfs", "vss_match_sfa", "vss_match_sld", "vss_match_orb",
    "vss_match_svwap", "vss_match_spb", "vss_match_scomp", "vss_match_seod",
    "vss_avg_match",
    # Time-of-day fitness per setup.
    "tof_fit_sfs", "tof_fit_sfa", "tof_fit_sld", "tof_fit_orb",
    "tof_fit_svwap", "tof_fit_spb", "tof_fit_scomp", "tof_fit_seod",
    "tof_avg_fit",
)


def derive_signal_features(layer1_outputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Compute the 5 derived features from Layer 1's 6 outputs.

    All inputs assumed to be 1-D numpy arrays of equal length, one per candidate.
    """
    pL = np.asarray(layer1_outputs["long_tp_prob"], dtype=np.float64)
    pS = np.asarray(layer1_outputs["short_tp_prob"], dtype=np.float64)
    mfeL = np.asarray(layer1_outputs["mfe_long"], dtype=np.float64)
    maeL = np.asarray(layer1_outputs["mae_long"], dtype=np.float64)
    mfeS = np.asarray(layer1_outputs["mfe_short"], dtype=np.float64)
    maeS = np.asarray(layer1_outputs["mae_short"], dtype=np.float64)

    direction = np.where(pL >= pS, 1.0, -1.0)
    strength = np.maximum(pL, pS)
    margin = np.abs(pL - pS)
    # 3-class softmax entropy (long_wins, short_wins, neither). Under the
    # post-pivot direction softmax, long_p and short_p are slices of a simplex
    # with implicit neither_p = 1 - long_p - short_p. True entropy proxies
    # Q30 (model confidence). High entropy = uncertain across all 3 outcomes.
    pN = np.clip(1.0 - pL - pS, EPS, 1.0)
    pL_c = np.clip(pL, EPS, 1.0)
    pS_c = np.clip(pS, EPS, 1.0)
    entropy = -(pL_c * np.log(pL_c) + pS_c * np.log(pS_c) + pN * np.log(pN))
    # Expected R-multiple from the winning side's predicted excursion
    winning_mfe = np.where(direction > 0, mfeL, mfeS)
    winning_mae = np.where(direction > 0, maeL, maeS)
    expected_r = winning_mfe / (winning_mae + EPS)

    return {
        "derived_direction": direction.astype(np.float32),
        "derived_strength": strength.astype(np.float32),
        "derived_margin": margin.astype(np.float32),
        "derived_entropy": entropy.astype(np.float32),
        "derived_expected_r": expected_r.astype(np.float32),
    }


def time_features_from_index(index: pd.DatetimeIndex) -> dict[str, np.ndarray]:
    """Sin/cos of hour-of-day and day-of-week. Same as Phase A session family."""
    hour_utc = (index.hour + index.minute / 60.0).to_numpy(dtype=np.float32)
    dow = index.dayofweek.to_numpy(dtype=np.float32)
    return {
        "time_hour_sin": np.sin(2 * np.pi * hour_utc / 24.0).astype(np.float32),
        "time_hour_cos": np.cos(2 * np.pi * hour_utc / 24.0).astype(np.float32),
        "time_dow_sin": np.sin(2 * np.pi * dow / 7.0).astype(np.float32),
        "time_dow_cos": np.cos(2 * np.pi * dow / 7.0).astype(np.float32),
    }


def build_layer2_input(
    layer1_outputs: dict[str, np.ndarray],
    structural_features: pd.DataFrame,
    index: pd.DatetimeIndex,
) -> pd.DataFrame:
    """Combine Layer 1 outputs + derived + structural + time into Layer 2's input matrix.

    Returns a DataFrame with columns in `LAYER2_INPUT_FEATURES` order, indexed
    by `index`. Caller is responsible for filtering to candidate signals.
    """
    derived = derive_signal_features(layer1_outputs)
    time_feats = time_features_from_index(index)

    data: dict[str, np.ndarray] = {}
    # Layer 1 outputs (prefix with l1_)
    for key in LAYER1_OUTPUT_KEYS:
        data[f"l1_{key.replace('_prob', '_tp_prob') if key.endswith('_tp_prob') else key}"] = (
            np.asarray(layer1_outputs[key], dtype=np.float32)
        )
    # Fix the naming: l1_long_tp_prob, l1_short_tp_prob, l1_mfe_long, etc.
    data = {
        "l1_long_tp_prob": np.asarray(layer1_outputs["long_tp_prob"], dtype=np.float32),
        "l1_short_tp_prob": np.asarray(layer1_outputs["short_tp_prob"], dtype=np.float32),
        "l1_mfe_long": np.asarray(layer1_outputs["mfe_long"], dtype=np.float32),
        "l1_mae_long": np.asarray(layer1_outputs["mae_long"], dtype=np.float32),
        "l1_mfe_short": np.asarray(layer1_outputs["mfe_short"], dtype=np.float32),
        "l1_mae_short": np.asarray(layer1_outputs["mae_short"], dtype=np.float32),
        **derived,
        **time_feats,
    }
    df = pd.DataFrame(data, index=index)

    # Join structural features (vol, trend, anomaly). Drop the causal shift the
    # structural pipeline adds, since we're already aligned to entry timestamps.
    struct_cols = [c for c in LAYER2_INPUT_FEATURES if c not in df.columns]
    avail_struct = structural_features.reindex(index)
    for col in struct_cols:
        if col in avail_struct.columns:
            df[col] = avail_struct[col].astype(np.float32).to_numpy()
        else:
            # Missing structural feature — fill with 0 and let model learn
            df[col] = np.zeros(len(index), dtype=np.float32)

    return df[list(LAYER2_INPUT_FEATURES)]


# ---------------------------------------------------------------------------
# The Layer 2 model itself
# ---------------------------------------------------------------------------

class Layer2MetaLabeler(nn.Module):
    """Small MLP meta-labeler. Tabular classifier — no sequence, no attention.

    Architecturally simple by design — meta-labeling sample sizes are small
    (only candidate signals, not all bars) so we keep the model tight to
    avoid overfitting. Validated empirically per our standing rule.

    Optional Layer 1 fusion embedding path: if cfg.embedding_dim > 0, the
    forward signature changes to accept (hand_crafted_features, embedding).
    The embedding is projected down to cfg.embedding_project_dim via a small
    Linear, then concatenated with the hand-crafted features before the MLP.
    """

    def __init__(self, cfg: Layer2Config) -> None:
        super().__init__()
        if cfg.input_dim <= 0:
            raise ValueError("Layer2Config.input_dim must be set before constructing the model")
        self.cfg = cfg

        # Optional embedding projector
        self.embedding_proj: nn.Module | None = None
        effective_input_dim = cfg.input_dim
        if cfg.embedding_dim > 0:
            self.embedding_proj = nn.Sequential(
                nn.Linear(cfg.embedding_dim, cfg.embedding_project_dim),
                nn.GELU(),
                nn.Dropout(cfg.dropout),
            )
            effective_input_dim += cfg.embedding_project_dim

        layers: list[nn.Module] = []
        in_dim = effective_input_dim
        for _ in range(cfg.n_hidden_layers):
            layers += [
                nn.Linear(in_dim, cfg.hidden_dim),
                nn.GELU(),
                nn.Dropout(cfg.dropout),
            ]
            in_dim = cfg.hidden_dim

        # Bottleneck to half-size hidden before output — helps generalization
        layers += [
            nn.Linear(in_dim, max(cfg.hidden_dim // 2, 8)),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(max(cfg.hidden_dim // 2, 8), 1),
        ]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, embedding: torch.Tensor | None = None) -> torch.Tensor:
        """
        x: (B, input_dim) hand-crafted features
        embedding: (B, embedding_dim) Layer 1 fusion embedding, required iff cfg.embedding_dim > 0
        Returns logits (B,); apply sigmoid for probabilities.
        """
        if self.embedding_proj is not None:
            if embedding is None:
                raise ValueError("Layer2 was configured with embedding_dim>0 but no embedding was passed")
            emb_proj = self.embedding_proj(embedding)
            x = torch.cat([x, emb_proj], dim=-1)
        return self.net(x).squeeze(-1)

    @torch.no_grad()
    def predict_proba(self, x: torch.Tensor, embedding: torch.Tensor | None = None) -> torch.Tensor:
        """Convenience: returns sigmoid(forward) as probabilities."""
        return torch.sigmoid(self.forward(x, embedding=embedding))


def build_layer2(input_dim: int, **overrides) -> Layer2MetaLabeler:
    """Convenience constructor."""
    cfg = Layer2Config(input_dim=input_dim)
    for k, v in overrides.items():
        if not hasattr(cfg, k):
            raise ValueError(f"unknown Layer2Config field: {k}")
        setattr(cfg, k, v)
    cfg.input_dim = input_dim
    return Layer2MetaLabeler(cfg)
