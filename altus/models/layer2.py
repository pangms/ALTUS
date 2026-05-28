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
    # --- C-tier predictive head outputs (2026-05-27 — fixes audit Disconnection #1) ---
    # Without these, the magnitude/path/clearance forecasts the predictive pivot
    # exists to deliver are trained but never reach the trade decision. They were
    # already extracted in _predict, used by the predictive-vs-pacing diagnostics,
    # and saved into the val_preds.npz — just not wired through to L2 input.
    "l1_return_H15_pred",
    "l1_return_H60_pred",
    "l1_path_shape_p0",         # continuation prob
    "l1_path_shape_p1",         # revert prob
    "l1_path_shape_p2",         # chop prob
    "l1_clears_level_prob",     # P(price clears ≥1 ATR forward)
    "l1_inflection_prob",       # P(price resolves against recent direction)
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
    # --- Per-setup detector outputs (2026-05-26 — Disconnection 2 fix) ---
    # Without these the L2 MLP cannot see which setup is firing, making
    # "setup-conditional WR" mathematically impossible. Each setup exposes:
    #   active   ∈ {0,1}   — fired this bar
    #   strength ∈ [0,1]   — soft confidence
    #   direction∈ {-1,0,+1} — proposed side
    #   + 2 setup-specific state features for richer context.
    # sfs — failed sweep (A3)
    "sfs_active", "sfs_strength", "sfs_direction",
    "sfs_age_bars", "sfs_level_type",
    # sfa — failed auction (A6)
    "sfa_active", "sfa_strength", "sfa_direction",
    "sfa_touch_count", "sfa_age_bars",
    # sld — level defense (A8)
    "sld_active", "sld_strength", "sld_direction",
    "sld_defense_count", "sld_dist_to_level_atr",
    # orb — Open Range Breakout (A1)
    "orb_active", "orb_strength", "orb_direction",
    "orb_breakout_age", "orb_range_atr",
    # svwap — VWAP rejection/reclaim (A2)
    "svwap_active", "svwap_strength", "svwap_direction",
    "svwap_dist_atr", "svwap_holds_count",
    # spb — pullback continuation (A4)
    "spb_active", "spb_strength", "spb_direction",
    "spb_pullback_depth", "spb_dist_to_ema21_atr",
    # scomp — compression breakout (A5)
    "scomp_active", "scomp_strength", "scomp_direction",
    "scomp_compression_ratio", "scomp_expansion_magnitude",
    # seod — EOD reversion (A7)
    "seod_active", "seod_strength", "seod_direction",
    "seod_band_position", "seod_mins_until_close",
    # --- Surfer bridge: per-setup × HTF agreement (2026-05-27 audit fix) ---
    # Engineers the setup×higher-timeframe interaction into the signal so L2
    # sees "ORB-long, 4h-aligned +0.6" instead of a timeframe-naked setup.
    # The thing the surfer principle is actually about.
    "sfs_htf_agree", "sfa_htf_agree", "sld_htf_agree", "orb_htf_agree",
    "svwap_htf_agree", "spb_htf_agree", "scomp_htf_agree", "seod_htf_agree",
    "shtf_primary_agree", "shtf_primary_60m",
    "shtf_aligned_count", "shtf_opposed_count",
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
    n_rows = len(index)
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
    # C-tier predictive head outputs — fixes audit Disconnection #1 (2026-05-27).
    # These are optional (older checkpoints may lack them); fall back to neutral
    # zero (regression) or 1/3 (path_shape softmax) when absent so L2 train
    # doesn't crash on legacy npz files.
    ctier_map = {
        "l1_return_H15_pred": ("return_H15_pred", 0.0),
        "l1_return_H60_pred": ("return_H60_pred", 0.0),
        "l1_path_shape_p0":   ("path_shape_p0",   1.0 / 3.0),
        "l1_path_shape_p1":   ("path_shape_p1",   1.0 / 3.0),
        "l1_path_shape_p2":   ("path_shape_p2",   1.0 / 3.0),
        "l1_clears_level_prob": ("clears_level_prob", 0.5),
        "l1_inflection_prob": ("inflection_prob", 0.5),
    }
    for l2_col, (l1_key, default) in ctier_map.items():
        if l1_key in layer1_outputs:
            data[l2_col] = np.asarray(layer1_outputs[l1_key], dtype=np.float32)
        else:
            data[l2_col] = np.full(n_rows, default, dtype=np.float32)
    df = pd.DataFrame(data, index=index)

    # Join structural features (vol, trend, anomaly). The structural matrix is
    # already causally-shifted by build_structural_features (X.shift(1) — row T
    # contains info from bar T-1), and we reindex to entry timestamps, so the
    # entry-bar features are correctly forward-blind.
    # Build the missing-column block en-bloc via pd.concat (was per-column
    # df[col] = ... which fragments the DataFrame and is O(N²) for 80+ cols).
    struct_cols = [c for c in LAYER2_INPUT_FEATURES if c not in df.columns]
    avail_struct = structural_features.reindex(index)
    struct_block: dict[str, np.ndarray] = {}
    n = len(index)
    for col in struct_cols:
        if col in avail_struct.columns:
            struct_block[col] = avail_struct[col].astype(np.float32).to_numpy()
        else:
            # Missing structural feature — fill with 0 and let model learn
            struct_block[col] = np.zeros(n, dtype=np.float32)
    if struct_block:
        df = pd.concat([df, pd.DataFrame(struct_block, index=index)], axis=1)

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
