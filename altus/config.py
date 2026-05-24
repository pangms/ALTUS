"""Central configuration for ALTUS.

Single source of truth for instrument constants, label parameters, model
hyperparameters, and run defaults. Anything that another module reads should
live here so we don't scatter magic numbers.

LAYER 1 v2 FINAL (locked 2026-05-24 after Phase A sweep + RevIN A/B):
  Architecture:  ModernTCN encoder only (no Mamba/xLSTM/PatchTST/Kronos)
  Input features: 52 multi-TF price features + structural features from
                  vol, trend, anomaly Phase A families (22 features) = 74 total
  Normalization:  RevIN per-instance + isotonic post-calibration
  Heads:          2 binary (long_tp, short_tp) + 4 regression (MFE/MAE)
  Loss:           Multi-task BCE + Huber with label smoothing
  Regularization: dropout 0.30, weight_decay 1e-4, label smoothing 0.05
  Performance:    OOS mean AUC 0.638, top-1% win rate 0.506 (3yr/3fold)

DROPPED FROM v2 (didn't earn their place per empirical bar):
  - session, exhaust structural families
  - Mamba/xLSTM long-context branches (CUDA-kernel speed issues)
  - PatchTST (Kronos planned as replacement, then deferred)
  - Kronos (deferred — Layer 2 has higher expected lift)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"
ARTIFACT_DIR = REPO_ROOT / "artifacts"
ARTIFACT_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# Instrument constants — MNQ (Micro E-mini Nasdaq-100)
# ---------------------------------------------------------------------------
TICK_SIZE = 0.25          # MNQ minimum price increment, in index points
TICK_VALUE_USD = 0.50     # USD per tick (MNQ is $2/point, 0.25 pt/tick)
POINT_VALUE_USD = 2.0     # USD per index point

# Cost model used when running the standalone Layer 1 trading simulation.
# These are reasonable retail-broker estimates for MNQ on TopStep-style platforms.
COMMISSION_RT_USD = 3.50   # round-trip commission, USD
SLIPPAGE_RT_POINTS = 0.5   # round-trip slippage budget, index points (~2 ticks)


# ---------------------------------------------------------------------------
# Label parameters — triple-barrier
# ---------------------------------------------------------------------------
TP_POINTS = 30.0   # take profit, index points
SL_POINTS = 30.0   # stop loss, index points  (1:1 RR by design)
LABEL_HORIZON_BARS = 60  # H — max bars to a barrier before timeout (1 hour on 1m)


# ---------------------------------------------------------------------------
# Session windows (UTC). MNQ trades nearly 24/5 on CME Globex.
# RTH = Regular Trading Hours, US equity session 09:30-16:00 ET.
# For 2024-2026 these are 13:30-20:00 UTC (EDT) or 14:30-21:00 UTC (EST).
# We train on full ETH but the trading-simulation gate restricts to liquid hours.
# ---------------------------------------------------------------------------
LIQUID_HOURS_UTC = [
    # (start_hour, end_hour) in UTC. Inclusive start, exclusive end.
    (13, 21),   # RTH overlap covering both EST/EDT US equity session
    (8, 10),    # London/Europe open overlap — first hour of Globex re-engagement
]


# ---------------------------------------------------------------------------
# Splits
# ---------------------------------------------------------------------------
OOS_LOCKBOX_MONTHS = 6           # last N months untouched until acceptance test
EMBARGO_BARS = LABEL_HORIZON_BARS  # purge >= label horizon to prevent leakage
N_WALK_FORWARD_FOLDS = 5         # walk-forward folds inside the dev set


# ---------------------------------------------------------------------------
# Feature pipeline
# ---------------------------------------------------------------------------
TIMEFRAMES_MIN = (1, 3, 5, 15)  # multi-timeframe stack
SEQ_LEN_BARS = 240               # context window length the model sees (1m bars)
                                 # 240 = 4 hours of 1m context, captures session memory
ROLL_NORM_WINDOW = 1440          # 1 day of 1m bars for rolling z-score normalization


# Layer 1 v2 final: survivors of Phase A sweep. Comma-separated string passed
# to StructuralSpec.from_string. Used by train_layer1_final.py.
LAYER1_V2_STRUCTURAL_FAMILIES = "vol,trend,anomaly"


# ---------------------------------------------------------------------------
# Layer 1 Model defaults
# ---------------------------------------------------------------------------
@dataclass
class ModelConfig:
    # Shared
    d_model: int = 96
    n_features_in: int = 0           # filled at build time from feature pipeline
    seq_len: int = SEQ_LEN_BARS

    # ModernTCN branch
    tcn_patch_size: int = 8
    tcn_n_blocks: int = 3
    tcn_kernel_size: int = 7
    tcn_dw_expansion: int = 2
    tcn_pw_expansion: int = 4

    # Mamba branch
    mamba_n_blocks: int = 3
    mamba_d_state: int = 16
    mamba_d_conv: int = 4
    mamba_expand: int = 2

    # xLSTM branch (alternative)
    xlstm_n_blocks: int = 3
    xlstm_n_heads: int = 4

    # Fusion
    fusion_hidden: int = 192
    fusion_dropout: float = 0.30   # was 0.15 — increased to fight overfitting
    tcn_dropout: float = 0.30      # was hardcoded 0.10 in modern_tcn.py
    mamba_dropout: float = 0.10
    xlstm_dropout: float = 0.10

    # Output heads — 6 learned outputs:
    #   2 binary (long_tp, short_tp), 4 regression (mfe_long, mae_long, mfe_short, mae_short)
    n_class_heads: int = 2
    n_reg_heads: int = 4

    # RevIN — Reversible Instance Normalization wrapping the input window.
    # Per-instance z-score + learnable affine. Addresses train/live distribution
    # shift (price/vol regimes change over the years). See altus/models/revin.py.
    use_revin: bool = False
    revin_affine: bool = True


# ---------------------------------------------------------------------------
# Layer 2 Model defaults — meta-labeling network
# ---------------------------------------------------------------------------
@dataclass
class Layer2Config:
    """Meta-labeling network. Small MLP that scores Layer 1 candidate signals.

    Input contract: feature vector built from Layer 1's 6 outputs + derived
    aggregates + structural features at the same bar. See altus/models/layer2.py.
    Output: P(profitable_trade) — calibrated, optionally conformal-wrapped.
    """
    input_dim: int = 0          # filled at build time from training data
    hidden_dim: int = 64
    n_hidden_layers: int = 2
    dropout: float = 0.30
    use_attention_pool: bool = False  # over context features; simple MLP for v1


@dataclass
class Layer2TrainConfig:
    batch_size: int = 1024
    n_epochs: int = 30
    lr: float = 1e-3
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    early_stop_patience: int = 5
    label_smoothing: float = 0.05
    cal_holdout_frac: float = 0.15
    # Selection of Layer 1 candidates to train Layer 2 on. Two modes:
    #   "top_k_percent": train Layer 2 only on bars where Layer 1's max prob is
    #                    in the top K% — these are the "would have traded" candidates
    #   "all_bars":      train on every bar (treats Layer 1's prob as a feature)
    # For meta-labeling, "top_k_percent" is more honest because it focuses on the
    # candidate-selection problem we actually face at inference.
    candidate_mode: str = "top_k_percent"
    candidate_top_k: float = 0.20  # top 20% of Layer 1 signals as candidates


# ---------------------------------------------------------------------------
# Training defaults (Layer 1)
# ---------------------------------------------------------------------------
@dataclass
class TrainConfig:
    batch_size: int = 256
    n_epochs: int = 20
    lr: float = 1e-3
    weight_decay: float = 1e-4     # was 1e-5 — stronger L2 to fight overfitting
    grad_clip: float = 1.0
    label_smoothing: float = 0.05  # softens BCE targets (1 -> 0.95, 0 -> 0.05) — reduces memorization
    input_feature_dropout: float = 0.05  # randomly zero this fraction of features per batch in training
    # Multi-task loss weights: classification dominates because that's what we trade on,
    # but the regression heads serve as auxiliary regularization for the shared encoder.
    cls_loss_weight: float = 1.0
    reg_loss_weight: float = 0.2
    early_stop_patience: int = 4
    val_metric: str = "mean_auc"     # what early stopping watches
    num_workers: int = 0              # MPS works best single-process
    device: str = "mps"               # falls back to cpu if MPS unavailable
    seed: int = 1337


# ---------------------------------------------------------------------------
# Acceptance criteria (Layer 1 -> Layer 2 gate)
# ---------------------------------------------------------------------------
@dataclass
class AcceptanceCriteria:
    min_auc_per_side: float = 0.54
    min_brier_improvement: float = 0.05
    top_decile_min_winrate: float = 0.58
    min_sharpe: float = 0.4
    max_drawdown_pct: float = 0.15
    min_pct_positive_months: float = 0.65


DEFAULT_MODEL_CFG = ModelConfig()
DEFAULT_TRAIN_CFG = TrainConfig()
DEFAULT_LAYER2_CFG = Layer2Config()
DEFAULT_LAYER2_TRAIN_CFG = Layer2TrainConfig()
DEFAULT_ACCEPTANCE = AcceptanceCriteria()
