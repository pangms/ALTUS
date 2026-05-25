"""Hybrid Layer 1 model: ModernTCN + (Mamba | xLSTM) peer branches, fused
with cross-attention, multi-head output.

DIRECTION HEAD (2026-05-25, post-architectural-audit):
The classification head outputs a 3-class softmax over {long_wins, short_wins,
neither} — replacing the original two independent sigmoid heads. Independent
BCE heads collapsed into a degenerate "P(any TP hits)" volatility detector
(median |P_long - P_short| ≈ 0.003, top-20% candidates 99.9% long vs 0.1%
short). The 3-class softmax forces direction prediction because the classes
compete in the partition of probability mass.

For backward compatibility with downstream code (sim_pnl, production_sim,
layer2_train), we still expose `long_tp_prob` / `short_tp_prob` — these are
now the softmax slices for classes 0 and 1. They no longer can both be high
simultaneously; their semantic meaning is "given the model has an opinion,
which direction" rather than "is each side individually likely."

Regression heads (mfe/mae) are still present in the network but contribute
zero loss when reg_loss_weight=0 (the new default). They survive as
inference outputs for any downstream consumer that wants them.

The 'long_context' branch is configurable: choose 'mamba' (default, matches
the original spec) or 'xlstm' (A/B variant). Both have the same I/O shape.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from altus.config import ModelConfig
from altus.models.mamba import MambaEncoder
from altus.models.modern_tcn import ModernTCNEncoder
from altus.models.revin import RevIN
from altus.models.xlstm import XLSTMEncoder


@dataclass
class HybridOutputs:
    # 3-class direction logits (B, 3): [long_wins, short_wins, neither].
    # Softmax of these gives the calibrated direction distribution.
    direction_logits: torch.Tensor
    mfe_long: torch.Tensor        # (B,)
    mae_long: torch.Tensor
    mfe_short: torch.Tensor
    mae_short: torch.Tensor
    # Phase H: P(inflection) — auxiliary head. Disabled by default in the
    # post-audit config (use_inflection=False); kept for A/B-able re-enable.
    inflection_logit: torch.Tensor | None = None
    # Fusion embedding: the (B, fusion_hidden) vector right before the output
    # heads. Captures the model's compressed representation of the input
    # window. Optional — only populated if `return_embedding=True` is passed
    # to forward(). Used as additional input to Layer 2 meta-labeling.
    fusion_embedding: torch.Tensor | None = None

    @property
    def direction_probs(self) -> torch.Tensor:
        """(B, 3) softmax over [long_wins, short_wins, neither]."""
        return torch.softmax(self.direction_logits, dim=-1)

    @property
    def long_tp_prob(self) -> torch.Tensor:
        """P(long_wins). Backward-compatible alias for downstream sim code."""
        return self.direction_probs[:, 0]

    @property
    def short_tp_prob(self) -> torch.Tensor:
        """P(short_wins). Backward-compatible alias for downstream sim code."""
        return self.direction_probs[:, 1]

    @property
    def neither_prob(self) -> torch.Tensor:
        """P(neither side hits TP within the horizon)."""
        return self.direction_probs[:, 2]

    @property
    def long_tp_logit(self) -> torch.Tensor:
        """Back-compat: 'logit' here is the raw direction logit for class 0."""
        return self.direction_logits[:, 0]

    @property
    def short_tp_logit(self) -> torch.Tensor:
        return self.direction_logits[:, 1]

    @property
    def inflection_prob(self) -> torch.Tensor | None:
        if self.inflection_logit is None:
            return None
        return torch.sigmoid(self.inflection_logit)

    def as_dict(self) -> dict[str, torch.Tensor]:
        d = {
            "direction_logits": self.direction_logits,
            "long_tp_logit": self.long_tp_logit,
            "short_tp_logit": self.short_tp_logit,
            "mfe_long": self.mfe_long,
            "mae_long": self.mae_long,
            "mfe_short": self.mfe_short,
            "mae_short": self.mae_short,
        }
        if self.inflection_logit is not None:
            d["inflection_logit"] = self.inflection_logit
        if self.fusion_embedding is not None:
            d["fusion_embedding"] = self.fusion_embedding
        return d


class _AttentionPool(nn.Module):
    """Learned query pooling: collapses (B, L, D) -> (B, D) via attention over time."""

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.attn = nn.MultiheadAttention(d_model, num_heads=4, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q = self.query.expand(x.shape[0], -1, -1)   # (B, 1, D)
        out, _ = self.attn(q, x, x, need_weights=False)
        return out.squeeze(1)


class HybridLayer1(nn.Module):
    """ModernTCN + (Mamba | xLSTM | none) peer branches with fused multi-head output.

    `long_context='none'` skips the long-context branch entirely (TCN-only). Useful
    when the sequential-scan branches are too slow on the available hardware — gives
    a strong parallel baseline that validates the rest of the pipeline.
    """

    def __init__(self, cfg: ModelConfig, long_context: str = "mamba") -> None:
        super().__init__()
        assert long_context in ("mamba", "xlstm", "none"), f"unknown long_context: {long_context}"
        self.cfg = cfg
        self.long_context_kind = long_context

        # Optional RevIN input normalization. Applied to the raw window before
        # any encoder. Cheap, addresses train/live distribution shift directly.
        self.revin = (
            RevIN(num_features=cfg.n_features_in, affine=cfg.revin_affine)
            if cfg.use_revin
            else None
        )

        # Peer branch 1: ModernTCN (local patterns) — always present
        self.tcn = ModernTCNEncoder(
            in_features=cfg.n_features_in,
            d_model=cfg.d_model,
            patch_size=cfg.tcn_patch_size,
            n_blocks=cfg.tcn_n_blocks,
            kernel_size=cfg.tcn_kernel_size,
            dw_expansion=cfg.tcn_dw_expansion,
            pw_expansion=cfg.tcn_pw_expansion,
            dropout=cfg.tcn_dropout,
        )
        self.tcn_pool = _AttentionPool(cfg.d_model)

        # Peer branch 2: long-context (optional)
        if long_context == "mamba":
            self.long_ctx = MambaEncoder(
                in_features=cfg.n_features_in,
                d_model=cfg.d_model,
                n_blocks=cfg.mamba_n_blocks,
                d_state=cfg.mamba_d_state,
                d_conv=cfg.mamba_d_conv,
                expand=cfg.mamba_expand,
                dropout=cfg.mamba_dropout,
            )
            self.ctx_pool = _AttentionPool(cfg.d_model)
            fusion_input_dim = 2 * cfg.d_model
        elif long_context == "xlstm":
            self.long_ctx = XLSTMEncoder(
                in_features=cfg.n_features_in,
                d_model=cfg.d_model,
                n_blocks=cfg.xlstm_n_blocks,
                n_heads=cfg.xlstm_n_heads,
                dropout=cfg.xlstm_dropout,
            )
            self.ctx_pool = _AttentionPool(cfg.d_model)
            fusion_input_dim = 2 * cfg.d_model
        else:  # "none" — TCN-only
            self.long_ctx = None
            self.ctx_pool = None
            fusion_input_dim = cfg.d_model

        # Fusion MLP
        self.fusion = nn.Sequential(
            nn.Linear(fusion_input_dim, cfg.fusion_hidden),
            nn.GELU(),
            nn.Dropout(cfg.fusion_dropout),
            nn.Linear(cfg.fusion_hidden, cfg.fusion_hidden),
            nn.GELU(),
            nn.Dropout(cfg.fusion_dropout),
        )

        # Output heads. cfg.n_class_heads = 3 under the new direction-softmax
        # design (long_wins, short_wins, neither). We assert here to fail loud
        # if someone tries to revert to the old dual-BCE config without
        # rewriting the loss in altus.training.train.
        assert cfg.n_class_heads == 3, (
            f"3-class direction head expected; got n_class_heads={cfg.n_class_heads}. "
            f"Direction is now a single softmax-3 (long/short/neither) — see hybrid.py docstring."
        )
        self.head_cls = nn.Linear(cfg.fusion_hidden, cfg.n_class_heads)
        self.head_reg = nn.Linear(cfg.fusion_hidden, cfg.n_reg_heads) if cfg.n_reg_heads > 0 else None
        # Phase H: auxiliary inflection head — small 2-layer MLP over fusion
        # embedding. Predicts P(price resolves AGAINST recent direction).
        if cfg.use_inflection:
            self.head_inflection = nn.Sequential(
                nn.Linear(cfg.fusion_hidden, cfg.fusion_hidden // 2),
                nn.GELU(),
                nn.Dropout(cfg.fusion_dropout),
                nn.Linear(cfg.fusion_hidden // 2, 1),
            )
        else:
            self.head_inflection = None

    def forward(self, x: torch.Tensor, return_embedding: bool = False) -> HybridOutputs:
        if self.revin is not None:
            x = self.revin(x, mode="norm")
        h_tcn = self.tcn(x)
        z_tcn = self.tcn_pool(h_tcn)

        if self.long_ctx is None:
            z = z_tcn
        else:
            h_ctx = self.long_ctx(x)
            z_ctx = self.ctx_pool(h_ctx)
            z = torch.cat([z_tcn, z_ctx], dim=-1)

        # Fusion MLP → (B, fusion_hidden). This is the embedding we expose to L2.
        z = self.fusion(z)
        direction_logits = self.head_cls(z)   # (B, 3) — [long_wins, short_wins, neither]
        if self.head_reg is not None:
            reg = torch.nn.functional.softplus(self.head_reg(z))
            mfe_long = reg[:, 0]
            mae_long = reg[:, 1]
            mfe_short = reg[:, 2]
            mae_short = reg[:, 3]
        else:
            # Regression heads disabled — emit zeros so downstream consumers
            # that read these fields still get a tensor of the right shape.
            zero = torch.zeros(direction_logits.shape[0], device=direction_logits.device)
            mfe_long = mae_long = mfe_short = mae_short = zero
        inflection_logit = self.head_inflection(z).squeeze(-1) if self.head_inflection is not None else None

        return HybridOutputs(
            direction_logits=direction_logits,
            mfe_long=mfe_long,
            mae_long=mae_long,
            mfe_short=mfe_short,
            mae_short=mae_short,
            inflection_logit=inflection_logit,
            fusion_embedding=z if return_embedding else None,
        )


def build_hybrid(n_features: int, long_context: str = "mamba", **overrides) -> HybridLayer1:
    """Convenience constructor that fills n_features_in and applies overrides."""
    cfg = ModelConfig(n_features_in=n_features)
    for k, v in overrides.items():
        if not hasattr(cfg, k):
            raise ValueError(f"unknown ModelConfig field: {k}")
        setattr(cfg, k, v)
    cfg.n_features_in = n_features  # ensure overrides didn't blank this
    return HybridLayer1(cfg, long_context=long_context)
