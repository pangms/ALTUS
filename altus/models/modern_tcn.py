"""ModernTCN encoder branch.

Reference: Luo et al. 2024, "ModernTCN: A Modern Pure Convolution Structure for
General Time Series Analysis." The key ideas are: (1) patch embedding to reduce
sequence length, (2) depthwise convolutions for per-channel temporal mixing,
(3) pointwise convolutions for cross-channel mixing, (4) residual connections.

For ALTUS Layer 1 we use it as the "local pattern" branch — strong at capturing
short price-action structures (engulfings, range expansions, micro-trends) at
multiple effective scales via stacked dilated kernels.

Note on causality: the model predicts at the END of the sequence given the
whole past-window input. The input itself is already all-past, so non-causal
(centered) convolutions are valid here and use the receptive field better.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class PatchEmbed(nn.Module):
    """Non-overlapping 1D patch embedding: (B, F, L) -> (B, D, L/P)."""

    def __init__(self, in_features: int, d_model: int, patch_size: int) -> None:
        super().__init__()
        self.proj = nn.Conv1d(in_features, d_model, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, F, L) where L is sequence length, F is features
        h = self.proj(x)                # (B, D, L/P)
        h = h.transpose(1, 2)           # (B, L/P, D)
        h = self.norm(h)
        return h.transpose(1, 2)        # (B, D, L/P) for downstream conv ops


class TCNBlock(nn.Module):
    """ModernTCN block: depthwise conv + per-feature pointwise expansion.

    All convs are centered (non-causal) — see module docstring.
    """

    def __init__(
        self,
        d_model: int,
        kernel_size: int = 7,
        dw_expansion: int = 2,
        pw_expansion: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        pad = kernel_size // 2

        # Depthwise temporal mixing — each channel evolves independently.
        # Expanding channels here (groups stays at d_model so it's still depthwise)
        # gives extra temporal capacity without inflating cross-channel params.
        d_dw = d_model * dw_expansion
        self.dw_proj = nn.Conv1d(d_model, d_dw, kernel_size=1)
        self.dw_conv = nn.Conv1d(d_dw, d_dw, kernel_size=kernel_size, padding=pad, groups=d_dw)
        self.dw_back = nn.Conv1d(d_dw, d_model, kernel_size=1)
        self.norm1 = nn.GroupNorm(1, d_model)  # acts like LayerNorm over (C, L)

        # Pointwise channel mixing — classic inverted-bottleneck a la ConvNeXt.
        d_pw = d_model * pw_expansion
        self.pw1 = nn.Conv1d(d_model, d_pw, kernel_size=1)
        self.act = nn.GELU()
        self.pw2 = nn.Conv1d(d_pw, d_model, kernel_size=1)
        self.norm2 = nn.GroupNorm(1, d_model)

        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, D, L_patched)
        residual = x
        h = self.dw_proj(x)
        h = self.dw_conv(h)
        h = self.dw_back(h)
        h = self.norm1(h)
        x = residual + self.drop(h)

        residual = x
        h = self.pw1(x)
        h = self.act(h)
        h = self.pw2(h)
        h = self.norm2(h)
        return residual + self.drop(h)


class ModernTCNEncoder(nn.Module):
    """Patch embed + N TCN blocks. Returns (B, D, L/P) for downstream fusion."""

    def __init__(
        self,
        in_features: int,
        d_model: int,
        patch_size: int = 8,
        n_blocks: int = 3,
        kernel_size: int = 7,
        dw_expansion: int = 2,
        pw_expansion: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.patch = PatchEmbed(in_features, d_model, patch_size)
        self.blocks = nn.ModuleList(
            [
                TCNBlock(d_model, kernel_size, dw_expansion, pw_expansion, dropout)
                for _ in range(n_blocks)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, F) -> permute to (B, F, L) for Conv1d
        h = x.transpose(1, 2)
        h = self.patch(h)
        for blk in self.blocks:
            h = blk(h)
        # Return as (B, L_patched, D) for token-style downstream fusion
        return h.transpose(1, 2)
