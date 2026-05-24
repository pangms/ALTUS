"""SimMTM — Simple Masked Time-series Modeling for self-supervised pretraining.

Phase K of ALTUS architecture. Answers Q27 (pattern similarity to historical
setups) — produces continuous representations of bar windows that capture
non-Markovian similarity structure not visible to supervised triple-barrier
training.

Reference: Liu et al. 2023, "SimMTM: A Simple Pre-Training Framework for
Masked Time-Series Modeling."

The pipeline has 3 stages:
  1. PRETRAIN (offline, scripts/pretrain_simmtm.py)
       - Encoder + reconstruction head trained on masked-bar reconstruction
       - Loss: MSE between predicted and original values at masked positions
       - Optional contrastive term: representations of similar series cluster
  2. CACHE (offline, scripts/build_simmtm_cache.py)
       - Frozen encoder run on every bar's window → 96-D embedding
       - Saved to artifacts/simmtm_embeddings.parquet
  3. FEATURE (online, altus/features/families/simmtm.py)
       - Cache loaded; embeddings exposed as 96 additional L1 features

Architecture:
  Input:    (B, L, F)  — window of bars, F features per bar (OHLCV + returns)
  Mask:     replace 50% of timesteps with learnable [MASK] embedding
  Encoder:  TCN backbone (reuses ModernTCN — proven on time-series)
            → (B, L, d_model)
  Pool:     attention pool → (B, d_model)  for series-level embedding
  Recon:    linear (d_model → F) per timestep → reconstruct original
"""
from __future__ import annotations

import torch
import torch.nn as nn

from altus.models.modern_tcn import ModernTCNEncoder


class SimMTMMasking:
    """Random temporal masking utility (used by both pretrain + inference paths)."""

    def __init__(self, mask_ratio: float = 0.5):
        self.mask_ratio = mask_ratio

    def __call__(
        self,
        x: torch.Tensor,
        mask_token: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Mask `mask_ratio` of timesteps per sample.

        Returns (masked_x, mask) where mask is (B, L) bool — True = masked.
        Masked positions are replaced with `mask_token` (shape (F,)).
        """
        B, L, F = x.shape
        # Random mask per (batch, time) — different mask each sample
        rand = torch.rand(B, L, device=x.device)
        mask = rand < self.mask_ratio
        # Broadcast mask_token (F,) to (B, L, F) only where mask is True
        x_masked = torch.where(mask.unsqueeze(-1), mask_token.view(1, 1, F), x)
        return x_masked, mask


class SimMTMEncoder(nn.Module):
    """The frozen-at-inference encoder. Maps (B, L, F) -> (B, L, d_model).

    Backbone is ModernTCN (same as L1's TCN branch). Trained via masked
    reconstruction; at inference, we strip the recon head and use the
    encoder + attention pool to produce per-bar embeddings.
    """

    def __init__(
        self,
        n_features_in: int,
        d_model: int = 96,
        n_blocks: int = 3,
        kernel_size: int = 7,
        patch_size: int = 8,
        dropout: float = 0.10,
        mask_ratio: float = 0.5,
    ) -> None:
        super().__init__()
        self.n_features_in = n_features_in
        self.d_model = d_model
        self.mask_ratio = mask_ratio

        # Learnable mask token — replaces feature values at masked positions
        self.mask_token = nn.Parameter(torch.zeros(n_features_in))

        # Backbone: TCN encoder
        self.backbone = ModernTCNEncoder(
            in_features=n_features_in,
            d_model=d_model,
            patch_size=patch_size,
            n_blocks=n_blocks,
            kernel_size=kernel_size,
            dw_expansion=2,
            pw_expansion=4,
            dropout=dropout,
        )
        # Attention pool: collapses (B, L, D) -> (B, D)
        self.pool_query = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.pool_attn = nn.MultiheadAttention(d_model, num_heads=4, batch_first=True)

        # Masking utility
        self.masking = SimMTMMasking(mask_ratio=mask_ratio)

    def forward_encoded(self, x: torch.Tensor) -> torch.Tensor:
        """Run backbone without masking. Returns (B, L, d_model).

        This is the path used at INFERENCE — no masking, full input.
        Returns per-timestep encodings.
        """
        return self.backbone(x)

    def forward_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """Inference path: return a single (B, d_model) embedding per series."""
        h = self.forward_encoded(x)  # (B, L, d_model)
        q = self.pool_query.expand(h.shape[0], -1, -1)
        out, _ = self.pool_attn(q, h, h, need_weights=False)
        return out.squeeze(1)  # (B, d_model)

    def forward_pretrain(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Pretrain path: mask input, encode, return (encodings, mask) for the
        reconstruction head to use.
        """
        x_masked, mask = self.masking(x, self.mask_token)
        h = self.backbone(x_masked)  # (B, L, d_model)
        return h, mask


class SimMTMReconstructionHead(nn.Module):
    """Per-timestep reconstruction head for pretraining only."""

    def __init__(self, d_model: int, n_features_in: int) -> None:
        super().__init__()
        self.proj = nn.Linear(d_model, n_features_in)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """h: (B, L, d_model) -> (B, L, n_features_in) reconstructed values."""
        return self.proj(h)


class SimMTMPretrainModel(nn.Module):
    """Combined encoder + recon head used only during pretraining.

    After pretraining, save the encoder's state_dict alone (not this wrapper)
    and load it back via SimMTMEncoder for the cache build + inference.
    """

    def __init__(
        self,
        n_features_in: int,
        d_model: int = 96,
        n_blocks: int = 3,
        kernel_size: int = 7,
        patch_size: int = 8,
        dropout: float = 0.10,
        mask_ratio: float = 0.5,
    ) -> None:
        super().__init__()
        self.encoder = SimMTMEncoder(
            n_features_in=n_features_in,
            d_model=d_model,
            n_blocks=n_blocks,
            kernel_size=kernel_size,
            patch_size=patch_size,
            dropout=dropout,
            mask_ratio=mask_ratio,
        )
        self.recon = SimMTMReconstructionHead(d_model=d_model, n_features_in=n_features_in)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Compute masked reconstruction loss inputs.

        Returns dict:
          - x_orig:    original input (B, L, F)
          - x_recon:   reconstructed values (B, L, F)
          - mask:      mask bool (B, L)
        Loss = MSE(x_orig[mask], x_recon[mask]) computed by caller.
        """
        h, mask = self.encoder.forward_pretrain(x)  # h: (B, L_patched, d_model)
        # TCN backbone patches the input; upsample h back to original L for
        # per-timestep reconstruction. Linear interpolation in the time axis.
        L_orig = x.shape[1]
        if h.shape[1] != L_orig:
            h_up = torch.nn.functional.interpolate(
                h.transpose(1, 2), size=L_orig, mode="linear", align_corners=False
            ).transpose(1, 2)
        else:
            h_up = h
        x_recon = self.recon(h_up)
        return {"x_orig": x, "x_recon": x_recon, "mask": mask}


def masked_reconstruction_loss(
    x_orig: torch.Tensor, x_recon: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    """MSE loss on masked positions only.

    x_orig, x_recon: (B, L, F)
    mask: (B, L) bool — True at masked positions
    """
    diff_sq = (x_orig - x_recon) ** 2  # (B, L, F)
    # Sum over features; then average only over masked positions
    per_pos = diff_sq.mean(dim=-1)  # (B, L)
    mask_f = mask.float()
    denom = mask_f.sum().clamp(min=1.0)
    return (per_pos * mask_f).sum() / denom
