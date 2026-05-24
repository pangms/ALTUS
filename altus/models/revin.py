"""Reversible Instance Normalization (RevIN).

Reference: Kim et al., "Reversible Instance Normalization for Accurate Time-Series
Forecasting against Distribution Shift," ICLR 2022.

Why this matters for ALTUS specifically: market data is non-stationary — the
statistical properties of price/volume in 2021 differ from 2024. A model
trained on historical data and deployed live encounters a different distribution.
RevIN solves this by normalizing each input window per-instance (using that
window's own mean/std) before the encoder sees it. The encoder always sees
inputs in standardized form regardless of the underlying regime, which directly
attacks the "trained-on-one-distribution, live-on-another" failure mode.

For classification tasks (like our triple-barrier heads), we only need the
forward normalization — there's no denormalization step because outputs are
logits, not values in the original scale. We still implement the full forward+
inverse API so we can later use RevIN inside regression-style models if needed.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class RevIN(nn.Module):
    """Per-instance reversible normalization.

    Forward (norm):  x_out = ((x - mean(x, dim=time)) / std(x, dim=time)) * gamma + beta
    Inverse (denorm): x_orig = ((x - beta) / gamma) * std + mean   # restores original scale

    The mean/std for each forward pass are cached on `self` for use in a
    subsequent inverse call (per-instance, so they're refreshed each batch).
    """

    def __init__(self, num_features: int, eps: float = 1e-5, affine: bool = True) -> None:
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        if affine:
            # Learnable per-feature scale and shift — gives the model freedom
            # to deviate from pure z-scoring if that's what optimizes the loss.
            self.affine_weight = nn.Parameter(torch.ones(num_features))
            self.affine_bias = nn.Parameter(torch.zeros(num_features))
        self._cached_mean: torch.Tensor | None = None
        self._cached_std: torch.Tensor | None = None

    def forward(self, x: torch.Tensor, mode: str = "norm") -> torch.Tensor:
        """
        x: (B, L, F)   batch, sequence length, features
        mode: 'norm' applies normalization; 'denorm' reverses it
        """
        if mode == "norm":
            return self._normalize(x)
        if mode == "denorm":
            return self._denormalize(x)
        raise ValueError(f"unknown mode: {mode}. Use 'norm' or 'denorm'.")

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        # Compute statistics along the TIME axis (dim=1) per instance.
        # Result shape: (B, 1, F) — broadcasts across the time dim.
        self._cached_mean = x.mean(dim=1, keepdim=True).detach()
        self._cached_std = torch.sqrt(
            x.var(dim=1, keepdim=True, unbiased=False) + self.eps
        ).detach()
        x = (x - self._cached_mean) / self._cached_std
        if self.affine:
            # Broadcast: weight/bias are (F,), x is (B, L, F)
            x = x * self.affine_weight + self.affine_bias
        return x

    def _denormalize(self, x: torch.Tensor) -> torch.Tensor:
        if self._cached_mean is None or self._cached_std is None:
            raise RuntimeError("denorm called before norm — no cached statistics")
        if self.affine:
            # No eps here: affine_weight is a learnable parameter; reversibility
            # requires we divide by exactly what we multiplied by in _normalize.
            # If affine_weight goes to zero during training, the model is broken
            # anyway — we want the loud failure, not a silent epsilon-fudged divide.
            x = (x - self.affine_bias) / self.affine_weight
        # _cached_std already has eps baked in, so multiplying back is self-consistent.
        x = x * self._cached_std + self._cached_mean
        return x
