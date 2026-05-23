"""Pure-PyTorch selective state-space model (Mamba-1 style) for MPS.

The official Mamba-1/Mamba-2 implementations require CUDA and a Triton kernel
for the parallel selective-scan. Apple Silicon has neither. This file implements
the same core mechanic with a sequential scan loop — correct, MPS-compatible,
~10-50x slower than CUDA. For ALTUS Layer 1 with seq_len=240 / d_model=96 this
is fast enough for dev and modest training runs.

Reference: Gu & Dao 2023, "Mamba: Linear-Time Sequence Modeling with Selective
State Spaces." Inspired by the johnma2006/mamba-minimal reference repo.

Block layout (one MambaBlock):
    x -> in_proj (D -> 2*expand*D) -> split into (u, gate)
       -> conv1d(u) -> SiLU
       -> selective-scan( delta=f(u), B=g(u), C=h(u), A=learned ) -> y
       -> y * SiLU(gate) -> out_proj (expand*D -> D)
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class SelectiveSSM(nn.Module):
    """The selective-scan core. Computes a state-space recurrence where A, B,
    C, and the discretization step Δ are all functions of the input."""

    def __init__(self, d_inner: int, d_state: int = 16, dt_rank: int | None = None) -> None:
        super().__init__()
        self.d_inner = d_inner
        self.d_state = d_state
        self.dt_rank = dt_rank if dt_rank is not None else max(1, d_inner // 16)

        # Project input to (Δ, B, C). Δ is low-rank (dt_rank) then projected up.
        self.x_proj = nn.Linear(d_inner, self.dt_rank + 2 * d_state, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, d_inner, bias=True)

        # Initialize dt_proj bias so initial Δ values are in a useful range.
        with torch.no_grad():
            dt_init_std = self.dt_rank**-0.5
            self.dt_proj.weight.uniform_(-dt_init_std, dt_init_std)
            # Inverse softplus so softplus(bias) is small but non-zero.
            dt = torch.exp(
                torch.rand(d_inner) * (math.log(0.1) - math.log(0.001)) + math.log(0.001)
            ).clamp(min=1e-4)
            inv_dt = dt + torch.log(-torch.expm1(-dt))
            self.dt_proj.bias.copy_(inv_dt)

        # The state matrix A in S4-style log form. Initialized to a stable
        # negative spectrum (HIPPO-like ramp): A = -[1..N].
        A = torch.arange(1, d_state + 1, dtype=torch.float32).unsqueeze(0).repeat(d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))   # (d_inner, d_state)
        self.D = nn.Parameter(torch.ones(d_inner))  # skip connection

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        """u: (B, L, d_inner) -> y: (B, L, d_inner). Sequential scan."""
        B_, L, D_inner = u.shape
        N = self.d_state

        # Compute Δ, B, C from input.
        x_dbl = self.x_proj(u)  # (B, L, dt_rank + 2*N)
        delta, B_in, C_in = torch.split(x_dbl, [self.dt_rank, N, N], dim=-1)
        delta = F.softplus(self.dt_proj(delta))  # (B, L, D_inner)

        # Discretize A and B with Δ (zero-order hold approximation).
        # A is shared across batch; expand for elementwise math.
        A = -torch.exp(self.A_log)  # (D_inner, N)
        # deltaA: (B, L, D_inner, N)
        deltaA = torch.exp(delta.unsqueeze(-1) * A)
        # deltaB_u: (B, L, D_inner, N)  -- B_in broadcast to (B, L, 1, N) * delta*u
        deltaB_u = delta.unsqueeze(-1) * B_in.unsqueeze(2) * u.unsqueeze(-1)

        # Sequential scan: h_t = deltaA_t * h_{t-1} + deltaB_u_t
        h = torch.zeros(B_, D_inner, N, device=u.device, dtype=u.dtype)
        ys = []
        for t in range(L):
            h = deltaA[:, t] * h + deltaB_u[:, t]   # (B, D_inner, N)
            # y_t = C_t · h_t  ->  (B, D_inner)
            y_t = (h * C_in[:, t].unsqueeze(1)).sum(dim=-1)
            ys.append(y_t)
        y = torch.stack(ys, dim=1)  # (B, L, D_inner)

        # Skip connection
        y = y + u * self.D
        return y


class MambaBlock(nn.Module):
    """Full Mamba block: gated SSM with local conv preconditioning."""

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        d_inner = expand * d_model

        self.in_proj = nn.Linear(d_model, 2 * d_inner, bias=False)
        # Depthwise local conv for short-range context before the SSM.
        self.conv1d = nn.Conv1d(
            d_inner, d_inner, kernel_size=d_conv, padding=d_conv - 1, groups=d_inner
        )
        self.d_conv = d_conv
        self.act = nn.SiLU()
        self.ssm = SelectiveSSM(d_inner, d_state=d_state)
        self.out_proj = nn.Linear(d_inner, d_model, bias=False)
        self.norm = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, d_model)
        residual = x
        x = self.norm(x)

        xz = self.in_proj(x)                          # (B, L, 2*d_inner)
        u, gate = xz.chunk(2, dim=-1)                 # each (B, L, d_inner)

        # Local conv preconditioning (causal: trim the padding from the right)
        u_t = u.transpose(1, 2)                       # (B, d_inner, L)
        u_t = self.conv1d(u_t)[..., : u.shape[1]]
        u = self.act(u_t.transpose(1, 2))

        y = self.ssm(u)                               # (B, L, d_inner)
        y = y * self.act(gate)                        # gated output
        y = self.out_proj(y)                          # (B, L, d_model)
        return residual + self.drop(y)


class MambaEncoder(nn.Module):
    """N stacked Mamba blocks operating directly on the input feature stream."""

    def __init__(
        self,
        in_features: int,
        d_model: int,
        n_blocks: int = 3,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.input_proj = nn.Linear(in_features, d_model)
        self.blocks = nn.ModuleList(
            [
                MambaBlock(d_model, d_state=d_state, d_conv=d_conv, expand=expand, dropout=dropout)
                for _ in range(n_blocks)
            ]
        )
        self.final_norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, F) -> (B, L, d_model)
        h = self.input_proj(x)
        for blk in self.blocks:
            h = blk(h)
        return self.final_norm(h)
