"""xLSTM encoder branch (parallel A/B variant of Mamba).

Reference: Beck, Pöppel, Spanring, Auer, Prudnikova, Kopp, Klambauer, Brandstetter,
Hochreiter 2024, "xLSTM: Extended Long Short-Term Memory."

Two flavors:
  * sLSTM: scalar-memory LSTM with exponential gating + normalizer state. Adds
    memory mixing across heads.
  * mLSTM: matrix-memory LSTM with key/value associative storage. Parallel
    across time (covariance-update form).

For ALTUS Layer 1 we stack [mLSTM, sLSTM, mLSTM, ...] alternating blocks. This
gives both fast parallel processing (mLSTM) and the scalar gating dynamics
(sLSTM) that some benchmarks show beat Mamba on long-range time-series tasks.

Implementation is a pragmatic pure-PyTorch version — runs on MPS, no special
kernels. Not the paper's exact factored-matrix layout but matches the core
gating + matrix-memory mechanic.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class MLSTMCell(nn.Module):
    """Matrix-memory LSTM cell with exponential input gating.

    State per sample: C (d_head x d_head matrix), n (d_head vector), m (scalar).
    Update: associative key/value write, query readout.
    """

    def __init__(self, d_model: int, n_heads: int = 4) -> None:
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.i_proj = nn.Linear(d_model, n_heads, bias=True)  # input gate (per-head scalar)
        self.f_proj = nn.Linear(d_model, n_heads, bias=True)  # forget gate (per-head scalar)
        self.o_proj = nn.Linear(d_model, d_model, bias=True)  # output gate

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, d_model) -> (B, L, d_model)
        B, L, D = x.shape
        H, dh = self.n_heads, self.d_head

        q = self.q_proj(x).view(B, L, H, dh) / math.sqrt(dh)
        k = self.k_proj(x).view(B, L, H, dh) / math.sqrt(dh)
        v = self.v_proj(x).view(B, L, H, dh)
        i_raw = self.i_proj(x)  # (B, L, H)
        f_raw = self.f_proj(x)  # (B, L, H)
        o = torch.sigmoid(self.o_proj(x)).view(B, L, H, dh)

        # Exponential gating with stabilizer m: prevents exp() blowup.
        # i_t = exp(i_raw_t - m_t), f_t = exp(f_raw_t + m_{t-1} - m_t)
        # Choose m_t = max(f_raw_t + m_{t-1}, i_raw_t). See xLSTM paper Appendix.
        C = x.new_zeros(B, H, dh, dh)   # matrix memory
        n = x.new_zeros(B, H, dh)        # normalizer
        m = x.new_full((B, H), -1e4)     # log-stabilizer

        outs = []
        for t in range(L):
            f_log = f_raw[:, t]                              # (B, H)
            i_log = i_raw[:, t]
            m_new = torch.maximum(f_log + m, i_log)          # (B, H)
            f_gate = torch.exp(f_log + m - m_new)
            i_gate = torch.exp(i_log - m_new)

            # C_t = f * C_{t-1} + i * (v k^T)
            vk = v[:, t].unsqueeze(-1) * k[:, t].unsqueeze(-2)   # (B, H, dh, dh)
            C = f_gate[..., None, None] * C + i_gate[..., None, None] * vk
            n = f_gate[..., None] * n + i_gate[..., None] * k[:, t]
            m = m_new

            # Readout: h_t = o_t * (C_t q_t) / max(|n_t^T q_t|, 1)
            num = (C * q[:, t].unsqueeze(-2)).sum(dim=-1)        # (B, H, dh)
            denom = (n * q[:, t]).sum(dim=-1, keepdim=True).abs().clamp(min=1.0)
            h = num / denom
            outs.append((o[:, t] * h).reshape(B, D))

        return torch.stack(outs, dim=1)


class SLSTMCell(nn.Module):
    """Scalar-memory LSTM with exponential gating + normalizer (xLSTM sLSTM)."""

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.d_model = d_model
        self.proj = nn.Linear(d_model, 4 * d_model, bias=True)  # i, f, z, o

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, D = x.shape
        h = x.new_zeros(B, D)
        c = x.new_zeros(B, D)
        n = x.new_zeros(B, D)
        m = x.new_full((B, D), -1e4)
        outs = []
        for t in range(L):
            i_raw, f_raw, z, o_raw = self.proj(x[:, t]).chunk(4, dim=-1)
            m_new = torch.maximum(f_raw + m, i_raw)
            i_g = torch.exp(i_raw - m_new)
            f_g = torch.exp(f_raw + m - m_new)
            z = torch.tanh(z)
            o = torch.sigmoid(o_raw)
            c = f_g * c + i_g * z
            n = f_g * n + i_g
            h = o * c / n.clamp(min=1e-6)
            m = m_new
            outs.append(h)
        return torch.stack(outs, dim=1)


class XLSTMBlock(nn.Module):
    """One xLSTM block. Alternates between mLSTM and sLSTM by `kind`."""

    def __init__(self, d_model: int, kind: str = "m", n_heads: int = 4, dropout: float = 0.1) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.cell = MLSTMCell(d_model, n_heads=n_heads) if kind == "m" else SLSTMCell(d_model)
        self.proj_in = nn.Linear(d_model, d_model)
        self.proj_out = nn.Linear(d_model, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        h = self.norm(x)
        h = F.silu(self.proj_in(h))
        h = self.cell(h)
        h = self.proj_out(h)
        return residual + self.drop(h)


class XLSTMEncoder(nn.Module):
    """N alternating xLSTM blocks operating on the input feature stream."""

    def __init__(
        self,
        in_features: int,
        d_model: int,
        n_blocks: int = 3,
        n_heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.input_proj = nn.Linear(in_features, d_model)
        kinds = ["m" if i % 2 == 0 else "s" for i in range(n_blocks)]
        self.blocks = nn.ModuleList(
            [XLSTMBlock(d_model, kind=kinds[i], n_heads=n_heads, dropout=dropout) for i in range(n_blocks)]
        )
        self.final_norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.input_proj(x)
        for blk in self.blocks:
            h = blk(h)
        return self.final_norm(h)
