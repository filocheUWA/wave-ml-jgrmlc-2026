"""Optional cross-attention: latent (T×F tokens) attends to per-T context tokens."""
from __future__ import annotations
import torch
import torch.nn as nn


class CrossAttentionBlock(nn.Module):
    """
    Single cross-attention block.

    Queries:    latent grid after Θ-collapse, reshaped into (T×F) tokens.
    Keys/vals: per-T context tokens (optionally fused date + tide).

    Parameters
    ----------
    C : int
        Latent/decoder channel width (query/key/value dimension).
    ctx_dim : int
        Width of context tokens (projected to C internally).
    n_heads : int, default=2
        Multi-head attention heads.
    dropout : float, default=0.1
        Dropout in attention.

    Inputs
    ------
    latent : Tensor, shape (B, C, T, F, 1)
    ctx_tokens : Tensor | None, shape (B, T, ctx_dim)

    Returns
    -------
    Tensor, shape (B, C, T, F, 1)
    """
    def __init__(self, C: int, ctx_dim: int, n_heads: int = 2, dropout: float = 0.1):
        super().__init__()
        self.ctx_proj = nn.Linear(ctx_dim, C)
        self.q_ln = nn.LayerNorm(C)
        self.kv_ln = nn.LayerNorm(C)
        self.mha = nn.MultiheadAttention(C, num_heads=n_heads, dropout=dropout, batch_first=False)
        self.dropout = nn.Dropout(dropout)

        # Zero-initialize the MHA output projection for stability at the start of training.
        nn.init.zeros_(self.mha.out_proj.weight)
        nn.init.zeros_(self.mha.out_proj.bias)

    def forward(self, latent: torch.Tensor, ctx_tokens: torch.Tensor | None) -> torch.Tensor:
        if ctx_tokens is None:
            return latent
        B, C, T, Fbins, _ = latent.shape
        # Queries: (TF,B,C)
        q = latent.permute(2, 3, 0, 1, 4).reshape(T * Fbins, B, C)
        q = self.q_ln(q)
        # Keys/values: (T,B,C)
        kv = self.ctx_proj(ctx_tokens)   # (B,T,C)
        kv = self.kv_ln(kv).permute(1, 0, 2)
        out, _ = self.mha(q, kv, kv, need_weights=False)
        out = self.dropout(out)
        out = out + q  # residual
        y = out.reshape(T, Fbins, B, C).permute(2, 3, 0, 1)  # (B,C,T,F)
        return y.unsqueeze(-1)