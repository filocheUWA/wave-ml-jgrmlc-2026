"""Feature-wise linear modulation (FiLM) modules for T and F axes."""
from __future__ import annotations
import torch
import torch.nn as nn


class FiLMT(nn.Module):
    """
    Per-T FiLM: (B,T,d_ctx) → (γ_T, β_T) with shape (B,C,T,1,1).

    Initialization sets γ=0, β=0 → identity modulation.
    """
    def __init__(self, d_ctx: int, C: int):
        super().__init__()
        self.to_gamma = nn.Linear(d_ctx, C)
        self.to_beta  = nn.Linear(d_ctx, C)
        nn.init.zeros_(self.to_gamma.weight); nn.init.zeros_(self.to_gamma.bias)
        nn.init.zeros_(self.to_beta .weight); nn.init.zeros_(self.to_beta .bias)

    def forward(self, perT: torch.Tensor | None):
        if perT is None:
            return None, None
        gamma = self.to_gamma(perT).permute(0, 2, 1).unsqueeze(-1).unsqueeze(-1)  # (B,C,T,1,1)
        beta  = self.to_beta (perT).permute(0, 2, 1).unsqueeze(-1).unsqueeze(-1)  # (B,C,T,1,1)
        return gamma, beta


class FiLMF(nn.Module):
    """
    Per-F FiLM: combine global code (B, d_global) with learned per-F embeddings
    to produce (γ_F, β_F) with shape (B, C, 1, F, 1).
    """
    def __init__(self, d_global: int, C: int, Fbins: int, d_fpos: int = 16):
        super().__init__()
        self.freq_embed = nn.Parameter(torch.randn(Fbins, d_fpos) * 0.02)
        self.mlp = nn.Sequential(
            nn.Linear(d_global + d_fpos, max(32, C)),
            nn.GELU(),
            nn.Linear(max(32, C), 2 * C),
        )
        with torch.no_grad():
            self.mlp[-1].weight.mul_(0.0); self.mlp[-1].bias.zero_()

    def forward(self, global_code: torch.Tensor | None):
        if global_code is None:
            return None, None
        B, dg = global_code.shape
        Fbins, dpos = self.freq_embed.shape
        g = global_code.unsqueeze(1).expand(B, Fbins, dg)   # (B,F,dg)
        e = self.freq_embed.unsqueeze(0).expand(B, Fbins, dpos)  # (B,F,dpos)
        z = torch.cat([g, e], dim=-1)                      # (B,F,dg+dpos)
        h = self.mlp(z)                                    # (B,F,2C)
        C2 = h.shape[-1] // 2
        gamma = h[..., :C2].permute(0, 2, 1).unsqueeze(-1) # (B,C,F,1)
        beta  = h[..., C2:].permute(0, 2, 1).unsqueeze(-1) # (B,C,F,1)
        gamma = gamma.unsqueeze(2)  # (B,C,1,F,1)
        beta  = beta .unsqueeze(2)  # (B,C,1,F,1)
        return gamma, beta


def apply_film(x: torch.Tensor,
               gamma_t: torch.Tensor | None, beta_t: torch.Tensor | None,
               gamma_f: torch.Tensor | None, beta_f: torch.Tensor | None) -> torch.Tensor:
    """
    Apply factorized FiLM on T and F.

    Parameters
    ----------
    x : Tensor, shape (B,C,T,F,1)
    gamma_t, beta_t : Tensor | None
        (B,C,T,1,1) per-T scales/biases.
    gamma_f, beta_f : Tensor | None
        (B,C,1,F,1) per-F scales/biases.

    Returns
    -------
    Tensor, shape (B,C,T,F,1)
    """
    gamma = 0; beta = 0
    if gamma_t is not None: gamma = gamma + gamma_t
    if gamma_f is not None: gamma = gamma + gamma_f
    if beta_t  is not None: beta  = beta  + beta_t
    if beta_f  is not None: beta  = beta  + beta_f
    return (1 + gamma) * x + beta
