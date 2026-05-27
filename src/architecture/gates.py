"""Time & channel gating blocks (“attention-like” but lightweight)."""
from __future__ import annotations
import torch
import torch.nn as nn

from .conv_theta import (
    theta_mixing_conv3x3,
    no_theta_conv3x3,
    conv3d_1x1,
)
from .film import apply_film


class SeqChanGate(nn.Module):
    """
    Global sequence+channel gate (like squeeze-excite over (F,Θ)).

    Pooled over (F,Θ) → MLP → per-(B,C,T,·,·) scale in [0,1].

    Input/Output
    ------------
    (B,C,T,F,Θ) → (B,C,T,F,Θ)  (reweighted)
    """
    def __init__(self, c: int):
        super().__init__()
        self.up = conv3d_1x1(c, 2 * c)
        self.down = conv3d_1x1(2 * c, c)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pooled = x.mean(dim=(-1, -2), keepdim=True)  # avg over (F,Θ)
        w = self.sigmoid(self.down(self.up(pooled)))
        return x * w


class TemporalGate1D(nn.Module):
    """
    Time gate: 1D conv with kernel (3,1,1), broadcast over T.

    Produces per-voxel weights and multiplies the input.

    Input/Output
    ------------
    (B,C,T,F,Θ) → (B,C,T,F,Θ)
    """
    def __init__(self, c: int, T: int):
        super().__init__()
        self.conv = nn.Conv3d(c, c, kernel_size=(3, 1, 1), padding=(1, 0, 0))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.sigmoid(self.conv(x))  # (B,C,T,F,Θ)
        return x * w


class TimeSeqGateBlock(nn.Module):
    """
    Two convs + SeqChanGate + TemporalGate1D + residual.

    Parameters
    ----------
    c : int
        Channels.
    input_size : tuple[int,int,int]
        (T, Fbins, Θbins) for constructing the TemporalGate kernel.
    ktheta : int
        If 1, use 3×3×1 convs (no Θ mixing). Else, use 3×3×3 convs.
    """
    def __init__(self, c: int, input_size: tuple[int, int, int], ktheta: int = 3):
        super().__init__()
        T, _, _ = input_size
        if ktheta == 1:
            self.conv1 = no_theta_conv3x3(c, c)
            self.conv2 = no_theta_conv3x3(c, c)
        else:
            self.conv1 = theta_mixing_conv3x3(c, c)
            self.conv2 = theta_mixing_conv3x3(c, c)
        self.sea = SeqChanGate(c)
        self.spa = TemporalGate1D(c, T)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.conv2(self.conv1(x))
        y = self.sea(y)
        y = self.spa(y)
        return x + y


class TimeSeqGateBlockIO(nn.Module):
    """
    Channel-changing gate block. Now supports FiLM conditioning.
    conv1 maps C_in→C_out, conv2 keeps C_out; SeqChan + Temporal gate; residual
    uses 1×1×1 projection when C_in != C_out.

    input:  (B, C_in,  T, F, Θ_or_1)
    output: (B, C_out, T, F, Θ_or_1)
    """
    def __init__(self, c_in: int, c_out: int, input_size: tuple[int, int, int], ktheta: int = 3):
        super().__init__()
        T, _, _ = input_size
        if ktheta == 1:
            self.conv1 = no_theta_conv3x3(c_in,  c_out)
            self.conv2 = no_theta_conv3x3(c_out, c_out)
        else:
            self.conv1 = theta_mixing_conv3x3(c_in,  c_out)
            self.conv2 = theta_mixing_conv3x3(c_out, c_out)
        self.sea = SeqChanGate(c_out)
        self.spa = TemporalGate1D(c_out, T)
        self.res_proj = None if c_in == c_out else conv3d_1x1(c_in, c_out)

    def forward(self, x: torch.Tensor,
                gamma_t=None, beta_t=None, gamma_f=None, beta_f=None) -> torch.Tensor:
        y = self.conv2(self.conv1(x))
        y = apply_film(y, gamma_t, beta_t, gamma_f, beta_f)
        y = self.sea(y)
        y = self.spa(y)
        res = x if self.res_proj is None else self.res_proj(x)
        return res + y


class ResidualGateStack(nn.Module):
    """
    Stack of N TimeSeqGateBlock with an outer residual.

    Parameters
    ----------
    n : int
        Number of inner blocks.
    c : int
        Channels.
    input_size : tuple[int,int,int]
        (T, Fbins, Θbins).
    ktheta : int
        1 for 3×3×1 convs (no Θ mixing), else 3×3×3 convs.
    """
    def __init__(self, n: int, c: int, input_size: tuple[int, int, int], ktheta: int = 3):
        super().__init__()
        self.blocks = nn.ModuleList([TimeSeqGateBlock(c, input_size, ktheta=ktheta) for _ in range(n)])
        self.conv_res = no_theta_conv3x3(c, c) if ktheta == 1 else theta_mixing_conv3x3(c, c)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = x
        for b in self.blocks:
            y = b(y)
        y = self.conv_res(y)
        return x + y