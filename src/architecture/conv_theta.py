"""Geometry-aware convolutions and direction (θ) operators.

Axes convention
---------------
All tensors use (B, C, T, F, Θ) unless Θ=1 post-collapse.

Padding policy
--------------
- Θ (direction): circular (periodic)
- T (lead time), F (frequency): replicate (edge) to avoid inventing energy
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as Fnn
from torch.nn import init

from .film import apply_film


# ───────────────────────── generic helpers ─────────────────────────
def conv3d_1x1(c_in: int, c_out: int, *, groups: int = 1) -> nn.Conv3d:
    """1×1×1 pointwise conv. Shape: (B,C,T,F,Θ) → (B,C',T,F,Θ)."""
    return nn.Conv3d(c_in, c_out, kernel_size=1, stride=1, groups=groups)

def batch_norm3d(c: int) -> nn.BatchNorm3d:
    """3D BatchNorm wrapper (channels=C)."""
    return nn.BatchNorm3d(c)


# ───────────────────── θ-aware 3×3×3 convolution ────────────────────
class ThetaAwareConv3D(nn.Module):
    """
    3×3×3 conv that applies circular padding on Θ (direction) and replicate
    padding on T,F. Keeps the input spatial/temporal shape.

    Parameters
    ----------
    c_in : int
        Input channels C.
    c_out : int
        Output channels C'.
    stride : int, default=1
        Stride for the conv (applies to T,F,Θ).
    bias : bool, default=True
        Whether to include a bias term.
    groups : int, default=1
        Grouped convolution.

    Notes
    -----
    - We pad Θ manually with Fnn.pad(..., mode="circular"); T,F rely on the
      internal conv replicate padding.
    """
    def __init__(self, c_in: int, c_out: int, *, stride: int = 1, bias: bool = True, groups: int = 1):
        super().__init__()
        self.conv = nn.Conv3d(
            c_in, c_out,
            kernel_size=(3, 3, 3),
            stride=stride,
            padding=(1, 1, 0),      # T=1, F=1, Θ=0 (Θ padded manually below)
            padding_mode="replicate",
            bias=bias,
            groups=groups,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # pad order for 3D: (Θ_left, Θ_right, F_left, F_right, T_left, T_right)
        x = Fnn.pad(x, (1, 1, 0, 0, 0, 0), mode="circular")  # circular along Θ only
        return self.conv(x)


def theta_mixing_conv3x3(c_in: int, c_out: int, *, stride: int = 1, bias: bool = True, groups: int = 1) -> nn.Module:
    """Alias for ThetaAwareConv3D (3×3×3, mixes Θ)."""
    return ThetaAwareConv3D(c_in, c_out, stride=stride, bias=bias, groups=groups)


def no_theta_conv3x3(c_in: int, c_out: int, *, stride: int = 1, bias: bool = True, groups: int = 1) -> nn.Conv3d:
    """
    3×3×1 conv that does not mix Θ (assume Θ=1). Replicate padding on T,F.

    Shape
    -----
    (B,C,T,F,1) → (B,C',T,F,1)
    """
    return nn.Conv3d(
        c_in, c_out,
        kernel_size=(3, 3, 1),
        stride=stride,
        padding=(1, 1, 0),
        padding_mode="replicate",
        bias=bias,
        groups=groups,
    )


# ───────────────────── Direction anti-alias & downsample ─────────────────────
class DirectionAntiAlias3(nn.Module):
    """
    Fixed depthwise 1×1×3 blur along Θ with circular padding.
    Kernel = [1, 2, 1] / 4 (anti-alias prior to θ/2 pooling).

    Input/Output
    ------------
    (B,C,T,F,Θ) → (B,C,T,F,Θ)
    """
    def __init__(self, channels: int):
        super().__init__()
        self.blur = nn.Conv3d(
            channels, channels,
            kernel_size=(1, 1, 3),
            padding=(0, 0, 1),
            stride=1,
            groups=channels,
            bias=False,
            padding_mode="circular",
        )
        with torch.no_grad():
            k = torch.tensor([1., 2., 1.]) / 4.0
            w = torch.zeros(channels, 1, 1, 1, 3)
            w[:, 0, 0, 0, :] = k
            self.blur.weight.copy_(w)
        for p in self.parameters():
            p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.blur(x)


class DirectionDownsampleAA(nn.Module):
    """
    Two θ-mixing convs; optional anti-aliased average pooling along Θ by 2.
    Now supports FiLM conditioning.

    Parameters
    ----------
    c_in : int
        Input channels C.
    c_out : int
        Output channels C'.
    pool : bool
        If True, apply anti-alias blur then AvgPool3d(1,1,2) along Θ.
    act : nn.Module
        Activation function module (e.g., ReLU/SilU).

    Shape
    -----
    If pool:
        (B,C,T,F,Θ) → (B,C',T,F,Θ/2)
    else:
        (B,C,T,F,Θ) → (B,C',T,F,Θ)
    """
    def __init__(self, c_in: int, c_out: int, *, pool: bool, act: nn.Module):
        super().__init__()
        self.pool = pool
        self.act  = act
        self.conv1 = theta_mixing_conv3x3(c_in,  c_out)
        self.bn1   = batch_norm3d(c_out)
        self.conv2 = theta_mixing_conv3x3(c_out, c_out)
        if pool:
            self.blur_theta = DirectionAntiAlias3(c_out)
            self.pool_theta = nn.AvgPool3d(kernel_size=(1, 1, 2), stride=(1, 1, 2))

    def forward(self, x: torch.Tensor,
                gamma_t=None, beta_t=None, gamma_f=None, beta_f=None) -> torch.Tensor:
        x = self.act(self.bn1(self.conv1(x)))
        x = self.act(self.conv2(x))
        x = apply_film(x, gamma_t, beta_t, gamma_f, beta_f)
        if self.pool:
            x = self.blur_theta(x)
            x = self.pool_theta(x)  # Δθ constant → average pool
        return x


# ───────────────────────── Direction soft collapse ─────────────────────────
class DirectionSoftCollapse(nn.Module):
    """
    Learnable soft collapse of Θ using a temperature-scaled softmax.

    y = Σ_Θ softmax(logits/τ) ⊙ x  →  (B,C,T,F,1)

    Parameters
    ----------
    channels : int
        Number of channels C in the input.
    temperature : float, default=1.0
        Softmax temperature τ; lower makes the collapse sharper.

    Input
    -----
    x : Tensor, shape (B, C, T, F, Θ)

    Returns
    -------
    Tensor, shape (B, C, T, F, 1)
    """
    def __init__(self, channels: int, temperature: float = 1.0):
        super().__init__()
        self.logits = conv3d_1x1(channels, 1)   # (B,1,T,F,Θ)
        self.tau = temperature

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a = torch.softmax(self.logits(x) / self.tau, dim=-1)  # softmax over Θ
        return (x * a).sum(dim=-1, keepdim=True)


# ─────────────────────────── init helpers (optional) ─────────────────────────
def init_xavier_conv(m: nn.Module) -> None:
    """Xavier init for convs; zero bias if present."""
    if isinstance(m, (nn.Conv3d, nn.Conv1d)):
        init.xavier_normal_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)