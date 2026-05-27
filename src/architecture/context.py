"""Context encoders for date features and tide time series."""
from __future__ import annotations
import torch
import torch.nn as nn


class DateEncoderFF(nn.Module):
    """
    Fourier feature encoder for per-T date features.

    Parameters
    ----------
    d_ctx : int
        Output context width per time step.
    n_freqs : int, default=8
        Number of Fourier frequencies per scalar (sin+cos → 2*n_freqs per scalar).

    Input
    -----
    date_enc : Tensor, shape (B, 6, T)
        Normalized [0,1]-ish features per T: lead_norm, ISO-week%52, DOY, DOM, DOW, hour.

    Returns
    -------
    Tensor, shape (B, T, d_ctx)
    """
    def __init__(self, d_ctx: int, n_freqs: int = 8):
        super().__init__()
        self.register_buffer("freqs", 2.0**torch.arange(n_freqs).float() * 2 * torch.pi)
        in_dim = 6 * (2 * n_freqs)  # sin/cos per scalar
        hidden = max(32, d_ctx)
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, d_ctx),
        )

    def forward(self, date_enc: torch.Tensor) -> torch.Tensor:
        B, S, T = date_enc.shape
        x = date_enc.transpose(1, 2)  # (B,T,6)
        feats = []
        for s in range(S):
            v = x[..., s:s+1]                               # (B,T,1)
            angles = v * self.freqs.view(1, 1, -1)          # (B,T,n_freqs)
            feats += [torch.sin(angles), torch.cos(angles)] # list of (B,T,n_freqs)
        z = torch.cat(feats, dim=-1)                        # (B,T,6*2*n_freqs)
        return self.net(z)                                  # (B,T,d_ctx)


class TideEncoderResNet1D(nn.Module):
    """
    ResNet-like 1D encoder over 480 tide samples → exactly T tokens.

    Parameters
    ----------
    d_ctx : int
        Output context width per time step.
    T : int
        Number of lead steps; we downsample 480 → T with stride=12.

    Input
    -----
    tide : Tensor, shape (B, 1, 480)

    Returns
    -------
    Tensor, shape (B, T, d_ctx)
    """
    def __init__(self, d_ctx: int, T: int):
        super().__init__()
        self.conv_in = nn.Conv1d(1, d_ctx, kernel_size=25, stride=12, padding=12)  # 480 → T=40
        self.block1  = nn.Sequential(
            nn.Conv1d(d_ctx, d_ctx, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(d_ctx, d_ctx, kernel_size=3, padding=1),
        )
        self.block2  = nn.Sequential(
            nn.Conv1d(d_ctx, d_ctx, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(d_ctx, d_ctx, kernel_size=3, padding=1),
        )
        self.act = nn.GELU()

    def forward(self, tide: torch.Tensor) -> torch.Tensor:
        x = self.conv_in(tide)                    # (B,d_ctx,T)
        x = x + self.block1(x); x = self.act(x)  # (B,d_ctx,T)
        x = x + self.block2(x); x = self.act(x)  # (B,d_ctx,T)
        return x.transpose(1, 2)                 # (B,T,d_ctx)


class HistoryEncoder(nn.Module):
    """
    Encodes T=40 history steps using a 2D Convolutional network.
    Treats the (Time, Frequency) grid as an image, preserving spectral topology
    and drastically reducing parameter count compared to a flattened 1D approach.
    """
    def __init__(self, d_ctx: int, F_bins: int, n_vars: int):
        super().__init__()
        # Note: F_bins is kept in the signature for compatibility with model_specx.py, 
        # but is no longer needed to define the input channels.
        
        # Channels = (Value + Mask) * n_vars
        in_channels = 2 * n_vars
        
        self.conv_in = nn.Conv2d(in_channels, d_ctx, kernel_size=3, padding=1)
        self.block1 = nn.Sequential(
            nn.Conv2d(d_ctx, d_ctx, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(d_ctx, d_ctx, kernel_size=3, padding=1),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(d_ctx, d_ctx, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(d_ctx, d_ctx, kernel_size=3, padding=1),
        )
        self.act = nn.GELU()

    def forward(self, history: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # history/mask: (B, T, F, n_vars)
        
        # 1. Concatenate variables and masks along the feature dimension
        x = torch.cat([history, mask], dim=-1) # (B, T, F, 2 * n_vars)
        
        # 2. Permute to (B, Channels, T, F) for Conv2d
        x = x.permute(0, 3, 1, 2)
        
        # 3. 2D Convolutional path
        x = self.conv_in(x)
        x = x + self.block1(x)
        x = self.act(x)
        x = x + self.block2(x)
        x = self.act(x) # (B, d_ctx, T, F)
        
        # 4. Collapse the frequency dimension (Average Pooling)
        x = x.mean(dim=-1) # (B, d_ctx, T)
        
        # 5. Transpose to sequence format for ContextFusion
        return x.transpose(1, 2) # (B, T, d_ctx)


class ContextFusion(nn.Module):
    """
    Fuse optional per-T tokens (date/tide/history) into one per-T code.
    Handles dynamic number of inputs (1 to 3).
    """
    def __init__(self, d_ctx: int):
        super().__init__()
        self.proj2 = nn.Linear(2 * d_ctx, d_ctx)
        self.proj3 = nn.Linear(3 * d_ctx, d_ctx)

    def forward(self, date_tok=None, tide_tok=None, hist_tok=None):
        # Collect active inputs
        toks = [t for t in [date_tok, tide_tok, hist_tok] if t is not None]
        
        if len(toks) == 0:
            return None, None
            
        if len(toks) == 1:
            perT = toks[0]
        elif len(toks) == 2:
            perT = self.proj2(torch.cat(toks, dim=-1))
        else: # len == 3
            perT = self.proj3(torch.cat(toks, dim=-1))
                
        global_code = perT.mean(dim=1)  # (B,d_ctx)
        return perT, global_code