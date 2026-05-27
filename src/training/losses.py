"""Masked spectral loss functions used for supervised post-processing."""

import torch
import torch.nn as nn


class MaskedMSELoss(nn.Module):
    """Masked mean squared error over valid buoy observations."""
    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps
    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        diff2 = (y_pred - y_true).pow(2) * mask
        denom = mask.sum().clamp_min(1.0)
        mse   = diff2.sum() / denom
        return mse

class MaskedMAELoss(nn.Module):
    """Masked mean absolute error over valid buoy observations."""
    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps
    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        diff_abs = (y_pred - y_true).abs() * mask
        denom = mask.sum().clamp_min(1.0)
        mae   = diff_abs.sum() / denom
        return mae


class MaskedMSEBinWeightedLoss(nn.Module):
    """
    Bin-width-weighted MSE along F using a Riemann sum integration.
    Avoids trapezoidal bias over discontinuous observation masks.
    """
    def __init__(self, freqs: torch.Tensor, eps: float = 1e-8):
        super().__init__()
        
        # Frequency-bin widths account for the non-uniform spectral grid.
        df = torch.zeros_like(freqs)
        if len(freqs) > 1:
            df[1:-1] = (freqs[2:] - freqs[:-2]) / 2.0
            df[0] = freqs[1] - freqs[0]
            df[-1] = freqs[-1] - freqs[-2]
        else:
            df[0] = 1.0
            
        # Broadcast over `(B, C, T, F, Theta)`.
        self.register_buffer("df", df.view(1, 1, 1, -1, 1))
        self.eps = eps

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        diff2 = (y_pred - y_true).pow(2) * mask                 # (B,1,T,F,1)
        
        # Integrate along frequency with a Riemann sum.
        num   = (diff2 * self.df).sum(dim=-2)                   # (B,1,T,1)
        den   = (mask * self.df).sum(dim=-2).clamp_min(self.eps)
        mseT  = num / den                                       # (B,1,T,1)
        
        return mseT.mean()                                      # scalar


class MaskedMAEBinWeightedLoss(nn.Module):
    """
    Bin-width-weighted MAE along F using a Riemann sum integration.
    Avoids trapezoidal bias over discontinuous observation masks.
    """
    def __init__(self, freqs: torch.Tensor, eps: float = 1e-8):
        super().__init__()
        
        # Frequency-bin widths account for the non-uniform spectral grid.
        df = torch.zeros_like(freqs)
        if len(freqs) > 1:
            df[1:-1] = (freqs[2:] - freqs[:-2]) / 2.0
            df[0] = freqs[1] - freqs[0]
            df[-1] = freqs[-1] - freqs[-2]
        else:
            df[0] = 1.0
            
        # Broadcast over `(B, C, T, F, Theta)`.
        self.register_buffer("df", df.view(1, 1, 1, -1, 1))
        self.eps = eps

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        diff_abs = (y_pred - y_true).abs() * mask               # (B,1,T,F,1)
        
        # Integrate along frequency with a Riemann sum.
        num   = (diff_abs * self.df).sum(dim=-2)                # (B,1,T,1)
        den   = (mask * self.df).sum(dim=-2).clamp_min(self.eps)
        maeT  = num / den                                       # (B,1,T,1)
        
        return maeT.mean()                                      # scalar


def make_criterion(loss_type: str, freqs: torch.Tensor) -> nn.Module:
    if loss_type == "mse":
        return MaskedMSELoss()
    elif loss_type == "mae":
        return MaskedMAELoss()
    elif loss_type == "binweighted_mse":
        return MaskedMSEBinWeightedLoss(freqs)
    elif loss_type == "binweighted_mae":
        return MaskedMAEBinWeightedLoss(freqs)
    else:
        raise ValueError(f"Unknown loss_type={loss_type!r}")
