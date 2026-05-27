"""Baseline neural models sharing the SpecX training interface."""

import torch.nn as nn
import torch

from .gates import TimeSeqGateBlockIO
from .conv_theta import conv3d_1x1

class MLPBaseline(nn.Module):
    """
    A simple MLP baseline that processes the (Time, Freq) grid as a flat vector.

    This model ignores the 2D structure of the input spectrum and learns a direct
    mapping from the flattened input to the flattened output. It is a good test
    for the basic data flow and loss calculation framework.
    """
    def __init__(self, input_size=(40, 29, 36), **kwargs):
        """Accepts SpecX parameters for compatibility, but only uses input_size."""
        super().__init__()
        # This attribute is needed for the training script's input-selection logic
        self.input_mode = kwargs.get("input_mode")
        
        T, F, _ = input_size
        n_features = T * F
        
        self.net = nn.Sequential(
            nn.Linear(n_features, n_features * 2),
            nn.ReLU(),
            nn.Linear(n_features * 2, n_features)
        )
        self.T = T
        self.F = F

    def forward(self, x_spec: torch.Tensor | None = None, x_coeff: torch.Tensor | None = None, **kwargs) -> torch.Tensor:
        """
        Takes the 1D spectrum (x_coeff), flattens it, passes it through an MLP,
        and reshapes it back to the original 5D tensor format.
        """
        if x_coeff is None:
            raise ValueError("MLPBaseline requires the 'x_coeff' (Y_model) input.")

        B, C, T, F, _ = x_coeff.shape
        
        # Reshape for MLP: (B, C, T, F, 1) -> (B*C, T*F)
        x = x_coeff.squeeze(-1).view(B * C, T * F)
        
        # Process through MLP
        residual = self.net(x)
        
        # Reshape back to original format: (B*C, T*F) -> (B, C, T, F, 1)
        residual = residual.view(B, C, T, F).unsqueeze(-1)

        # In a residual training mode, this residual would be added to the original x_coeff
        # In a direct mode, this is the final prediction
        return residual

# Helper class for the improved ConvBaseline
class ResidualBlock(nn.Module):
    """A standard residual block with two 2D convolutional layers."""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, padding_mode='replicate')
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, padding_mode='replicate')
        
        # 1x1 conv for the skip connection if channel counts differ
        self.skip = nn.Conv2d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        residual = self.skip(x)
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        out += residual
        out = self.relu(out)
        return out

class ConvBaseline(nn.Module):
    """
    An improved CNN baseline that processes the (Time, Freq) grid as an image.

    This version uses 'replicate' padding to mitigate border effects and a deeper,
    bottleneck architecture with residual connections for increased expressive power.
    """
    def __init__(self, n_coeff_ch: int = 1, **kwargs):
        """Accepts SpecX parameters for compatibility."""
        super().__init__()
        # This attribute is needed for the training script's input-selection logic
        self.input_mode = kwargs.get("input_mode")
        
        ch1 = 32  # Wider than the original 16
        ch2 = 64  # Bottleneck with more channels

        # Residual bottleneck architecture for the 1D spectral grid.
        self.in_proj = nn.Conv2d(n_coeff_ch, ch1, kernel_size=3, padding=1, padding_mode='replicate')
        
        self.layer1 = ResidualBlock(ch1, ch2)
        self.layer2 = ResidualBlock(ch2, ch1)

        self.out_proj = nn.Conv2d(ch1, n_coeff_ch, kernel_size=3, padding=1, padding_mode='replicate')

    def forward(self, x_spec: torch.Tensor | None = None, x_coeff: torch.Tensor | None = None, **kwargs) -> torch.Tensor:
        """
        Takes the 1D spectrum (x_coeff), treats it as a 2D image, passes it
        through the ResNet, and returns the result in the original 5D format.
        """
        if x_coeff is None:
            raise ValueError("ConvBaseline requires the 'x_coeff' (Y_model) input.")
        
        x = x_coeff.squeeze(-1)
        
        # Forward pass through the residual convolutional blocks.
        y = self.in_proj(x)
        y = self.layer1(y)
        y = self.layer2(y)
        residual = self.out_proj(y)
        
        return (x + residual).unsqueeze(-1)


class FakeSpecX(nn.Module):
    """
    Fake SpecX that just returns the 1D coefficient input (Y_model).
    Same input/output interface as SpecX for sanity checks and establishing a baseline.
    """
    def __init__(self, **kwargs):
        """Accepts all SpecX parameters to act as a drop-in replacement, but ignores them."""
        super().__init__()
        # This attribute is needed for the training script's input-selection logic
        self.input_mode = kwargs.get("input_mode")

    def forward(self, x_spec: torch.Tensor | None = None, x_coeff: torch.Tensor | None = None, **kwargs) -> torch.Tensor:
        """
        Returns the x_coeff tensor, ensuring it has the 5D shape the training pipeline expects.
        This baseline is intended for use in 'coeffs' or 'fused' mode.
        """
        if x_coeff is None:
            raise ValueError("FakeSpecX is a baseline that requires the 'x_coeff' (Y_model) input.")

        # The training script expects a 5D output, but the input might be 4D
        if x_coeff.dim() == 4:
            return x_coeff.unsqueeze(-1)
        return x_coeff


# Gated residual baseline using the same 5D spectral layout as SpecX.
class ResNetGateBaseline(nn.Module):
    """
    A ResNet-style baseline using the advanced TimeSeqGateBlockIO from gates.py.

    This model processes the (Time, Freq) grid using 3D convolutions (with a
    singleton dimension for Theta) and attention-like gating mechanisms. It serves
    as a much stronger baseline than the simple ConvNet.
    """
    def __init__(self, n_coeff_ch: int = 1, input_size: tuple[int, int, int] = (40, 29, 1), **kwargs):
        """Accepts SpecX parameters for compatibility."""
        super().__init__()
        self.input_mode = kwargs.get("input_mode")
        hidden_channels = 64 # A bit more capacity than the simple ConvBaseline
        n_blocks = 4         # Stack two residual gate blocks

        # 1. An initial convolution to project input channels to hidden channels
        self.in_proj = conv3d_1x1(n_coeff_ch, hidden_channels)

        # 2. A stack of residual gating blocks
        self.gate_blocks = nn.ModuleList(
            [TimeSeqGateBlockIO(hidden_channels, hidden_channels, input_size, ktheta=1) for _ in range(n_blocks)]
        )
        
        # 3. A final convolution to project back to the original channel count
        self.out_proj = conv3d_1x1(hidden_channels, n_coeff_ch)

    def forward(self, x_spec: torch.Tensor | None = None, x_coeff: torch.Tensor | None = None, **kwargs) -> torch.Tensor:
        """
        Takes the 5D spectrum (x_coeff), passes it through a small ResNet of
        gating blocks, and adds the final output to the original input.
        """
# Duplicate baseline definition below is preserved for compatibility.
class ResNetGateBaseline(nn.Module):
    """
    A ResNet-style baseline using the advanced TimeSeqGateBlockIO from gates.py.

    This model processes the (Time, Freq) grid using 3D convolutions (with a
    singleton dimension for Theta) and attention-like gating mechanisms. It serves
    as a much stronger baseline than the simple ConvNet.
    """
    def __init__(self, n_coeff_ch: int = 1, input_size: tuple[int, int, int] = (40, 29, 1), **kwargs):
        """Accepts SpecX parameters for compatibility."""
        super().__init__()
        self.input_mode = kwargs.get("input_mode")
        hidden_channels = 32 # A bit more capacity than the simple ConvBaseline
        n_blocks = 2         # Stack two residual gate blocks

        # 1. An initial convolution to project input channels to hidden channels
        self.in_proj = conv3d_1x1(n_coeff_ch, hidden_channels)

        # 2. A stack of residual gating blocks
        self.gate_blocks = nn.ModuleList(
            [TimeSeqGateBlockIO(hidden_channels, hidden_channels, input_size, ktheta=1) for _ in range(n_blocks)]
        )
        
        # 3. A final convolution to project back to the original channel count
        self.out_proj = conv3d_1x1(hidden_channels, n_coeff_ch)

    def forward(self, x_spec: torch.Tensor | None = None, x_coeff: torch.Tensor | None = None, **kwargs) -> torch.Tensor:
        """
        Takes the 5D spectrum (x_coeff), passes it through a small ResNet of
        gating blocks, and adds the final output to the original input.
        """
        if x_coeff is None:
            raise ValueError("ResNetGateBaseline requires the 'x_coeff' (Y_model) input.")

        # x_coeff is already in the correct 5D shape: (B, C, T, F, 1)
        
        # Project to hidden channels
        y = self.in_proj(x_coeff)
        
        # Pass through gating blocks
        for block in self.gate_blocks:
            y = block(y)
            
        # Project back to original channels
        residual = self.out_proj(y)

        # Final residual connection
        return x_coeff + residual
        if x_coeff is None:
            raise ValueError("ResNetGateBaseline requires the 'x_coeff' (Y_model) input.")

        # x_coeff is already in the correct 5D shape: (B, C, T, F, 1)
        
        # Project to hidden channels
        y = self.in_proj(x_coeff)
        
        # Pass through gating blocks
        for block in self.gate_blocks:
            y = block(y)
            
        # Project back to original channels
        residual = self.out_proj(y)

        # Final residual connection
        return x_coeff + residual


# Helper block for the pure convolutional ResNet baseline.
class ResidualBlock(nn.Module):
    """A standard residual block with two 2D convolutional layers."""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        
        # Projection for the skip connection if channel counts differ
        self.skip = nn.Conv2d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        residual = self.skip(x)
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        out += residual
        out = self.relu(out)
        return out

# Pure convolutional ResNet baseline for the 1D spectral grid.
class DeepConvResNet(nn.Module):
    """
    A deeper, pure convolutional ResNet baseline with a bottleneck structure.
    This model does not use any of the advanced gating mechanisms.
    """
    def __init__(self, n_coeff_ch: int = 1, **kwargs):
        """Accepts SpecX parameters for compatibility."""
        super().__init__()
        self.input_mode = kwargs.get("input_mode")
        
        ch1 = 16
        ch2 = 32
        ch3 = 64 # Max filters in the middle

        self.in_proj = nn.Conv2d(n_coeff_ch, ch1, kernel_size=3, padding=1)

        self.layer1 = ResidualBlock(ch1, ch2)
        self.layer2 = ResidualBlock(ch2, ch3)
        self.layer3 = ResidualBlock(ch3, ch2)
        self.layer4 = ResidualBlock(ch2, ch1)

        self.out_proj = nn.Conv2d(ch1, n_coeff_ch, kernel_size=3, padding=1)

    def forward(self, x_spec: torch.Tensor | None = None, x_coeff: torch.Tensor | None = None, **kwargs) -> torch.Tensor:
        if x_coeff is None:
            raise ValueError("DeepConvResNet requires the 'x_coeff' (Y_model) input.")
        
        # Input is (B, C, T, F, 1), squeeze to (B, C, T, F) for Conv2d
        x = x_coeff.squeeze(-1)
        
        # Pass through the network
        y = self.in_proj(x)
        y = self.layer1(y)
        y = self.layer2(y)
        y = self.layer3(y)
        y = self.layer4(y)
        residual = self.out_proj(y)
        
        # Add final residual and unsqueeze back to 5D format
        return (x + residual).unsqueeze(-1)
