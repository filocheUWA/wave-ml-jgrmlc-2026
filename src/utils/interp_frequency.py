"""Frequency-grid interpolation utilities for spectral harmonization."""

import torch

def interp_frequency(X, target_freqs, initial_freqs):
    
    """
    Linear interpolation for monotonically increasing data points.

    Interpolate values from `initial_freqs` to `target_freqs`.

    Args:
        target_freqs (Tensor): Target frequencies at which to interpolate.
        initial_freqs (Tensor): Frequencies of known data points.
        X (Tensor): Values of the function at `initial_freqs`.

    Returns:
        Tensor: Interpolated values at `target_freqs`.
    """
    m = (X[1:] - X[:-1]) / (initial_freqs[1:] - initial_freqs[:-1])
    b = X[:-1] - (m * initial_freqs[:-1])

    indicies = torch.sum(torch.ge(target_freqs[:, None], initial_freqs[None, :]), 1) - 1
    indicies = torch.clamp(indicies, 0, len(m) - 1)

    return m[indicies] * target_freqs + b[indicies]
