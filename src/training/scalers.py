"""Scaling transforms for spectral inputs, targets, and residuals."""

import torch
import numpy as np
from torch.utils.data import Dataset

class Scaler:
    """
    Abstract base class for data scalers.
    Provides a consistent API for fitting, transforming, and inverse-transforming data.
    """
    def __init__(self, epsilon: float = 1e-8):
        self.epsilon = epsilon

    def fit(self, dataset: Dataset, fit_on: str):
        """
        Fits the scaler by computing the necessary statistics from a dataset.

        Args:
            dataset (Dataset): An unscaled instance of ECMWFDataset.
            fit_on (str): Specifies which data to fit on.
                          Options: 'X', 'Y', 'Y-YM'.
        """
        raise NotImplementedError

    def transform(self, tensor: torch.Tensor) -> torch.Tensor:
        """Applies the forward scaling transformation."""
        raise NotImplementedError

    def inverse_transform(self, tensor: torch.Tensor) -> torch.Tensor:
        """Applies the inverse scaling transformation."""
        raise NotImplementedError


class IdentityScaler(Scaler):
    """
    A dummy scaler that performs no operation.
    Useful for disabling scaling while maintaining a consistent API.
    """
    def fit(self, dataset: Dataset, fit_on: str):
        """No fitting is required for the identity scaler."""
        pass

    def transform(self, tensor: torch.Tensor) -> torch.Tensor:
        """Returns the tensor unchanged."""
        return tensor

    def inverse_transform(self, tensor: torch.Tensor) -> torch.Tensor:
        """Returns the tensor unchanged."""
        return tensor


class LogZScoreScaler(Scaler):
    """
    Applies a log10 transform followed by z-score normalization.
    Computes a single global mean and std dev over the entire dataset.
    """
    def __init__(self, epsilon: float = 1e-8):
        super().__init__(epsilon)
        self.mean_log = None
        self.std_log = None

    def fit(self, dataset: Dataset, fit_on: str):
        """
        Computes the global mean and std of the log10-transformed data.
        """
        print(f"Fitting LogZScoreScaler on '{fit_on}'...")
        all_values = []
        for i in range(len(dataset)):
            # Use the shared dataset tuple without depending on context fields.
            data_sample = dataset[i]
            X, Y, M, _, _, YM = data_sample[:6]

            if fit_on == 'X':
                tensor = X
            elif fit_on == 'Y':
                tensor = Y
            elif fit_on == 'Y-YM':
                tensor = Y - YM
            else:
                raise ValueError(f"Unknown fit_on value: {fit_on}")

            # Fit target statistics only on valid buoy observations.
            mask = M if fit_on != 'X' else torch.ones_like(tensor).bool()
            valid_values = tensor[mask.bool()]
            all_values.append(valid_values)

        all_values_cat = torch.cat(all_values)
        log_values = torch.log10(all_values_cat + self.epsilon)

        self.mean_log = log_values.mean()
        self.std_log = log_values.std()
        print(f"  - Fit complete. Mean: {self.mean_log:.4f}, Std: {self.std_log:.4f}")

    def transform(self, tensor: torch.Tensor) -> torch.Tensor:
        if self.mean_log is None or self.std_log is None:
            raise RuntimeError("Scaler has not been fitted yet.")
        tensor_log = torch.log10(tensor + self.epsilon)
        return (tensor_log - self.mean_log) / (self.std_log + self.epsilon)

    def inverse_transform(self, tensor: torch.Tensor) -> torch.Tensor:
        if self.mean_log is None or self.std_log is None:
            raise RuntimeError("Scaler has not been fitted yet.")
        tensor_unscaled_log = (tensor * self.std_log) + self.mean_log
        return torch.pow(10, tensor_unscaled_log) - self.epsilon


class BinwiseZScoreScaler(Scaler):
    """
    Applies z-score normalization independently for each frequency bin.
    Handles both 4D (B,T,C,F) and 5D (B,C,T,F,D) tensors.
    """
    def __init__(self, epsilon: float = 1e-8):
        super().__init__(epsilon)
        self.mean_bin = None
        self.std_bin = None

    def fit(self, dataset: Dataset, fit_on: str):
        """
        Computes the mean and std for each frequency bin across the dataset.
        """
        print(f"Fitting BinwiseZScoreScaler on '{fit_on}'...")
        num_freq_bins = dataset.F
        all_values_per_bin = [[] for _ in range(num_freq_bins)]

        for i in range(len(dataset)):
            data_sample = dataset[i]
            _, Y, M, _, _, YM = data_sample[:6]

            if fit_on == 'Y':
                tensor = Y
            elif fit_on == 'Y-YM':
                tensor = Y - YM
            else:
                raise ValueError(f"BinwiseZScoreScaler only supports 'Y' or 'Y-YM', not '{fit_on}'")

            mask = M.bool()
            # The tensor from the dataset is 4D: (V, T, C, F)
            for f_idx in range(num_freq_bins):
                valid_values = tensor[..., f_idx][mask[..., f_idx]]
                all_values_per_bin[f_idx].append(valid_values)

        means, stds = [], []
        for f_idx in range(num_freq_bins):
            bin_values = torch.cat(all_values_per_bin[f_idx])
            means.append(bin_values.mean())
            stds.append(bin_values.std())

        # Store 1D bin statistics; reshape at transform time for each layout.
        self.mean_bin = torch.tensor(means)
        self.std_bin = torch.tensor(stds)
        print(f"  - Fit complete. Stats computed for {num_freq_bins} bins.")

    def _get_reshaped_stats(self, target_tensor: torch.Tensor):
        """Helper to reshape stats to match the target tensor's dimension."""
        device = target_tensor.device
        stats_shape = [1] * target_tensor.ndim
        # The frequency dimension is second-to-last in 5D (B,C,T,F,D) from collate
        # and last in 4D (V,T,C,F) from dataset.
        freq_dim_index = -2 if target_tensor.ndim == 5 else -1
        stats_shape[freq_dim_index] = self.mean_bin.numel()

        mean = self.mean_bin.view(stats_shape).to(device)
        std = self.std_bin.view(stats_shape).to(device)
        return mean, std

    def transform(self, tensor: torch.Tensor) -> torch.Tensor:
        if self.mean_bin is None or self.std_bin is None:
            raise RuntimeError("Scaler has not been fitted yet.")
        mean, std = self._get_reshaped_stats(tensor)
        return (tensor - mean) / (std + self.epsilon)

    def inverse_transform(self, tensor: torch.Tensor) -> torch.Tensor:
        if self.mean_bin is None or self.std_bin is None:
            raise RuntimeError("Scaler has not been fitted yet.")
        mean, std = self._get_reshaped_stats(tensor)
        return (tensor * std) + mean
