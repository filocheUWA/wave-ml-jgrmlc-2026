"""Offline computation of log-spectral scaler statistics."""

import json
import torch
import numpy as np
import os
from tqdm import tqdm
from torch.utils.data import DataLoader
from .dataset import ECMWFDataset
from src.training.data_management import collate

def compute_and_save_scaler_stats(
    train_ds: ECMWFDataset,
    save_path: str,
    *,
    batch_size: int = 16,
    num_workers: int = 4,
    epsilon: float = 1e-6,
):
    """Compute global log10 mean/std from valid target and baseline spectra."""
    print("Iterating through the training dataset to compute scaler statistics...")

    loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate,
    )

    all_values = []
    for batch in tqdm(loader, desc="Collecting data"):
        _, Y, M, _, _, YM = batch
        Y_squeezed = Y.squeeze(1).squeeze(-1)
        M_squeezed = M.squeeze(1).squeeze(-1)
        YM_squeezed = YM.squeeze(1).squeeze(-1)

        valid_mask = M_squeezed > 0.5
        if valid_mask.any():
            all_values.append(Y_squeezed[valid_mask].cpu())
            all_values.append(YM_squeezed[valid_mask].cpu())

    if not all_values:
        raise ValueError("No valid data found in the dataset to compute scaler statistics.")

    all_values_tensor = torch.cat(all_values)
    log_transformed_values = torch.log10(all_values_tensor + epsilon)

    mean_log = torch.mean(log_transformed_values).item()
    std_log = torch.std(log_transformed_values).item()

    print(f"\nComputed Statistics (log10 scale):")
    print(f"  Mean: {mean_log:.4f}")
    print(f"  Std Dev: {std_log:.4f}")

    scaler_stats = {
        'mean_log': mean_log,
        'std_log': std_log,
    }
    
    # Store plain JSON so the statistics are portable across environments.
    with open(save_path, 'w') as f:
        json.dump(scaler_stats, f, indent=4)
        
    print(f"\nScaler statistics saved to: {save_path}")
