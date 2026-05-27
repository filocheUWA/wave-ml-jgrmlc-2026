"""Optimizer parameter grouping and learning-rate schedules."""

import torch
import torch.nn as nn

from torch.optim.lr_scheduler import LambdaLR
import math

def build_param_groups_for_adamw(model: nn.Module, weight_decay: float):
    """Separate decay and no-decay parameters for AdamW."""
    decay, no_decay = [], []
    for p in model.parameters():
        if not p.requires_grad:
            continue
        # Biases and normalization scales are conventionally left unregularized.
        (no_decay if p.ndim <= 1 else decay).append(p)
    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]

def build_warmup_cosine_scheduler(optimizer, warmup_epochs: int, total_epochs: int, min_lr_factor: float):
    """Build the linear warmup plus cosine annealing schedule used in training."""
    def lr_lambda(current_epoch: int):
        if current_epoch < warmup_epochs and warmup_epochs > 0:
            return float(current_epoch + 1) / float(max(1, warmup_epochs))
        # cosine phase
        progress = (current_epoch - warmup_epochs) / float(max(1, total_epochs - warmup_epochs))
        progress = min(max(progress, 0.0), 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_factor + (1.0 - min_lr_factor) * cosine
    return LambdaLR(optimizer, lr_lambda=lr_lambda)
