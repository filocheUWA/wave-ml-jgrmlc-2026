"""Utility helpers for reporting and testing."""
from __future__ import annotations
import torch
import torch.nn as nn


def size_report(m: nn.Module) -> str:
    """Return a one-line model size report (params, buffers, ~MB @ fp32)."""
    params_total = sum(p.numel() for p in m.parameters())
    params_train = sum(p.numel() for p in m.parameters() if p.requires_grad)
    buffers_total = sum(b.numel() for b in m.buffers())
    bytes_total = 4 * (params_total + buffers_total)  # fp32
    return (f"[model size] params={params_total:,} "
            f"(trainable={params_train:,}), buffers={buffers_total:,} "
            f"→ ~{bytes_total/1e6:.2f} MB")


def grad_norm(m: nn.Module) -> float:
    """Compute global L2 grad-norm across all parameters (for debugging)."""
    total = 0.0
    for p in m.parameters():
        if p.grad is not None:
            total += float(p.grad.detach().pow(2).sum().item())
    return total ** 0.5
