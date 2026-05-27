"""Shared paper-style helpers for manuscript figures."""

from __future__ import annotations

import os
from typing import Iterable, Sequence

import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

OBS_COLOR = "black"
ECMWF_COLOR = "#e41a1c"
ML_COLOR = "#377eb8"
F_CUTOFF = 0.10


def apply_paper_style(*, large_text: bool = False) -> None:
    params = {
        "font.family": "serif",
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "grid.linewidth": 0.5,
        "grid.alpha": 0.25,
        "lines.linewidth": 1.3,
    }
    if large_text:
        params.update(
            {
                "font.size": 11,
                "axes.titlesize": 13,
                "axes.labelsize": 12,
                "xtick.labelsize": 11,
                "ytick.labelsize": 11,
                "legend.fontsize": 11,
                "axes.linewidth": 1.0,
            }
        )
    else:
        params.update(
            {
                "font.size": 9,
                "axes.titlesize": 10,
                "axes.labelsize": 9,
                "xtick.labelsize": 8,
                "ytick.labelsize": 8,
                "legend.fontsize": 8,
                "axes.linewidth": 0.8,
            }
        )
    plt.rcParams.update(params)


def despine(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def add_swell_windsea_line(ax, y: float = F_CUTOFF) -> None:
    ax.axhline(y, color="black", linestyle="--", linewidth=0.8, alpha=0.75, zorder=5)


def add_swell_windsea_vline(ax, x: float = F_CUTOFF) -> None:
    ax.axvline(x, color="black", linestyle="--", linewidth=0.8, alpha=0.75, zorder=5)


def _filter_ticks(candidates: Sequence[float], lo: float, hi: float) -> list[float]:
    return [tick for tick in candidates if lo <= tick <= hi]


def set_frequency_ticks_linear(ax, freqs: Iterable[float] | None = None) -> None:
    candidates = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
    if freqs is not None:
        values = list(freqs)
        lo, hi = (min(values), max(values)) if values else ax.get_ylim()
    else:
        lo, hi = ax.get_ylim()
    ticks = _filter_ticks(candidates, lo, hi)
    if ticks:
        ax.set_yticks(ticks)
        ax.set_yticklabels([f"{tick:.2f}" for tick in ticks])


def set_frequency_ticks_log(ax, freqs: Iterable[float] | None = None) -> None:
    candidates = [0.05, 0.10, 0.20, 0.50]
    if freqs is not None:
        values = list(freqs)
        lo, hi = (min(values), max(values)) if values else ax.get_ylim()
    else:
        lo, hi = ax.get_ylim()
    ticks = _filter_ticks(candidates, lo, hi)
    if ticks:
        ax.set_yticks(ticks)
        ax.get_yaxis().set_major_formatter(ScalarFormatter())
        ax.set_yticklabels([f"{tick:.2f}" for tick in ticks])


def save_figure(fig, save_path: str, *, also_png: bool = False) -> None:
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    if also_png:
        root, ext = os.path.splitext(save_path)
        if ext.lower() != ".png":
            fig.savefig(f"{root}.png", dpi=300, bbox_inches="tight")
