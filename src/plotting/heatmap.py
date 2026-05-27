"""
heatmap.py — 2D Error Analysis Visualization.

This module provides functions to generate high-dimensional error heatmaps
(Bias and Standard Deviation) across Lead Time and Frequency.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.gridspec as gridspec
from typing import Optional
from .style import apply_paper_style, add_swell_windsea_line, set_frequency_ticks_log, despine, save_figure

def plot_heatmap_bias_std(
    df_eval: pd.DataFrame,
    save_path: Optional[str] = None,
    ref_model: str = "ecmwf_raw",
    ref_label: str = "ECMWF",
    exp_model: str = "ecmwf_ml",
    exp_label: str = "ML",
    variable: str = "E"
):
    """
    Generates a high-quality 2D error heatmap (Bias & Std) for a specific variable.
    Compares two models (Reference vs Experiment) across Lead Time and Frequency.

    Args:
        df_eval: Error dataframe containing 'model_id', 'lead_h', 'freq', 'err'.
        save_path: Full path to save the figure (optional).
        ref_model: ID of the reference model (default: 'ecmwf_raw').
        ref_label: Display label for reference (default: 'ECMWF').
        exp_model: ID of the experiment model (default: 'ecmwf_ml').
        exp_label: Display label for experiment (default: 'ML').
        variable: Variable to plot (default: 'E').
    """
    # --- 1. Data Preparation ---
    apply_paper_style()
    df_spec = df_eval[df_eval["variable"] == variable].copy()
    if df_spec.empty:
        print(f"Warning: No data found for variable '{variable}'. Skipping heatmap.")
        return

    # Build Model List & Label Map
    unique_models = df_spec["model_id"].unique()
    model_list = []
    
    # Only add models if they exist in the data
    if ref_model in unique_models: model_list.append(ref_model)
    if exp_model in unique_models: model_list.append(exp_model)
    
    if not model_list:
        print(f"Warning: Neither {ref_model} nor {exp_model} found in data.")
        return

    labels_map = {ref_model: ref_label, exp_model: exp_label}

    # Compute Statistics
    print("Computing 2D statistics for heatmaps...")
    stats = (
        df_spec.groupby(["model_id", "lead_h", "freq"])["err"]
        .agg(["mean", "std"])
        .reset_index()
    )

    def get_grid(mid, metric):
        subset = stats[stats["model_id"] == mid]
        if subset.empty: return None, None, None
        matrix = subset.pivot(index="freq", columns="lead_h", values=metric).sort_index()
        # Drop rows (freqs) if fully NaN
        matrix = matrix.dropna(how='all', axis=0)
        return matrix.columns.values, matrix.index.values, matrix.values

    # --- 2. Figure Setup ---
    fig = plt.figure(figsize=(7.8, 5.9), constrained_layout=True)
    # Width ratios: Plot 1, Plot 2, Colorbar
    gs = gridspec.GridSpec(2, 3, figure=fig, width_ratios=[1, 1, 0.05])

    # Shared Limits (across both models for fair comparison)
    max_bias = stats["mean"].abs().max()
    clim_bias = (-max_bias, max_bias)
    
    max_std = stats["std"].max()
    clim_std = (0, max_std)

    # Styling Constants
    LABEL_SIZE = 9
    TITLE_SIZE = 10
    TICK_SIZE = 8
    
    # --- 3. Plotting Loop ---
    
    # Row 0: Bias
    for col, mid in enumerate(model_list):
        if col > 1: break # Safety limit (2 columns max)
        ax = fig.add_subplot(gs[0, col])
        
        X, Y, Z = get_grid(mid, "mean")
        if Z is None: continue
        
        im_b = ax.pcolormesh(X, Y, Z, cmap="RdBu_r", vmin=clim_bias[0], vmax=clim_bias[1], shading='nearest')
        
        panel = chr(97 + col)
        ax.set_title(f"({panel}) Bias: {labels_map.get(mid, mid)}", fontsize=TITLE_SIZE)
        
        ax.set_yscale("log")
        ax.set_ylim(Y.min(), Y.max())
        ax.tick_params(axis='both', which='major', labelsize=TICK_SIZE)
        add_swell_windsea_line(ax, 0.10)
        despine(ax)
        
        # Y-Axis Labels (Only on left plot)
        if col == 0:
            ax.set_ylabel("Frequency [Hz]", fontsize=LABEL_SIZE)
            set_frequency_ticks_log(ax, Y)
        else:
            ax.set_yticklabels([])
        
        # X-Axis Labels (Hidden for top row)
        ax.tick_params(labelbottom=False)

    # Bias Colorbar (Right column)
    cax_b = fig.add_subplot(gs[0, 2])
    cb_b = plt.colorbar(im_b, cax=cax_b)
    cb_b.set_label(r"Mean error [m$^2$ Hz$^{-1}$]", fontsize=LABEL_SIZE)
    cb_b.ax.tick_params(labelsize=TICK_SIZE)

    # Row 1: Std
    for col, mid in enumerate(model_list):
        if col > 1: break
        ax = fig.add_subplot(gs[1, col])
        
        X, Y, Z = get_grid(mid, "std")
        if Z is None: continue
        
        im_s = ax.pcolormesh(X, Y, Z, cmap="inferno", vmin=clim_std[0], vmax=clim_std[1], shading='nearest')
        
        panel = chr(99 + col)
        ax.set_title(f"({panel}) Std.: {labels_map.get(mid, mid)}", fontsize=TITLE_SIZE)
        
        ax.set_yscale("log")
        ax.set_ylim(Y.min(), Y.max())
        ax.tick_params(axis='both', which='major', labelsize=TICK_SIZE)
        ax.set_xlabel("Lead time [h]", fontsize=LABEL_SIZE)
        add_swell_windsea_line(ax, 0.10)
        despine(ax)
        
        # Integer formatting for Lead Time
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f"{int(x)}"))
        
        # Y-Axis Labels (Only on left plot)
        if col == 0:
            ax.set_ylabel("Frequency [Hz]", fontsize=LABEL_SIZE)
            set_frequency_ticks_log(ax, Y)
        else:
            ax.set_yticklabels([])

    # Std Colorbar (Right column)
    cax_s = fig.add_subplot(gs[1, 2])
    cb_s = plt.colorbar(im_s, cax=cax_s)
    cb_s.set_label(r"Standard deviation [m$^2$ Hz$^{-1}$]", fontsize=LABEL_SIZE)
    cb_s.ax.tick_params(labelsize=TICK_SIZE)
    
    # Save
    if save_path:
        print(f"Saving heatmap to {save_path}...")
        save_figure(fig, save_path)
        
    plt.show()
