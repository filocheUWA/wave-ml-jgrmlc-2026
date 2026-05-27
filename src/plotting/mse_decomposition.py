"""
mse_decomposition.py — Error Component Analysis.

This module analyzes the Mean Squared Error (MSE) by decomposing it into 
Bias² (Systematic Error) and Variance (Random Error) for specific wave partitions.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Optional
from .style import apply_paper_style, despine, save_figure

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
# Partition ID to Name mapping
PART_MAP = {1: "Swell", 2: "Wind sea"}

def plot_mse_decomposition(
    err_df: pd.DataFrame,
    save_path: Optional[str] = None,
    ref_model: str = "ecmwf_raw",
    ref_label: str = "ECMWF",
    exp_model: str = "ecmwf_ml",
    exp_label: str = "ML"
):
    """
    Decomposes MSE into Bias² and Variance for Swell and Wind Sea partitions.
    Generates a 2x2 stacked bar chart comparison.
    """
    apply_paper_style()
    # --- 0. Data Standardization ---
    df_clean = err_df.copy()
    rename_map = {'lead_time': 'lead_h', 'partition': 'f_idx'}
    df_clean.rename(columns={k: v for k, v in rename_map.items() if k in df_clean.columns}, inplace=True)

    # --- 1. Configuration ---
    # Colors: Variance=Light, Bias²=Dark/Saturated
    colors = {
        ref_model: {"var": "#ffcccb", "bias2": "#d32f2f"}, # Red theme (Ref)
        exp_model: {"var": "#bbdefb", "bias2": "#1976d2"}  # Blue theme (Exp)
    }

    model_labels = {ref_model: ref_label, exp_model: exp_label}
    target_models = [ref_model, exp_model]

    # --- 2. Data Preparation ---
    # Filter for relevant data (Hs only, Swell/Sea partitions)
    mask = (
        (df_clean["variable"] == "Hs") & 
        (df_clean["f_idx"].isin(PART_MAP.keys())) &
        (df_clean["model_id"].isin(target_models))
    )
    df = df_clean[mask].copy()
    
    if df.empty:
        print("Warning: No data found for Hs partitions 1 & 2 for the specified models.")
        return

    # Drop missing ground truth
    df = df[df["y_true"] > 1e-4]
    
    # Calculate Error
    df["error"] = df["y_pred"] - df["y_true"]
    
    # --- STRICT COMPUTATION ---
    # MSE = Bias^2 + Variance
    stats = (
        df.groupby(["model_id", "f_idx", "lead_h"])["error"]
        .agg(
            mean="mean",           # Bias
            var=lambda x: np.var(x, ddof=0) # Variance
        )
        .reset_index()
    )
    
    stats["bias2"] = stats["mean"] ** 2
    stats["mse"] = stats["bias2"] + stats["var"]

    # --- 3. Plotting ---
    fig, axes = plt.subplots(2, 2, figsize=(7.8, 5.6), sharex=True, sharey="row")
    
    leads = np.sort(stats["lead_h"].unique())
    # Adjust bar width based on lead time resolution
    bar_width = (leads[1] - leads[0]) * 0.68 if len(leads) > 1 else 1
    
    partitions = [1, 2] 
    
    for row_idx, part_idx in enumerate(partitions):
        part_name = PART_MAP[part_idx]
        
        # Calculate Y-max for this row to keep scales consistent between models
        row_data = stats[stats["f_idx"] == part_idx]
        if row_data.empty:
            continue
        y_max = row_data["mse"].max() * 1.1

        for col_idx, mod_id in enumerate(target_models):
            ax = axes[row_idx, col_idx]
            
            subset = stats[
                (stats["model_id"] == mod_id) & 
                (stats["f_idx"] == part_idx)
            ].sort_values("lead_h")
            
            if subset.empty: 
                ax.text(0.5, 0.5, "No Data", ha='center', transform=ax.transAxes)
                continue
                
            X = subset["lead_h"]
            Y_var = subset["var"]
            Y_bias = subset["bias2"]
            
            # Stacked Bars
            # Variance (Base)
            ax.bar(X, Y_var, width=bar_width, label="Variance (Random)", 
                   color=colors[mod_id]["var"], edgecolor='none', alpha=0.9)
            # Bias² (Top)
            ax.bar(X, Y_bias, width=bar_width, bottom=Y_var, label="Bias² (Systematic)", 
                   color=colors[mod_id]["bias2"], edgecolor='none', alpha=0.9)
            
            # Styling
            panel = chr(97 + row_idx * 2 + col_idx)
            title_str = f"({panel}) {part_name}: {model_labels.get(mod_id, mod_id)}"
            ax.set_title(title_str, fontsize=10)
            ax.grid(True, axis='y', linestyle='--', alpha=0.18)
            ax.set_ylim(0, y_max)
            despine(ax)
            
            # Axis Labels
            if col_idx == 0: ax.set_ylabel(r"MSE of $H_s$ [m$^2$]")
            if row_idx == 1: ax.set_xlabel("Lead time [h]")
                
            # Internal Legend (Only need it once per figure, top-left is standard)
            if row_idx == 0 and col_idx == 0:
                ax.legend(loc="upper left", framealpha=0.95, fontsize=8)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 0.985))
        axes[0, 0].legend_.remove()
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    
    # Save
    if save_path:
        save_figure(fig, save_path)
        print(f"✅ MSE Decomposition plot saved to: {save_path}")
    
    plt.show()
