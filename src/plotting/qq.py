"""
qq.py — Quantile-Quantile (Q-Q) Visualization.

This module provides functions to generate empirical Q-Q plots for validating 
forecast distributions against observations, specifically for Total, Swell and Wind Sea partitions.
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
PART_MAP = {0: "Total", 1: "Swell", 2: "Wind Sea"}
TARGET_PCTS = [50, 90, 99]

# Default Colors
C_REF = "#d32f2f"  # Red
C_EXP = "#1976d2"  # Blue


def plot_qq_empirical(
    err_df: pd.DataFrame,
    lead_time: int,
    variable: str = "Hs",  
    save_dir: Optional[str] = None,
    show_plot: bool = False,
    ref_model: str = "ecmwf_raw",
    ref_label: str = "ECMWF",
    exp_model: str = "ecmwf_ml",
    exp_label: str = "ML"
):
    """
    Generates Q-Q plots using DIRECT EMPIRICAL DATA (No Interpolation).
    Compares Reference vs Experiment models against Observations.
    
    Layout: 1 Row x 3 Columns (Total, Swell, Wind Sea)
    """
    # --- 1. Determine Units ---
    apply_paper_style()
    if variable == "Hs":
        unit = "m"
    elif variable in ["Tp", "Tm02", "Tm01", "Tm"]:
        unit = "s"
    else:
        unit = ""

    # --- 2. Filter Data ---
    df_lead = err_df[
        (err_df["lead_h"] == lead_time) & 
        (err_df["variable"] == variable)
    ].copy()

    if df_lead.empty: 
        print(f"Warning: No '{variable}' data found for lead {lead_time}h")
        return

    # --- 3. Plotting Loop ---
    fig, axes = plt.subplots(1, 3, figsize=(7.8, 2.9))
    plt.subplots_adjust(wspace=0.22)
    
    sorted_parts = sorted(PART_MAP.items()) 
    
    for i, (part_idx, part_name) in enumerate(sorted_parts):
        ax = axes[i]
        
        # A. Data Prep: Filter by partition
        mask_part = df_lead["f_idx"] == part_idx
        df_ref = df_lead[mask_part & (df_lead["model_id"] == ref_model)]
        df_exp = df_lead[mask_part & (df_lead["model_id"] == exp_model)]

        # B. Inner Merge (Sync Timestamps)
        merged = pd.merge(
            df_ref[["issuance_time", "y_true", "y_pred"]],
            df_exp[["issuance_time", "y_pred"]],
            on="issuance_time",
            suffixes=("_ref", "_exp"),
            how="inner"
        ).dropna()
        
        n = len(merged)
        
        if n < 10: 
            ax.text(0.5, 0.5, "Insufficient Data", ha='center', transform=ax.transAxes)
            ax.set_title(f"({chr(97+i)}) {part_name}")
            continue

        # C. EMPIRICAL SORT
        obs_sorted = np.sort(merged["y_true"])
        ref_sorted = np.sort(merged["y_pred_ref"])
        exp_sorted = np.sort(merged["y_pred_exp"])
        
        # D. Plotting Axis Limits
        max_val = max(obs_sorted.max(), ref_sorted.max(), exp_sorted.max()) * 1.05
        min_val_raw = min(obs_sorted.min(), ref_sorted.min(), exp_sorted.min())
        
        # Dynamic Minimum Flooring
        if variable == "Hs":
            min_bound = 0.0
        else:
            # Add a 5% buffer below the actual minimum, ensuring it doesn't go negative
            min_bound = max(0.0, min_val_raw * 0.95)
            
        ax.plot([min_bound, max_val], [min_bound, max_val], 'k--', alpha=0.35, lw=1.0, label="Ideal")
        
        # Step Plot
        ax.step(obs_sorted, ref_sorted, where='mid', color=C_REF, lw=1.5, label=ref_label, alpha=0.85)
        ax.step(obs_sorted, exp_sorted, where='mid', color=C_EXP, lw=1.5, label=exp_label, alpha=0.90)

        # E. Annotations (Percentiles)
        for pct in TARGET_PCTS:
            idx = int(n * (pct / 100.0)) - 1
            idx = max(0, min(idx, n - 1))
            
            x_val = obs_sorted[idx]
            y_exp = exp_sorted[idx]
            y_ref = ref_sorted[idx]
            
            # Markers
            ax.scatter(x_val, y_ref, color=C_REF, s=18, edgecolor='white', linewidth=0.4, zorder=4)
            ax.scatter(x_val, y_exp, color=C_EXP, s=18, edgecolor='white', linewidth=0.4, zorder=4)
            
            # Label
            label_txt = f"$P_{{{pct}}}$"
            ax.annotate(label_txt, xy=(x_val, y_exp), xytext=(-18, 6),
                        textcoords='offset points', fontsize=7, color='black')

        # F. Rug Plot (Shifted slightly above min_bound)
        rug_y = min_bound + (max_val - min_bound) * 0.015
        ax.scatter(obs_sorted, np.zeros_like(obs_sorted) + rug_y, 
                   marker='|', color='gray', alpha=0.45, s=22)

        # Styling
        if variable == "Hs":
            symbol = r"$H_s$"
        elif variable == "Tp":
            symbol = r"$T_p$"
        elif variable == "Tm02":
            symbol = r"$T_{m02}$"
        else:
            symbol = variable
        ax.set_title(f"({chr(97+i)}) {part_name if part_name != 'Wind Sea' else 'Wind sea'}")
        ax.set_xlabel(f"Observed {symbol} [{unit}]" if unit else f"Observed {symbol}")
        ax.set_ylabel(f"Predicted {symbol} [{unit}]" if unit else f"Predicted {symbol}")
        
        # Set new bounds
        ax.set_xlim(min_bound, max_val)
        ax.set_ylim(min_bound, max_val)
        ax.set_aspect('equal', adjustable='box')
        ax.grid(True, linestyle=':', alpha=0.6)
        despine(ax)
        
        if i == 0:
            ax.legend(loc='upper left', frameon=True, fontsize=8)

    # --- Save ---
    if save_dir:
        sub_dir = os.path.join(save_dir, "qqplots", variable)
        os.makedirs(sub_dir, exist_ok=True)
        
        fname = f"qq_{variable}_lead{lead_time:03d}.png"
        save_path = os.path.join(sub_dir, fname)
        save_figure(fig, save_path)
        print(f"✅ Saved Q-Q Plot: {save_path}")

    # --- Show or Close ---
    if show_plot:
        plt.show()
    else:
        plt.close(fig)
