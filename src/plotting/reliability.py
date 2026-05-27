"""
reliability.py — Ensemble calibration and reliability metrics.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from .style import apply_paper_style, despine, save_figure

def plot_stratified_ranks(
    checkpoint_dir: str,
    obs_df: pd.DataFrame,
    config: list,
    leads_to_plot: list,
    save_dir: str = "."
):
    """
    Generates a grid of Stratified Rank Histograms.
    
    Args:
        checkpoint_dir: Directory containing the member inference CSVs.
        obs_df: DataFrame containing the reference observations (y_true).
        config: List of dictionaries defining the variables to plot.
        leads_to_plot: List of forecast lead times.
        save_dir: Directory to save the output PDF.
    """
    print(f"📊 Generating {len(config)}x{len(leads_to_plot)} Rank Histogram Grid...")
    
    apply_paper_style()

    # --- 2. Data Loading & Merging ---
    csv_files = sorted([f for f in os.listdir(checkpoint_dir) if f.endswith(".csv")])
    if not csv_files:
        print(f"❌ No CSV files found in {checkpoint_dir}")
        return
        
    join_keys = ["issuance_time", "lead_h", "f_idx", "variable"]
    full_member_df = None
    
    for i, f in enumerate(csv_files):
        path = os.path.join(checkpoint_dir, f)
        df = pd.read_csv(path)
        
        row_masks = []
        for cfg in config:
            mask = (df["variable"] == cfg["var"]) & (df["f_idx"] == cfg["f_idx"])
            row_masks.append(df[mask])
        
        df_filtered = pd.concat(row_masks)
        
        if "issuance_time" in df_filtered.columns:
            df_filtered["issuance_time"] = pd.to_datetime(df_filtered["issuance_time"])
        
        member_col = f"member_{i}"
        df_subset = df_filtered[join_keys + ["y_pred"]].rename(columns={"y_pred": member_col})
        
        if full_member_df is None:
            full_member_df = df_subset
        else:
            full_member_df = pd.merge(full_member_df, df_subset, on=join_keys, how="inner")

    # Get Observations
    obs_ref = obs_df[obs_df["model_id"] == "ecmwf_ml"][join_keys + ["y_true"]].drop_duplicates(subset=join_keys)

    final_df = pd.merge(full_member_df, obs_ref, on=join_keys, how="inner")
    member_cols = [c for c in final_df.columns if c.startswith("member_")]
    n_bins = len(member_cols) + 1

    # --- 3. Plotting ---
    fig, axes = plt.subplots(len(config), len(leads_to_plot), figsize=(8.0, 6.0), sharey='row')
    
    if len(config) == 1 and len(leads_to_plot) == 1:
        axes = np.array([[axes]])
    elif len(config) == 1:
        axes = axes[np.newaxis, :]
    elif len(leads_to_plot) == 1:
        axes = axes[:, np.newaxis]

    for i, cfg in enumerate(config):
        for j, lead in enumerate(leads_to_plot):
            ax = axes[i, j]
            
            subset = final_df[
                (final_df["variable"] == cfg["var"]) & 
                (final_df["f_idx"] == cfg["f_idx"]) & 
                (final_df["lead_h"] == lead)
            ]
            
            if subset.empty:
                ax.text(0.5, 0.5, "Insufficient Data", ha='center')
                continue

            ens_vals = subset[member_cols].values
            obs_vals = subset["y_true"].values.reshape(-1, 1)
            ranks = (ens_vals < obs_vals).sum(axis=1)
            
            ax.hist(ranks, bins=np.arange(n_bins+1)-0.5, density=True,
                    color='#377eb8', edgecolor='black', alpha=0.7, linewidth=0.6)
            
            ax.axhline(1.0/n_bins, color='black', linestyle='--', linewidth=1.5, label="Ideal")
            
            if i == 0:
                ax.set_title(f"Lead: t+{lead} h", fontweight="bold", pad=8)
            if j == 0: 
                ax.set_ylabel(f"{cfg['label']}\nProbability", labelpad=8)
            if i == len(config) - 1: 
                ax.set_xlabel("Rank")
            
            tick_step = 5 if n_bins > 10 else 1
            ax.set_xticks(range(0, n_bins, tick_step))
            ax.grid(axis='y', alpha=0.2, linestyle=':')
            despine(ax)

    plt.tight_layout()
    
    # --- Updated Save Logic ---
    out_dir = os.path.join(save_dir, "uncertainty_quantification")
    os.makedirs(out_dir, exist_ok=True)
    save_path = os.path.join(out_dir, 'rank_histogram.pdf')
    
    save_figure(fig, save_path)
    print(f"✅ Figure saved successfully to: {save_path}")
    
    plt.show()


def plot_spectral_uncertainty_diagnostics(
    checkpoint_dir: str,
    obs_df: pd.DataFrame,
    save_dir: str = "."
):
    """
    Generates a 1x2 spectral Hovmöller diagram for Probabilistic Error (CRPS) 
    and Spread-Skill Ratio (SSR).
    
    Args:
        checkpoint_dir: Directory containing the member inference CSVs.
        obs_df: DataFrame containing the reference observations (y_true).
        save_dir: Directory to save the output PDF.
    """
    print("🌊 Computing Spectral Uncertainty Diagnostics (CRPS & SSR)...")
    
    # --- 1. Typography Setup ---
    apply_paper_style()

    # --- 2. Data Loading ---
    csv_files = sorted([f for f in os.listdir(checkpoint_dir) if f.endswith(".csv")])
    if not csv_files:
        print(f"❌ No CSV files found in {checkpoint_dir}")
        return
        
    join_keys = ["issuance_time", "lead_h", "f_idx"]
    full_member_df = None
    
    for i, f in enumerate(csv_files):
        path = os.path.join(checkpoint_dir, f)
        df = pd.read_csv(path)
        df = df[df["variable"] == "E"] 
        
        if "issuance_time" in df.columns:
            df["issuance_time"] = pd.to_datetime(df["issuance_time"])
            
        member_col = f"member_{i}"
        df_subset = df[join_keys + ["y_pred"]].rename(columns={"y_pred": member_col})
        
        if full_member_df is None:
            full_member_df = df_subset
        else:
            full_member_df = pd.merge(full_member_df, df_subset, on=join_keys, how="inner")

    obs_ref = obs_df[
        (obs_df["variable"] == "E") & 
        (obs_df["model_id"] == "ecmwf_ml")
    ][join_keys + ["y_true"]].drop_duplicates(subset=join_keys)
    
    full_member_df["f_idx"] = full_member_df["f_idx"].astype(float).astype(int)
    obs_ref["f_idx"] = obs_ref["f_idx"].astype(float).astype(int)

    final_df = pd.merge(full_member_df, obs_ref, on=join_keys, how="inner")
    if final_df.empty: 
        print("❌ ERROR: No data survived the merge.")
        return

    # --- 3. Vectorized Base Calculations ---
    member_cols = [c for c in final_df.columns if c.startswith("member_")]
    final_df["ens_mean"] = final_df[member_cols].mean(axis=1)
    final_df["ens_spread"] = final_df[member_cols].std(axis=1)
    final_df["error_sq"] = (final_df["ens_mean"] - final_df["y_true"])**2

    # Fast CRPS
    ens_vals = final_df[member_cols].values
    obs_vals = final_df["y_true"].values
    N = ens_vals.shape[1]
    ens_sorted = np.sort(ens_vals, axis=1)
    mae_term = np.mean(np.abs(ens_vals - obs_vals[:, None]), axis=1)
    i_idx = np.arange(N)
    coef = (2 * i_idx - N + 1) / (N ** 2)
    spread_term = np.sum(ens_sorted * coef, axis=1)
    final_df["crps"] = mae_term - spread_term

    # --- 4. Groupby Aggregation ---
    records = []
    for (f_idx, lead), group in final_df.groupby(["f_idx", "lead_h"]):
        # SSR
        rmse = np.sqrt(group["error_sq"].mean())
        ssr = group["ens_spread"].mean() / rmse if rmse > 1e-6 else np.nan
        
        # CRPS
        crps_val = group["crps"].mean()
        
        records.append({
            "f_idx": f_idx, "lead_h": lead, 
            "ssr": ssr, "crps": crps_val
        })
                
    master_df = pd.DataFrame(records)

    # --- 5. Create 1x2 Subplots ---
    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.6), sharex=True, sharey=True)
    
    metrics = [
        {"col": "crps", "ax": axes[0], "cmap": "viridis", "vmin": 0.0, "vmax": np.nanpercentile(master_df["crps"], 95), "title": "(a) Probabilistic error (CRPS)", "extend": "max", "label": "CRPS"},
        {"col": "ssr", "ax": axes[1], "cmap": "RdBu_r", "vmin": 0.0, "vmax": 2.0, "title": "(b) Ensemble spread / RMSE (SSR)", "extend": "max", "label": "SSR"},
    ]

    for m in metrics:
        matrix = master_df.pivot(index="f_idx", columns="lead_h", values=m["col"])
        X = matrix.columns.values
        Y = matrix.index.values
        Z = matrix.values
        
        ax = m["ax"]
        pcm = ax.pcolormesh(X, Y, Z, cmap=m["cmap"], vmin=m["vmin"], vmax=m["vmax"], shading='nearest')
        
        cb = fig.colorbar(pcm, ax=ax, pad=0.02, extend=m["extend"])
        cb.set_label(m["label"], rotation=270, labelpad=12)
        
        ax.set_title(m["title"], pad=8)
        ax.grid(True, axis='x', color='black', alpha=0.3, linestyle=':')
        despine(ax)
        ax.set_xlabel("Lead time [h]")

    # --- 6. Tick Formatting (Hz only) ---
    f0 = 0.03453 
    multiplier = 1.1 
    tick_indices = Y[::4] 
    tick_labels = [f"{f0 * (multiplier ** idx):.3f}" for idx in tick_indices]
    
    # Apply Y-labels only to the left-most edge
    axes[0].set_yticks(tick_indices)
    axes[0].set_yticklabels(tick_labels)
    axes[0].set_ylabel("Frequency [Hz]")

    plt.tight_layout()
    
    # --- 7. Save Setup ---
    out_dir = os.path.join(save_dir, "uncertainty_quantification")
    os.makedirs(out_dir, exist_ok=True)
    save_path = os.path.join(out_dir, "spectral_crps_ssr.pdf")
    
    save_figure(fig, save_path)
    print(f"✅ Figure saved successfully to: {save_path}")
    
    plt.show()
