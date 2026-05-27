"""
pca_analysis.py — Error Structure Analysis via PCA.

This module performs Principal Component Analysis (PCA) to identify dominant 
error modes in the reference model and compares how the experiment model 
projects onto these modes.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.decomposition import PCA
from typing import Optional, Tuple
from .style import apply_paper_style, add_swell_windsea_line, set_frequency_ticks_linear, despine, save_figure

def _get_error_matrix(
    err_df: pd.DataFrame, 
    model_id: str, 
    variable: str
) -> Tuple[Optional[pd.DataFrame], Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Helper to pivot error dataframe into a matrix (Samples x Features).
    Features are flattened (Lead Time x Frequency).
    """
    mask = (err_df["model_id"] == model_id) & (err_df["variable"] == variable)
    sub = err_df[mask].copy()
    
    if sub.empty: 
        return None, None, None
    
    # Pivot: Index=Issuance, Cols=(Lead, Freq)
    mat = sub.pivot_table(index="issuance_time", columns=["lead_h", "f_idx"], values="err", aggfunc="mean")
    
    # Fill missing with 0 (assuming no error if missing, or strictly required for PCA)
    if mat.isna().any().any(): 
        mat = mat.fillna(0.0)
    
    # Extract Axis Metadata
    cols = mat.columns
    leads = cols.get_level_values("lead_h").unique().sort_values()
    f_idxs = cols.get_level_values("f_idx").unique().sort_values()
    
    # Map f_idx back to Frequency Hz
    f_map = sub[["f_idx", "freq"]].drop_duplicates("f_idx").set_index("f_idx")["freq"]
    freqs = f_map.loc[f_idxs].values
    
    return mat, leads, freqs


def plot_pca_structure(
    err_df: pd.DataFrame, 
    save_dir: Optional[str] = None,
    ref_model: str = "ecmwf_raw", 
    ref_label: str = "ECMWF",
    exp_model: str = "ecmwf_ml",
    exp_label: str = "ML",
    variable: str = "E", 
    n_modes: int = 10
):
    """
    Generates a research-grade PCA analysis plot 

[Image of PCA analysis plot]
.
    1. Fits PCA on Reference Model Errors.
    2. Projects Experiment Errors onto Reference Modes.
    3. Plots Scree (Energy) comparison and Mode Shapes (Heatmaps).

    Args:
        err_df: Error DataFrame.
        save_dir: Directory to save the output.
        ref_model: ID of reference model (PCA fitted on this).
        ref_label: Label for reference.
        exp_model: ID of experiment model (Projected onto Ref modes).
        exp_label: Label for experiment.
        variable: Variable to analyze (default 'E').
        n_modes: Number of PCA modes to retain.
    """
    
    # --- 1. Data Prep ---
    print(f"    -> Preparing PCA matrices for {variable}...")
    X_ref, leads, freqs = _get_error_matrix(err_df, ref_model, variable)
    X_exp, _, _         = _get_error_matrix(err_df, exp_model, variable)

    if X_ref is None or X_exp is None:
        print(f"Error: Missing data for models {ref_model} or {exp_model}. PCA aborted.")
        return

    # Align Columns (Ensure Exp has exact same features as Ref)
    X_exp = X_exp.reindex(columns=X_ref.columns).fillna(0.0)

    # --- 2. Fit PCA on Reference ---
    print(f"    -> Fitting PCA on {ref_label} ({ref_model})...")
    pca = PCA(n_components=n_modes)
    pca.fit(X_ref)
    
    # --- 3. Project & Calculate Energy ---
    scores_ref = pca.transform(X_ref)
    scores_exp = pca.transform(X_exp)
    
    var_ref = np.mean(scores_ref**2, axis=0)
    var_exp = np.mean(scores_exp**2, axis=0)
    var_ratios = pca.explained_variance_ratio_
    cum_var = np.cumsum(var_ratios)

    # --- 4. Plotting Setup ---
    apply_paper_style(large_text=True)

    fig = plt.figure(figsize=(9.5, 7.6))
    gs = fig.add_gridspec(
        5,
        n_modes,
        height_ratios=[1.55, 0.12, 1.45, 0.18, 0.14],
        left=0.07,
        right=0.96,
        top=0.86,
        bottom=0.10,
        hspace=0.30,
        wspace=0.12,
    )

    # --- A. Top Panel: Comparison Scree Plot ---
    ax_scree = fig.add_subplot(gs[0, :])
    
    modes = np.arange(1, n_modes + 1)
    width = 0.35
    
    # Bars
    ax_scree.bar(modes - width/2, var_ref, width, label=f'{ref_label} (Raw Error)', 
                 color='#d62728', alpha=0.85, edgecolor='k', linewidth=0.5)
    ax_scree.bar(modes + width/2, var_exp, width, label=f'{exp_label} (Projected Error)', 
                 color='#1f77b4', alpha=0.85, edgecolor='k', linewidth=0.5)

    # Percentage change annotations above the projected-error bars.
    positive_vals = np.concatenate([var_ref[var_ref > 0], var_exp[var_exp > 0]])
    min_positive = positive_vals.min() if positive_vals.size else 1e-12
    annotation_tops = []
    for mode, v_r, v_t in zip(modes, var_ref, var_exp):
        pct_change = ((v_t - v_r) / v_r * 100.0) if v_r > 0 else 0.0
        is_decrease = pct_change < 0
        txt_col = '#2ca02c' if is_decrease else '#d62728'
        marker = "▼" if is_decrease else "▲"
        y_pos = max(v_t, min_positive) * 1.35
        annotation_tops.append(y_pos)
        ax_scree.text(
            mode + width / 2,
            y_pos,
            f"{marker}{abs(pct_change):.0f}%",
            ha='center',
            va='bottom',
            fontsize=9,
            fontweight='bold',
            color=txt_col,
            clip_on=False,
        )
    
    # Cumulative Line (Right Axis)
    ax_cum = ax_scree.twinx()
    ax_cum.plot(modes, cum_var * 100, color='black', marker='o', linestyle='--', linewidth=1.6, 
                label=f'Cumulative Variance ({ref_label})')
    ax_cum.set_ylim(0, 105)
    ax_cum.set_ylabel("Cumulative variance [%]")
    
    # Styling Left Axis
    ax_scree.set_ylabel("Error energy (variance)")
    ax_scree.set_yscale('log')
    ax_scree.set_xticks(modes)
    ax_scree.set_xlim(0.5, n_modes + 0.5)
    
    # Y-Limits (Avoid clipping)
    max_val = max(var_ref.max(), var_exp.max(), max(annotation_tops, default=0.0))
    min_val = min_positive
    ax_scree.set_ylim(bottom=min_val * 0.5, top=max_val * 2.6) 
    
    ax_scree.grid(True, axis='y', which="both", linestyle=':', alpha=0.5)
    
    # Legend
    h1, l1 = ax_scree.get_legend_handles_labels()
    h2, l2 = ax_cum.get_legend_handles_labels()
    fig.legend(h1 + h2, l1 + l2, loc='upper center', bbox_to_anchor=(0.5, 0.975), ncol=3, frameon=False, fontsize=10.5)
    ax_scree.set_title(f"(a) PCA scree and cumulative variance", pad=10)
    despine(ax_scree)
    despine(ax_cum)

    # --- B. Bottom Panel: Mode Heatmaps ---
    freq_mask = freqs <= 0.30
    freqs_plot = freqs[freq_mask]
    X_grid, Y_grid = np.meshgrid(leads, freqs_plot)
    
    # Reshape components back to (Lead x Freq)
    all_comps = np.array([pca.components_[i].reshape(len(leads), len(freqs)).T for i in range(n_modes)])
    all_comps_plot = all_comps[:, freq_mask, :]
    global_lim = np.max(np.abs(all_comps_plot))

    heatmap_axes = []
    for i in range(n_modes):
        ax = fig.add_subplot(gs[2, i])
        heatmap_axes.append(ax)
        comp = all_comps_plot[i]
        
        im = ax.pcolormesh(X_grid, Y_grid, comp, cmap="RdBu_r", vmin=-global_lim, vmax=global_lim, shading='nearest')
        
        ax.set_title(f"Mode {i+1}\n({var_ratios[i]:.1%})", fontsize=11)
        ax.tick_params(axis='both', which='major', labelsize=10.5)
        ax.set_ylim(freqs_plot.min(), freqs_plot.max())
        add_swell_windsea_line(ax, 0.10)
        set_frequency_ticks_linear(ax, freqs_plot)
        despine(ax)
        
        if i == 0:
            ax.set_ylabel("Frequency [Hz]")
        else:
            ax.set_ylabel("")
            ax.set_yticklabels([])
            ax.tick_params(axis="y", labelleft=False)
        ax.set_xlabel("")

    # Shared x-axis label and colorbar
    ax_shared_xlabel = fig.add_subplot(gs[3, :])
    ax_shared_xlabel.axis("off")
    ax_shared_xlabel.text(0.5, 0.50, "Lead time [h]", ha="center", va="center", fontsize=12)

    cbar_ax = fig.add_subplot(gs[4, 1:-1])
    cbar = fig.colorbar(im, cax=cbar_ax, orientation='horizontal')
    cbar.set_label("Eigenvector amplitude", fontsize=12, labelpad=5)
    cbar.ax.tick_params(labelsize=11)
    
    # --- Save Logic ---
    if save_dir:
        # Create a dedicated subfolder
        sub_dir = os.path.join(save_dir, "pca_analysis")
        os.makedirs(sub_dir, exist_ok=True)
        
        fname = "pca_structure_comparison.png"
        save_path = os.path.join(sub_dir, fname)
        save_figure(fig, save_path)
        pdf_path = os.path.join(sub_dir, "pca_structure_comparison.pdf")
        save_figure(fig, pdf_path)
        print(f"✅ PCA Analysis saved to: {save_path}")

    plt.show()

def plot_pca_mode_swell(
    err_df: pd.DataFrame, 
    ref_model: str = "ecmwf_raw", 
    exp_model: str = "ecmwf_ml", 
    variable: str = "E", 
    mode_x: int = 8, 
    mode_y: int = 9,
    save_dir: Optional[str] = None
):
    """
    Generates a publication-grade 1x3 figure analyzing specific PCA error modes.
    Driven strictly by the phase errors (opposite signs) of the reference model.
    """
    print(f"Preparing deterministic PCA for Modes {mode_x} & {mode_y}...")
    
    # --- 1. Data Prep & PCA ---
    X_ref, leads, freqs = _get_error_matrix(err_df, ref_model, variable)
    X_exp, _, _ = _get_error_matrix(err_df, exp_model, variable)
    
    if X_ref is None or X_exp is None:
        print(f"Error: Missing data for models {ref_model} or {exp_model}. PCA aborted.")
        return
        
    X_exp = X_exp.reindex(columns=X_ref.columns).fillna(0.0)
    
    n_components = max(mode_x, mode_y)
    pca = PCA(n_components=n_components, svd_solver='full')
    pca.fit(X_ref)
    
    scores_ref = pca.transform(X_ref)
    scores_exp = pca.transform(X_exp)
    
    idx_x, idx_y = mode_x - 1, mode_y - 1
    pcX_ref, pcY_ref = scores_ref[:, idx_x], scores_ref[:, idx_y]
    pcX_exp, pcY_exp = scores_exp[:, idx_x], scores_exp[:, idx_y]
    
    # Calculate masks ONLY based on the Reference Model (ECMWF)
    mask_opp_ref = np.sign(pcX_ref) == -np.sign(pcY_ref)
    mask_same_ref = ~mask_opp_ref
    
    # Calculate Magnitudes
    mag_ref = pcX_ref**2 + pcY_ref**2
    mag_exp = pcX_exp**2 + pcY_exp**2

    # --- 2. Figure Setup ---
    apply_paper_style(large_text=True)
    fig = plt.figure(figsize=(9.5, 4.8))
    gs = fig.add_gridspec(
        2,
        3,
        height_ratios=[1.0, 0.12],
        width_ratios=[1.05, 1.0, 1.0],
        left=0.08,
        right=0.98,
        top=0.88,
        bottom=0.26,
        wspace=0.32,
        hspace=0.35,
    )
    
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[0, 2])

    # ==========================================================
    # PANEL (a): Spatial Structure (Mode X - Mode Y)
    # ==========================================================
    comp_x = pca.components_[idx_x].reshape(len(leads), len(freqs)).T
    comp_y = pca.components_[idx_y].reshape(len(leads), len(freqs)).T
    comp_diff = comp_x - comp_y
    freq_mask = freqs <= 0.30
    freqs_plot = freqs[freq_mask]
    comp_diff_plot = comp_diff[freq_mask, :]
    X_grid, Y_grid = np.meshgrid(leads, freqs_plot)

    global_lim = np.nanpercentile(np.abs(comp_diff_plot), 98)
    if not np.isfinite(global_lim) or global_lim == 0:
        global_lim = np.max(np.abs(comp_diff_plot))

    im = ax1.pcolormesh(X_grid, Y_grid, comp_diff_plot, cmap="RdBu_r",
                        vmin=-global_lim, vmax=global_lim, shading='nearest')
    add_swell_windsea_line(ax1, 0.10)
    ax1.set_ylim(freqs_plot.min(), freqs_plot.max())
    set_frequency_ticks_linear(ax1, freqs_plot)
    ax1.set_xlabel("Lead time [h]")
    ax1.set_ylabel("Frequency [Hz]")
    ax1.set_title(rf"(a) Mode {mode_x} $-$ Mode {mode_y}")
    despine(ax1)
    
    cax = fig.add_subplot(gs[1, 0])
    cbar = fig.colorbar(im, cax=cax, orientation="horizontal")
    cbar.ax.tick_params(labelsize=11)
    cbar.set_label("Amplitude", fontsize=12)

    # ==========================================================
    # PANEL (b): Mode Projection (Scatter)
    # ==========================================================
    # ECMWF (Reference) 
    ax2.scatter(pcX_ref[mask_same_ref], pcY_ref[mask_same_ref],
                c='#d62728', s=16, alpha=0.18, edgecolors='none', label='ECMWF same sign')
    ax2.scatter(pcX_ref[mask_opp_ref], pcY_ref[mask_opp_ref],
                c='#d62728', s=28, alpha=0.68, edgecolors='white', linewidths=0.25, label='ECMWF opposite sign')
    
    # ML-Ensemble (Experiment) - Highlighted based ONLY on ECMWF's mask
    ax2.scatter(pcX_exp[mask_same_ref], pcY_exp[mask_same_ref],
                c='#1f77b4', s=16, alpha=0.18, edgecolors='none', zorder=3, label='ML same sign')
    ax2.scatter(pcX_exp[mask_opp_ref], pcY_exp[mask_opp_ref],
                c='#1f77b4', s=28, alpha=0.68, edgecolors='white', linewidths=0.25, zorder=4, label='ML opposite sign')
    
    # Reference Lines - Using Global Max to prevent clipping the massive ML outliers
    lim = max(
        np.max(np.abs(pcX_ref)), np.max(np.abs(pcY_ref)),
        np.max(np.abs(pcX_exp)), np.max(np.abs(pcY_exp))
    ) * 1.05
    
    ax2.axhline(0, color='k', linestyle='-', alpha=0.22, lw=0.9)
    ax2.axvline(0, color='k', linestyle='-', alpha=0.22, lw=0.9)
    ax2.plot([-lim, lim], [-lim, lim], color='k', linestyle='--', alpha=0.25, lw=0.9)
    ax2.plot([-lim, lim], [lim, -lim], color='k', linestyle='--', alpha=0.25, lw=0.9)
    
    ax2.set_xlim(-lim, lim)
    ax2.set_ylim(-lim, lim)
    ax2.set_aspect('equal')
    ax2.set_xlabel(f"PC{mode_x} coefficient")
    ax2.set_ylabel(f"PC{mode_y} coefficient")
    ax2.set_title(f"(b) Error-mode projection")
    ax2.grid(True, linestyle=':', alpha=0.6)
    despine(ax2)
    
    # ==========================================================
    # PANEL (c): Histogram of Opposite Sign Magnitudes
    # ==========================================================
    # Filter both magnitudes using ONLY the ECMWF mask
    filt_mag_ref = mag_ref[mask_opp_ref]
    filt_mag_exp = mag_exp[mask_opp_ref]
    
    bins = np.histogram(np.hstack((filt_mag_ref, filt_mag_exp)), bins=40)[1]
    
    ax3.hist(filt_mag_ref, bins=bins, alpha=0.65, color='#d62728', histtype='stepfilled', linewidth=0.8)
    ax3.hist(filt_mag_exp, bins=bins, alpha=0.45, color='#1f77b4', histtype='stepfilled', linewidth=0.8)
    
    ax3.set_yscale('log')
    ax3.set_xlabel(f"Amplitude ($PC_{{{mode_x}}}^2 + PC_{{{mode_y}}}^2$)")
    ax3.set_ylabel("Count")
    
    ax3.set_title(f"(c) Dispersive-error magnitude")
    ax3.grid(True, linestyle=':', alpha=0.6)
    despine(ax3)

    ax2.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.22),
        ncol=2,
        frameon=False,
        fontsize=10.5,
        handletextpad=0.5,
        columnspacing=1.0,
    )

    # --- Global Formatting & Save ---
    if save_dir:
        sub_dir = os.path.join(save_dir, "pca_analysis")
        os.makedirs(sub_dir, exist_ok=True)
        save_path = os.path.join(sub_dir, "mode_error_analysis.pdf")
        
        save_figure(fig, save_path)
        print(f"✅ Figure saved successfully to: {save_path}")
        
    plt.show()


def get_worst_phase_error_dates(
    err_df: pd.DataFrame, 
    ref_model: str = "ecmwf_raw", 
    exp_model: str = "ecmwf_ml", 
    variable: str = "E", 
    mode_x: int = 8, 
    mode_y: int = 9,
    n_extremes: int = 5
):
    """
    Extracts extreme dates based ONLY on the reference model's phase shift (opposite signs).
    """
    X_ref, _, _ = _get_error_matrix(err_df, ref_model, variable)
    X_exp, _, _ = _get_error_matrix(err_df, exp_model, variable)
    X_exp = X_exp.reindex(columns=X_ref.columns).fillna(0.0)
    
    pca = PCA(n_components=max(mode_x, mode_y), svd_solver='full')
    pca.fit(X_ref)
    
    scores_ref = pca.transform(X_ref)
    scores_exp = pca.transform(X_exp)
    
    idx_x, idx_y = mode_x - 1, mode_y - 1
    pcX_ref, pcY_ref = scores_ref[:, idx_x], scores_ref[:, idx_y]
    pcX_exp, pcY_exp = scores_exp[:, idx_x], scores_exp[:, idx_y]
    
    # 1. CREATE MASK BASED ONLY ON ECMWF
    mask_opp_ref = np.sign(pcX_ref) == -np.sign(pcY_ref)
    
    mag_ref = pcX_ref**2 + pcY_ref**2
    mag_exp = pcX_exp**2 + pcY_exp**2
    
    # 2. FILTER BOTH MAGNITUDES USING ECMWF MASK
    filt_mag_ref = mag_ref[mask_opp_ref]
    filt_mag_exp = mag_exp[mask_opp_ref]
    filt_dates = X_ref.index[mask_opp_ref]
    
    # 3. RANK BASED ON THEIR RESPECTIVE MAGNITUDES WITHIN THAT SHARED SUBSET
    top_ml_dates = [filt_dates[i] for i in np.argsort(filt_mag_exp)[::-1][:n_extremes]]
    top_ecmwf_dates = [filt_dates[i] for i in np.argsort(filt_mag_ref)[::-1][:n_extremes]]
    
    return top_ml_dates, top_ecmwf_dates
