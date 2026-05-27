"""
dashboard.py — Forecast visualization and animation utilities.

This module provides research-grade dashboards for visualizing forecast issuances:
- plot_dashboard_hs_components: Detailed breakdown of Hs (Total, Swell, Wind Sea) + Spectra.
- plot_dashboard_bulk_params: Overview of Bulk Parameters (Hs, Tp, Tm02) + Spectra.
- generate_forecast_video: Automation to animate these dashboards over time.

Updates:
- Added 'is_ensemble' flag for Uncertainty Quantification (Mean +/- Std).
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.animation as animation
from typing import Optional, Callable, Tuple
from .style import (
    apply_paper_style,
    despine,
    add_swell_windsea_vline,
    save_figure,
    OBS_COLOR,
    ECMWF_COLOR,
    ML_COLOR,
    F_CUTOFF,
)

# ---------------------------------------------------------------------------
# Constants & Styling
# ---------------------------------------------------------------------------
M_RAW, M_ML = "ecmwf_raw", "ecmwf_ml"
C_OBS, C_RAW, C_ML = OBS_COLOR, ECMWF_COLOR, ML_COLOR
SPECTRUM_LEADS = (3, 60, 120)


def _default_dashboard_figsize(dashboard_func: Callable) -> Tuple[float, float]:
    name = getattr(dashboard_func, "__name__", "")
    if name == "plot_dashboard_hs_components":
        return (9.5, 8.2)
    if name == "plot_dashboard_bulk_params":
        return (9.5, 9.0)
    return (9.5, 8.2)

# ---------------------------------------------------------------------------
# Shared Helper Functions (Private)
# ---------------------------------------------------------------------------
def _plot_time_series(
    ax: plt.Axes,
    df_iss: pd.DataFrame,
    var_name: str,
    f_idx: int,
    label_y: str,
    unit: str = "m",
    show_xlabel: bool = False,
    is_main: bool = False,
    is_ensemble: bool = False
):
    """
    Plots a time-series comparison (Obs vs ECMWF vs ML).
    Supports Ensemble Uncertainty shading if is_ensemble=True.
    """
    # Filter by variable AND f_idx
    mask = (df_iss["variable"] == var_name) & (df_iss["f_idx"] == f_idx)
    df_p = df_iss[mask].sort_values("lead_h")
    
    if df_p.empty:
        ax.text(0.5, 0.5, f"No data for {var_name} (idx {f_idx})", 
                ha='center', va='center', transform=ax.transAxes)
        return

    # 1. Plot Uncertainty Band (Ensemble Only)
    # We do this first so it stays in the background
    if is_ensemble and "y_std" in df_p.columns:
        df_ml = df_p[df_p["model_id"] == M_ML]
        if not df_ml.empty:
            ax.fill_between(
                df_ml["lead_h"],
                df_ml["y_pred"] - df_ml["y_std"],
                df_ml["y_pred"] + df_ml["y_std"],
                color=C_ML, alpha=0.22, label="Ensemble spread (±1σ)" if is_main else ""
            )

    # 2. Plot Traces (Thicker lines for visibility)
    # Observer
    df_obs = df_p[df_p["model_id"]==M_RAW]
    ax.plot(df_obs["lead_h"], df_obs["y_true"], 
            c=C_OBS, ls="--", lw=1.8, label="Obs" if is_main else "")
    
    # Raw Model
    df_raw = df_p[df_p["model_id"]==M_RAW]
    ax.plot(df_raw["lead_h"], df_raw["y_pred"], 
            c=C_RAW, ls="-", lw=1.6, label="ECMWF" if is_main else "")
    
    # ML Model (Mean)
    df_ml = df_p[df_p["model_id"]==M_ML]
    ax.plot(df_ml["lead_h"],  df_ml["y_pred"],  
            c=C_ML,  ls="-", lw=1.8, label="ML-Ensemble" if (is_main and is_ensemble) else ("ML" if is_main else ""))
    
    # Styling
    for lead in SPECTRUM_LEADS:
        ax.axvline(lead, color="0.25", lw=0.9, ls=":", alpha=0.55, zorder=0)

    ax.set_ylabel(f"{label_y} [{unit}]")
    ax.tick_params(axis='both', which='major', labelsize=11)
    ax.grid(True, which='major', alpha=0.20, color='0.5', linestyle='-')
    despine(ax)
    
    if show_xlabel:
        ax.set_xlabel("Lead time [h]")
    else:
        ax.tick_params(labelbottom=False)

def _plot_spectrum(
    ax: plt.Axes,
    df_iss: pd.DataFrame,
    lead: int,
    is_leftmost: bool = False,
    is_ensemble: bool = False,
    title_text: Optional[str] = None,
):
    """
    Plots the energy density spectrum E(f) at a specific lead time.
    Supports Ensemble Uncertainty shading if is_ensemble=True.
    """
    df_s = df_iss[(df_iss["variable"] == "E") & (df_iss["lead_h"] == lead)].sort_values("freq")
    freqs = df_s["freq"].unique()
    
    if len(freqs) == 0:
        return

    # Helper to get the line for a specific model (handling duplicates via groupby)
    def get_curve(model_id, col):
        return df_s[df_s["model_id"]==model_id].groupby("freq")[col].first()

    # 1. Plot Uncertainty Band (Ensemble Only)
    if is_ensemble and "y_std" in df_s.columns:
        mu = get_curve(M_ML, "y_pred")
        sigma = get_curve(M_ML, "y_std")
        if not mu.empty:
            ax.fill_between(freqs, mu - sigma, mu + sigma, color=C_ML, alpha=0.22)

    # 2. Plot Traces
    ax.plot(freqs, get_curve(M_RAW, "y_true"), c=C_OBS, ls="--", lw=1.8)
    ax.plot(freqs, get_curve(M_RAW, "y_pred"), c=C_RAW, ls="-", lw=1.5)
    ax.plot(freqs, get_curve(M_ML,  "y_pred"), c=C_ML,  ls="-", lw=1.8)
    
    # Cutoff Line
    add_swell_windsea_vline(ax, F_CUTOFF)
    
    # Annotations (Swell / Wind Sea)
    y_min, y_max = ax.get_ylim()
    y_text = y_min + 0.94 * (y_max - y_min)
    ax.text(F_CUTOFF * 0.92, y_text, "Swell", ha='right', fontsize=9, color='0.35')
    ax.text(F_CUTOFF * 1.10, y_text, "Wind sea", ha='left', fontsize=9, color='0.35')
    
    # Styling
    ax.set_xscale("log")
    ax.set_xticks([0.04, 0.1, 0.2, 0.4])
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax.tick_params(axis='both', labelsize=11)
    ax.set_title(title_text if title_text is not None else f"+{lead} h", fontsize=12)
    ax.grid(True, which="both", alpha=0.2, ls=':', lw=0.5)
    despine(ax)
    
    if is_leftmost:
        ax.set_ylabel(r"$E(f)$ [m$^2$ Hz$^{-1}$]")
    ax.set_xlabel("Frequency [Hz]")


# ---------------------------------------------------------------------------
# Dashboard 1: Hs Components (Total, Swell, Wind Sea)
# ---------------------------------------------------------------------------
def plot_dashboard_hs_components(df_iss: pd.DataFrame, fig: plt.Figure, is_ensemble: bool = False):
    """
    Dashboard 1: Hs Breakdown.
    Args:
        is_ensemble (bool): If True, plots uncertainty bands (Mean +/- Std).
    """
    apply_paper_style(large_text=True)
    # Grid Layout
    gs = gridspec.GridSpec(4, 3, figure=fig, hspace=0.58, wspace=0.32, height_ratios=[1.0, 0.95, 0.95, 1.25])
    
    # Subplots (Time Series)
    ax_hs_total = fig.add_subplot(gs[0, :])
    ax_hs_swell = fig.add_subplot(gs[1, :], sharex=ax_hs_total)
    ax_hs_wind  = fig.add_subplot(gs[2, :], sharex=ax_hs_total) 
    
    # Subplots (Spectra)
    ax_spec_3   = fig.add_subplot(gs[3, 0])
    ax_spec_60  = fig.add_subplot(gs[3, 1]) 
    ax_spec_120 = fig.add_subplot(gs[3, 2])
    
    # --- Execute Plots ---
    # 1. Time Series
    _plot_time_series(ax_hs_total, df_iss, "Hs", 0, r"$H_s$", is_main=True, is_ensemble=is_ensemble)
    _plot_time_series(ax_hs_swell, df_iss, "Hs", 1, r"Swell $H_s$", is_ensemble=is_ensemble)
    _plot_time_series(ax_hs_wind,  df_iss, "Hs", 2, r"Wind sea $H_s$", show_xlabel=True, is_ensemble=is_ensemble)
    
    # 2. Spectra
    _plot_spectrum(ax_spec_3,   df_iss, 3, is_leftmost=True, is_ensemble=is_ensemble, title_text=r"(d) Spectra: +3 h")
    _plot_spectrum(ax_spec_60,  df_iss, 60, is_ensemble=is_ensemble, title_text=r"+60 h")
    _plot_spectrum(ax_spec_120, df_iss, 120, is_ensemble=is_ensemble, title_text=r"+120 h")
    handles, labels = ax_hs_total.get_legend_handles_labels()
    if handles:
        fig.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.58, 0.985),
            ncol=4,
            frameon=False,
            fontsize=10.5,
            handlelength=2.2,
            columnspacing=1.2,
        )
    ax_hs_total.set_title(r"(a) $H_s$", loc="left", pad=5)
    ax_hs_swell.set_title(r"(b) Swell $H_s$", loc="left", pad=4)
    ax_hs_wind.set_title(r"(c) Wind sea $H_s$", loc="left", pad=4)
    
    fig.subplots_adjust(top=0.88, left=0.09, right=0.985, bottom=0.09)


# ---------------------------------------------------------------------------
# Dashboard 2: Bulk Parameters (Hs, Tp, Tm02)
# ---------------------------------------------------------------------------
def plot_dashboard_bulk_params(df_iss: pd.DataFrame, fig: plt.Figure, is_ensemble: bool = False):
    """
    Dashboard 2: Bulk Parameters.
    Args:
        is_ensemble (bool): If True, plots uncertainty bands (Mean +/- Std).
    """
    apply_paper_style(large_text=True)
    gs = gridspec.GridSpec(
        4,
        3,
        figure=fig,
        height_ratios=[1.0, 1.0, 1.0, 1.35],
        hspace=0.62,
        wspace=0.34,
    )
    
    # Subplots (Time Series)
    ax_hs_total   = fig.add_subplot(gs[0, :])
    ax_tp_total   = fig.add_subplot(gs[1, :], sharex=ax_hs_total)
    ax_tm02_total = fig.add_subplot(gs[2, :], sharex=ax_hs_total)
    
    # Subplots (Spectra)
    ax_spec_3   = fig.add_subplot(gs[3, 0])
    ax_spec_60  = fig.add_subplot(gs[3, 1]) 
    ax_spec_120 = fig.add_subplot(gs[3, 2])
    
    # --- Execute Plots ---
    # 1. Time Series
    _plot_time_series(ax_hs_total,   df_iss, "Hs",   0, r"$H_s$", unit="m", is_main=True, is_ensemble=is_ensemble)
    _plot_time_series(ax_tp_total,   df_iss, "Tp",   0, r"$T_p$", unit="s", is_ensemble=is_ensemble)
    _plot_time_series(ax_tm02_total, df_iss, "Tm02", 0, r"$T_{m02}$", unit="s", show_xlabel=True, is_ensemble=is_ensemble)
    
    # 2. Spectra
    _plot_spectrum(ax_spec_3,   df_iss, 3, is_leftmost=True, is_ensemble=is_ensemble, title_text=r"(d) $E(f)$: +3 h")
    _plot_spectrum(ax_spec_60,  df_iss, 60, is_ensemble=is_ensemble, title_text=r"+60 h")
    _plot_spectrum(ax_spec_120, df_iss, 120, is_ensemble=is_ensemble, title_text=r"+120 h")
    handles, labels = ax_hs_total.get_legend_handles_labels()
    if handles:
        fig.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.58, 0.985),
            ncol=4,
            frameon=False,
            fontsize=10.5,
            handlelength=2.2,
            columnspacing=1.2,
        )
    ax_hs_total.set_title(r"(a) $H_s$", loc="left", pad=5)
    ax_tp_total.set_title(r"(b) $T_p$", loc="left", pad=5)
    ax_tm02_total.set_title(r"(c) $T_{m02}$", loc="left", pad=5)
    
    fig.subplots_adjust(top=0.88, left=0.09, right=0.985, bottom=0.08)


# ---------------------------------------------------------------------------
# Animation / Video Generation
# ---------------------------------------------------------------------------
def generate_forecast_video(
    dashboard_func: Callable,
    df: pd.DataFrame,
    save_dir: str,
    output_filename: str = "dashboard.gif",
    fps: int = 1,
    is_ensemble: bool = False,
    figsize: Optional[Tuple[float, float]] = None,
):
    """
    Wrapper to generate a GIF for ANY dashboard function.
    
    Args:
        dashboard_func: The plotting function (e.g., plot_dashboard_hs_components)
        df: The error dataframe containing multiple issuances.
        save_dir: Directory to save the video.
        output_filename: Filename.
        fps: Frames per second.
        is_ensemble: Enable uncertainty visualization.
    """
    # 1. Setup Path
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, output_filename)
    
    # 2. Get Issuances
    issuances = sorted(df["issuance_time"].unique())
    print(f"🎥 Generating {output_filename} for {len(issuances)} frames (Ensemble={is_ensemble})...")

    # 3. Setup Figure
    if figsize is None:
        figsize = _default_dashboard_figsize(dashboard_func)
    fig = plt.figure(figsize=figsize)

    def update(issuance):
        fig.clear()
        subset_df = df[df["issuance_time"] == issuance]
        # Pass is_ensemble to the dashboard function
        dashboard_func(subset_df, fig, is_ensemble=is_ensemble)

    # 4. Render
    ani = animation.FuncAnimation(fig, update, frames=issuances, repeat=True)
    ani.save(save_path, writer='pillow', fps=fps)
    
    plt.close(fig)
    print(f"✅ Video saved: {save_path}")


def generate_extreme_error_dashboards(
    err_df: pd.DataFrame, 
    dates: list, 
    save_dir: str, 
    prefix: str = "ML-Ensemble",
    format: str = "pdf"
):
    # Force the output to be inside the pca_analysis directory
    base_out = os.path.join(save_dir, "pca_analysis")
    sub_folder = f"extreme_errors_{prefix.lower().replace('-', '_')}"
    final_out_dir = os.path.join(base_out, sub_folder)
    os.makedirs(final_out_dir, exist_ok=True)
    
    for i, date in enumerate(dates):
        subset_df = err_df[err_df['issuance_time'] == date].copy()
        if subset_df.empty: continue
        
        fig = plt.figure(figsize=(9.5, 9.0))
        plot_dashboard_bulk_params(subset_df, fig, is_ensemble=True)
        
        safe_date = str(date).replace(" ", "_").replace(":", "")
        save_path = os.path.join(final_out_dir, f"rank_{i+1:02d}_{safe_date}.{format}")

        save_figure(fig, save_path, also_png=(format.lower() == "pdf"))
        plt.close(fig) # Prevent memory leaks
