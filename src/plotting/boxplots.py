"""
boxplots.py — Grouped box-plot utilities and summary generation.

This module provides plotting functions for error analysis:
- plot_err_box_by_freq : grouped boxplots of error vs frequency (fixed lead)
- plot_err_box_vs_lead : grouped boxplots of error vs lead-time (scalar stats)
- create_summary_boxplots: High-level wrapper to generate and save a standard suite of plots.
"""

import os
from typing import Optional, Sequence, Tuple, List, Dict, Any
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import PathPatch
from .style import apply_paper_style, despine


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
DEFAULT_LABELS = ["Forecast", "Bias correction", "ML correction"]
DEFAULT_COLORS = ["#e41a1c", "#4daf4a", "#377eb8"]  # red, green, blue


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------
def _parse_freq_column(df: pd.DataFrame) -> np.ndarray:
    """Extract numeric frequency grid (Hz), sorted ascending."""
    if "freq" in df.columns:
        freqs = df["freq"].astype(float).unique()
    elif "x_label" in df.columns:
        freqs = (
            df["x_label"]
            .astype(str)
            .str.replace(" Hz", "", regex=False)
            .str.replace("Hz", "", regex=False)
            .str.replace("f=", "", regex=False)
            .astype(float)
            .unique()
        )
    else:
        raise ValueError("Need 'freq' or parseable 'x_label' in err_df.")
    return np.sort(freqs)


def _metric_to_canonical(metric: str, df_cols: Sequence[str]) -> str:
    """Map metric aliases to canonical table columns."""
    alias = {
        "raw": "err", "raw_err": "err",
        "abs": "abs_err", "absolute": "abs_err",
        "rel": "rel_err", "relative": "rel_err", "relative_err": "rel_err",
    }
    m = alias.get(metric, metric)
    if m not in df_cols:
        raise ValueError(
            f"Metric '{metric}'→'{m}' not found in DataFrame. "
            f"Available: {sorted(set(df_cols))}"
        )
    return m


def _units_for(variable: str) -> Optional[str]:
    """Return unit strings for y-axis labeling."""
    return {
        "E": "m$^2$/Hz", "Hs": "m", "Tm01": "s", "Tm02": "s", "Tp": "s",
    }.get(variable, None)


def _ylabel(variable: str, metric_col: str) -> str:
    """Compose a LaTeX-friendly ylabel."""
    if variable == "E":
        units = "m$^2$ Hz$^{-1}$"
    else:
        units = _units_for(variable)
    if metric_col == "rel_err":
        return f"|Δ{variable}|/{variable}"
    if variable == "E":
        base = {"err": r"ΔE(f)", "abs_err": r"|ΔE(f)|"}
    else:
        base = {"err": f"Δ{variable}", "abs_err": f"|Δ{variable}|"}
    text = base.get(metric_col, metric_col.replace("_", " "))
    return f"{text} [{units}]" if units else text


def _title_text(variable: str, metric_col: str, lead_time: int) -> str:
    """Default title for frequency boxplots."""
    base = {"err": f"Δ{variable}", "abs_err": f"|Δ{variable}|", "rel_err": f"|Δ{variable}|/{variable}"}
    return f"{base.get(metric_col, metric_col)} spectrum at +{lead_time} h"


def _auto_title(variable: str, metric_col: str, part_label: Optional[str]) -> str:
    """Default title for lead-time boxplots."""
    base = {"err": f"Δ{variable}", "abs_err": f"|Δ{variable}|", "rel_err": f"|Δ{variable}|/{variable}"}
    t = f"{base.get(metric_col, metric_col)} vs lead"
    if variable == "Hs" and part_label:
        t += f" (partition: {part_label})"
    elif part_label:
        t += f" [{part_label}]"
    return t


# ---------------------------------------------------------------------------
# Plot 1: Error vs FREQUENCY
# ---------------------------------------------------------------------------
def plot_err_box_by_freq(
    err_df: pd.DataFrame,
    *,
    variable: str = "E",
    model_ids: Optional[Sequence[str]] = None,
    metric: str = "abs_err",
    lead_time: int = 120,
    labels: Optional[Sequence[str]] = None,
    colors: Optional[Sequence[str]] = None,
    desired_xticks: Tuple[float, ...] = (0.0345, 0.059, 0.1, 0.14, 0.25, 0.49),
    vertical_lines: bool = True,
    ylim: Optional[Tuple[float, float]] = None,
    ax: Optional[plt.Axes] = None,
    base_log_width: float = 0.035,
    alpha: float = 0.5,
    add_legend: bool = True,
    auto_title: bool = False,
    title: Optional[str] = None,
    title_fmt: Optional[str] = None,
    title_kwargs: Optional[Dict[str, Any]] = None,
):
    """Grouped boxplots of error vs frequency at a fixed lead time."""
    df = err_df.loc[(err_df["variable"] == variable) & (err_df["lead_h"] == lead_time)].copy()
    if df.empty:
        raise ValueError(f"No rows for variable='{variable}' at lead_h={lead_time} in err_df.")

    metric_col = _metric_to_canonical(metric, df.columns)

    if "model_id" in df.columns:
        if model_ids is None: model_ids = list(df["model_id"].unique())
        else: model_ids = [m for m in model_ids if m in df["model_id"].unique()]
    else:
        model_ids = ["_single_"]
        df = df.assign(model_id=model_ids[0])

    freqs = _parse_freq_column(df)
    log_f = np.log10(freqs)
    min_gap = np.diff(log_f).min() if len(log_f) > 1 else 0.3

    n_models = len(model_ids)
    labels = list(labels) if labels is not None else DEFAULT_LABELS[:n_models]
    colors = list(colors) if colors is not None else DEFAULT_COLORS[:n_models]

    apply_paper_style(large_text=True)
    if ax is None: _, ax = plt.subplots(figsize=(9.0, 4.4))

    group_half = min(base_log_width * n_models / 2, min_gap * 0.35)
    model_half = group_half / n_models
    offsets_log = np.linspace(-group_half + model_half, group_half - model_half, n_models)
    legend_handles = []

    for m, (mid, lab, col) in enumerate(zip(model_ids, labels, colors)):
        d_m = df[df["model_id"] == mid]
        eps_f = []
        for f in freqs:
            if "freq" in d_m.columns:
                mask_f = np.isclose(d_m["freq"].astype(float).values, f)
            else:
                xl = d_m["x_label"].astype(str).str.replace(r"[^\d\.]", "", regex=True).astype(float)
                mask_f = np.isclose(xl.values, f)
            
            vals = d_m.loc[mask_f, metric_col].to_numpy(dtype=float)
            eps_f.append(vals[~np.isnan(vals)])

        pos = 10 ** (log_f + offsets_log[m])
        width = 10 ** (log_f + model_half) - 10 ** (log_f - model_half)
        
        non_empty = [i for i, arr in enumerate(eps_f) if arr.size > 0]
        if non_empty:
            bp = ax.boxplot(
                [eps_f[i] for i in non_empty],
                positions=pos[non_empty], widths=width[non_empty],
                patch_artist=True, showfliers=False,
                boxprops=dict(facecolor=col, edgecolor=col, linewidth=1.2),
                medianprops=dict(color=col, linewidth=1.4),
                whiskerprops=dict(color=col, linewidth=1.2),
                capprops=dict(color=col, linewidth=1.2),
            )
            for patch in bp["boxes"]: patch.set_alpha(alpha)

        legend_handles.append(PathPatch([], facecolor=col, edgecolor=col, alpha=alpha, label=lab))

    ax.set_xscale("log")
    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel(_ylabel(variable, metric_col))

    xt = [x for x in desired_xticks if freqs.min() <= x <= freqs.max()]
    if xt:
        ax.set_xticks(xt)
        ax.set_xticklabels([str(f) for f in xt])
        if vertical_lines:
            for x in xt: ax.axvline(x, color="grey", linewidth=0.4, alpha=0.3, zorder=0)

    if ylim: ax.set_ylim(*ylim)
    pad = 0.12
    ax.set_xlim(freqs.min() * (1 - pad), freqs.max() * (1 + pad))
    ax.grid(True, which="both", linestyle="--", linewidth=0.5)

    if add_legend:
        ax.legend(handles=legend_handles, loc="upper right", frameon=False)
    despine(ax)

    if title: ax.set_title(title, **(title_kwargs or {}))
    elif auto_title:
        metric_text = {"err": f"Δ{variable}", "abs_err": f"|Δ{variable}|", "rel_err": f"|Δ{variable}|/{variable}"}.get(metric_col, metric_col)
        text = title_fmt.format(var=variable, metric=metric_col, metric_text=metric_text, lead=lead_time) if title_fmt else _title_text(variable, metric_col, lead_time)
        ax.set_title(text, **(title_kwargs or {}))

    return ax


# ---------------------------------------------------------------------------
# Plot 2: Grouped box-plots of error vs LEAD TIME
# ---------------------------------------------------------------------------
def plot_err_box_vs_lead(
    err_df: pd.DataFrame,
    *,
    variable: str = "Hs",
    part_idx: Optional[int] = None,
    metric: str = "abs_err",
    model_ids: Optional[Sequence[str]] = None,
    labels: Optional[Sequence[str]] = None,
    colors: Optional[Sequence[str]] = None,
    lead_hours: Optional[Sequence[int]] = None,
    xticks: Optional[Sequence[int]] = None,
    ylim: Optional[Tuple[float, float]] = None,
    ax: Optional[plt.Axes] = None,
    width_factor: float = 0.8,
    alpha: float = 0.5,
    add_legend: bool = True,
    auto_title: bool = False,
    title: Optional[str] = None,
    title_kwargs: Optional[Dict[str, Any]] = None,
) -> plt.Axes:
    """Grouped boxplots of error vs lead-time."""
    
    # --- Filter Data ---
    df = err_df[err_df["variable"] == variable].copy()
    if df.empty: return ax 

    # Strict partition logic from notebook
    if part_idx is None:
         if (df["f_idx"] == -1).any():
             df = df[df["f_idx"] == -1].copy()
             part_label = None
         else:
             raise ValueError(f"Variable '{variable}' requires `part_idx`.")
    else:
        df = df[df["f_idx"] == int(part_idx)].copy()
        if df.empty:
            raise ValueError(f"No rows for '{variable}' at partition index {part_idx}.")
        part_label = str(df["x_label"].dropna().iloc[0]) if "x_label" in df.columns else f"idx {part_idx}"

    metric_col = _metric_to_canonical(metric, df.columns)

    if "model_id" in df.columns:
        if model_ids is None: model_ids = list(df["model_id"].unique())
        else: model_ids = [m for m in model_ids if m in df["model_id"].unique()]
    else:
        model_ids = ["_single_"]
        df = df.assign(model_id=model_ids[0])

    # Lead Mapping
    if lead_hours is None:
        all_leads = sorted(df["lead_h"].astype(int).unique().tolist())
    else:
        all_leads = sorted([int(h) for h in lead_hours])

    if not all_leads: return ax
    lead_map = {h: i for i, h in enumerate(all_leads)}
    
    # Plotting
    apply_paper_style(large_text=True)
    if ax is None: _, ax = plt.subplots(figsize=(9.0, 4.4))
    
    # *** Zero Line for Raw Errors ***
    if metric_col == "err":
        ax.axhline(0, color='black', linestyle='--', linewidth=1.5, alpha=0.8, zorder=0)

    n_models = len(model_ids)
    labels = list(labels) if labels is not None else DEFAULT_LABELS[:n_models]
    colors = list(colors) if colors is not None else DEFAULT_COLORS[:n_models]

    spacing = 1.0
    group_w = spacing * width_factor
    box_w = group_w / n_models
    offsets = np.linspace(-group_w / 2 + box_w / 2, group_w / 2 - box_w / 2, n_models)
    legend_handles = []

    for m, (mid, lab, col) in enumerate(zip(model_ids, labels, colors)):
        d_m = df[df["model_id"] == mid]
        data, pos = [], []
        
        for h in all_leads:
            vals = d_m.loc[d_m["lead_h"] == h, metric_col].dropna().to_numpy(dtype=float)
            if vals.size:
                data.append(vals)
                pos.append(lead_map[h] + offsets[m])

        if data:
            bp = ax.boxplot(data, positions=pos, widths=box_w, patch_artist=True,
                            showfliers=False, boxprops=dict(facecolor=col, alpha=alpha))
            for item in ['boxes', 'whiskers', 'fliers', 'caps']:
                plt.setp(bp[item], color=col)
            plt.setp(bp["medians"], color=col) 

        legend_handles.append(PathPatch([], facecolor=col, edgecolor=col, alpha=alpha, label=lab))

    # Axis Formatting
    ax.set_xlabel("Lead time [h]")
    ax.set_ylabel(_ylabel(variable, metric_col))
    ax.set_xlim(-0.5, len(all_leads) - 0.5)

    if xticks is not None:
        tick_indices = [lead_map[h] for h in xticks if h in lead_map]
        tick_labels  = [str(h) for h in xticks if h in lead_map]
        ax.set_xticks(tick_indices)
        ax.set_xticklabels(tick_labels)
    else:
        preferred = [3, 24, 72, 120]
        preferred = [h for h in preferred if h in lead_map]
        if preferred:
            ax.set_xticks([lead_map[h] for h in preferred])
            ax.set_xticklabels([str(h) for h in preferred])
        else:
            ax.set_xticks(range(len(all_leads)))
            ax.set_xticklabels([str(h) for h in all_leads])

    if ylim: ax.set_ylim(*ylim)
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
    if add_legend: ax.legend(handles=legend_handles, frameon=False, loc="upper right")
    despine(ax)

    if title: ax.set_title(title, **(title_kwargs or {}))
    elif auto_title:
        m_name = {"err": "Error", "abs_err": "Absolute Error", "rel_err": "Relative Error"}.get(metric_col, metric_col)
        t_str = f"{m_name}: {variable}"
        if part_idx is not None and part_label: 
            t_str += f" [{part_label}]"
        ax.set_title(t_str)

    return ax


# ---------------------------------------------------------------------------
# High-Level Summary Wrapper
# ---------------------------------------------------------------------------
def create_summary_boxplots(
    err_df: pd.DataFrame,
    *,
    save_path: str,
    model_list: Optional[Sequence[str]] = None,
    labels: Optional[Sequence[str]] = None,
    colors: Optional[Sequence[str]] = None,
    freq_leads: Sequence[int] = (24, 120),
    scalar_leads: Sequence[int] = (3, 24, 72, 120),
    hs_partitions: Sequence[int] = (0, 1, 2),
    show_plots: bool = True
):
    """Generates and saves a complete suite of boxplots."""
    metrics_map = {"raw": "err", "absolute": "abs_err", "relative": "rel_err"}
    
    if model_list is None: 
        model_list = [m for m in ["ecmwf_raw", "ecmwf_ml"] if m in err_df["model_id"].unique()]
    
    if labels is None: 
        labels = ["ECMWF" if m == "ecmwf_raw" else "ML" for m in model_list]
    if colors is None: 
        colors = ["#e41a1c" if m == "ecmwf_raw" else "#377eb8" for m in model_list]
    
    partitioned_vars = ["Hs", "Tm02", "Tp"]

    for folder_name, metric_key in metrics_map.items():
        metric_dir = os.path.join(save_path, folder_name)
        os.makedirs(metric_dir, exist_ok=True)
        
        # A. Energy
        for lead in freq_leads:
            fig, ax = plt.subplots(figsize=(9.0, 4.4))
            try:
                plot_err_box_by_freq(
                    err_df, variable="E", metric=metric_key, lead_time=lead,
                    model_ids=model_list, labels=labels, colors=colors,
                    auto_title=True, ylim=None, ax=ax
                )
                fig.savefig(os.path.join(metric_dir, f"e_{lead}h.png"), bbox_inches='tight', dpi=300)
            except Exception:
                pass # Silently skip missing data to avoid crash
            finally:
                if not show_plots: plt.close(fig)

        # B. Bulk Vars
        for var in partitioned_vars:
            if var not in err_df["variable"].values: continue
            for p_idx in hs_partitions:
                # Existence Check to prevent Ghost Plots
                if not ((err_df["variable"] == var) & (err_df["f_idx"] == p_idx)).any(): continue
                
                fig, ax = plt.subplots(figsize=(9.0, 4.4))
                try:
                    plot_err_box_vs_lead(
                        err_df, variable=var, part_idx=p_idx, metric=metric_key,
                        lead_hours=None,          
                        xticks=scalar_leads,      
                        model_ids=model_list, labels=labels, colors=colors,
                        auto_title=True, ylim=None, ax=ax
                    )
                    fname = f"{var.lower()}_part{p_idx}.png"
                    fig.savefig(os.path.join(metric_dir, fname), bbox_inches='tight', dpi=300)
                except Exception:
                    pass
                finally:
                    if not show_plots: plt.close(fig)

    if show_plots: plt.show()
    print(f"✅ Boxplot summary updated (Zero line added). Plots saved to: {save_path}")
