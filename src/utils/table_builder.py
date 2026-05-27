"""
table_builder.py — Evaluation Table Construction.

This module provides the core logic for converting raw dataset outputs (tensors)
into structured Pandas DataFrames for analysis. It handles:
1. Physics calculations (Hs, Tm01, Tm02, Tp) on specific frequency partitions.
2. Iteration over dataset samples (Forecast vs Observation).
3. Merging forecast and observation tables to compute error metrics.
4. Filtering and Pivoting utilities for analysis.
"""

from dataclasses import dataclass
from typing import List, Tuple, Optional, Sequence, Literal, Dict, Any
import operator
import numpy as np
import pandas as pd
import torch
import os

# ---------------------------------------------------------------------------
# 1. Helper Utilities
# ---------------------------------------------------------------------------
def _lead_hours_from_len(T: int) -> List[int]:
    """Generates standard 3-hourly lead times based on sequence length."""
    return [3 * (i + 1) for i in range(T)]

def _format_band(fmin: float, fmax: float) -> str:
    """Formats a frequency band tuple into a string label."""
    return f"{float(fmin):.3g}–{float(fmax):.2g} Hz"

def _safe_tm01(e: torch.Tensor, f: torch.Tensor) -> float:
    """Computes Mean Period Tm01 safely (handling empty/NaNs)."""
    if e.numel() == 0: return float("nan")
    m0 = torch.trapz(e, f)
    if not torch.isfinite(m0) or m0 <= 1e-9: return float("nan")
    m1 = torch.trapz(f * e, f)
    if not torch.isfinite(m1) or m1 <= 1e-9: return float("nan")
    return float((m0 / m1).item())

def _safe_tm02(e: torch.Tensor, f: torch.Tensor) -> float:
    """Computes Zero-Crossing Period Tm02 safely."""
    if e.numel() == 0: return float("nan")
    m0 = torch.trapz(e, f)
    if not torch.isfinite(m0) or m0 <= 1e-9: return float("nan")
    m2 = torch.trapz((f**2) * e, f)
    if not torch.isfinite(m2) or m2 <= 1e-9: return float("nan")
    return float(torch.sqrt(m0 / m2).item())

def _safe_tp(e: torch.Tensor, f: torch.Tensor) -> float:
    """Computes Peak Period Tp safely."""
    if e.numel() == 0: return float("nan")
    peak_idx = torch.argmax(e)
    fpk = float(f[peak_idx].item())
    if fpk <= 1e-9: return float("nan")
    return 1.0 / fpk

def _coalesce_columns_after_merge(df: pd.DataFrame, names: Sequence[str]) -> pd.DataFrame:
    """Coalesce duplicated columns created by merge (e.g. x_label_x/y)."""
    for name in names:
        candidates = []
        if f"{name}_y" in df.columns: candidates.append(f"{name}_y")
        if f"{name}_x" in df.columns: candidates.append(f"{name}_x")
        if name in df.columns and name not in candidates: candidates.append(name)

        if not candidates: continue

        series = df[candidates[0]]
        for c in candidates[1:]:
            series = series.combine_first(df[c])
        
        df[name] = series
        for c in [f"{name}_x", f"{name}_y"]:
            if c in df.columns: df.drop(columns=c, inplace=True)
    return df


# ---------------------------------------------------------------------------
# 2. Iterators (Partition-Aware)
# ---------------------------------------------------------------------------

def _iter_ds_rows_E(ds, leads=None, want_obs=True, want_pred=True):
    """Iterates over raw Energy Density (E) at specific frequencies."""
    Freqs = torch.as_tensor(ds.selected_freqs, dtype=torch.float32)
    F = Freqs.numel()
    lead_hours_all = None
    
    for i in range(len(ds)):
        # --- FIX: Use dataset method for time to handle filtered indices correctly ---
        issuance_dt = ds.get_issuance_time(i)
        
        # --- FIX: Unpack 8 items instead of 6 ---
        _, Y, mask_Y, _, _, Y_model, _, _ = ds[i]
        
        Y, M, YM = Y.squeeze(), mask_Y.squeeze(), Y_model.squeeze()
        T = Y.shape[0]
        
        if lead_hours_all is None: lead_hours_all = _lead_hours_from_len(T)
        
        map_h2i = {h: idx for idx, h in enumerate(lead_hours_all)}
        sel_idx = [map_h2i[h] for h in leads if h in map_h2i] if leads else range(T)

        for t_idx in sel_idx:
            lead_h = lead_hours_all[t_idx]
            for f_idx in range(F):
                freq = float(Freqs[f_idx].item())
                y_true = float(Y[t_idx, f_idx].item()) if want_obs else None
                y_pred = float(YM[t_idx, f_idx].item()) if want_pred else None
                is_valid = bool(M[t_idx, f_idx].item()) if want_obs else True
                yield issuance_dt, lead_h, f_idx, freq, y_true, y_pred, is_valid

def _iter_ds_rows_Hs(ds, partitions, leads=None, want_obs=True, want_pred=True):
    """Iterates over Significant Wave Height (Hs) integrated over partitions."""
    Freqs = torch.as_tensor(ds.selected_freqs, dtype=torch.float32)
    part_masks = [(Freqs >= fmin) & (Freqs < fmax) for (fmin, fmax) in partitions]
    part_labels = [_format_band(fmin, fmax) for (fmin, fmax) in partitions]
    lead_hours_all = None

    for i in range(len(ds)):
        # --- FIX: Use dataset method for time to handle filtered indices correctly ---
        issuance_dt = ds.get_issuance_time(i)
        
        # --- FIX: Unpack 8 items instead of 6 ---
        _, Y, mask_Y, _, _, Y_model, _, _ = ds[i]
        
        Y, M, YM = Y.squeeze(), mask_Y.squeeze(), Y_model.squeeze()
        T = Y.shape[0]
        
        if lead_hours_all is None: lead_hours_all = _lead_hours_from_len(T)

        map_h2i = {h: idx for idx, h in enumerate(lead_hours_all)}
        sel_idx = [map_h2i[h] for h in leads if h in map_h2i] if leads else range(T)

        for t_idx in sel_idx:
            lead_h = lead_hours_all[t_idx]
            for p_idx, (pmask, plabel) in enumerate(zip(part_masks, part_labels)):
                # Obs
                if want_obs:
                    valid_bins = (M[t_idx] == 1) & pmask
                    if valid_bins.any():
                        m0 = torch.trapz(Y[t_idx][valid_bins], Freqs[valid_bins], dim=0).clamp_min(0.0)
                        hs_true = float((4.0 * torch.sqrt(m0)).item())
                        is_valid = True
                    else:
                        hs_true, is_valid = None, False
                else:
                    hs_true, is_valid = None, True

                # Model
                if want_pred:
                    e_m = YM[t_idx][pmask]
                    if e_m.numel() > 0:
                        m0m = torch.trapz(e_m, Freqs[pmask], dim=0).clamp_min(0.0)
                        hs_pred = float((4.0 * torch.sqrt(m0m)).item())
                    else:
                        hs_pred = None
                else:
                    hs_pred = None
                
                yield issuance_dt, lead_h, p_idx, plabel, hs_true, hs_pred, bool(is_valid)

def _iter_ds_rows_periods_partitioned(ds, partitions, leads=None, *, want_obs=True, want_pred=True):
    """
    Iterates over PARTITIONS to compute Tm01, Tm02, Tp for each band.
    Yields rows with x_kind="partition" and f_idx=p_idx.
    """
    Freqs = torch.as_tensor(ds.selected_freqs, dtype=torch.float32)
    part_masks = [(Freqs >= fmin) & (Freqs < fmax) for (fmin, fmax) in partitions]
    part_labels = [_format_band(fmin, fmax) for (fmin, fmax) in partitions]
    lead_hours_all = None

    for i in range(len(ds)):
        # --- FIX: Use dataset method for time to handle filtered indices correctly ---
        issuance_dt = ds.get_issuance_time(i)
        
        # --- FIX: Unpack 8 items instead of 6 ---
        _, Y, mask_Y, _, _, Y_model, _, _ = ds[i]
        
        Y, M, YM = Y.squeeze(), mask_Y.squeeze(), Y_model.squeeze()
        T = Y.shape[0]
        
        if lead_hours_all is None: lead_hours_all = _lead_hours_from_len(T)

        map_h2i = {h: idx for idx, h in enumerate(lead_hours_all)}
        sel_idx = [map_h2i[h] for h in leads if h in map_h2i] if leads else range(T)

        for t_idx in sel_idx:
            lead_h = lead_hours_all[t_idx]
            for p_idx, (pmask, plabel) in enumerate(zip(part_masks, part_labels)):
                
                # --- OBSERVATIONS ---
                if want_obs:
                    # Apply mask: Must match partition AND be valid data
                    valid_bins = (M[t_idx] == 1) & pmask
                    
                    if valid_bins.any():
                        e_obs = Y[t_idx][valid_bins]
                        f_obs = Freqs[valid_bins]
                        tm01_true = _safe_tm01(e_obs, f_obs)
                        tm02_true = _safe_tm02(e_obs, f_obs)
                        tp_true   = _safe_tp(e_obs, f_obs)
                        # Valid if at least one metric is finite
                        is_valid = np.isfinite([tm01_true, tm02_true, tp_true]).any()
                    else:
                        tm01_true = tm02_true = tp_true = None
                        is_valid = False
                else:
                    tm01_true = tm02_true = tp_true = None
                    is_valid = True

                # --- MODEL ---
                if want_pred:
                    # Model always has data on the grid, just mask by freq
                    e_mod = YM[t_idx][pmask]
                    f_mod = Freqs[pmask]
                    
                    if e_mod.numel() > 0:
                        tm01_pred = _safe_tm01(e_mod, f_mod)
                        tm02_pred = _safe_tm02(e_mod, f_mod)
                        tp_pred   = _safe_tp(e_mod, f_mod)
                    else:
                        tm01_pred = tm02_pred = tp_pred = None
                else:
                    tm01_pred = tm02_pred = tp_pred = None

                # Yield one row per variable for this partition
                yield issuance_dt, lead_h, "Tm01", p_idx, f"Tm01[{plabel}]", tm01_true, tm01_pred, bool(is_valid)
                yield issuance_dt, lead_h, "Tm02", p_idx, f"Tm02[{plabel}]", tm02_true, tm02_pred, bool(is_valid)
                yield issuance_dt, lead_h, "Tp",   p_idx, f"Tp[{plabel}]",   tp_true,   tp_pred,   bool(is_valid)


# ---------------------------------------------------------------------------
# 3. Public Builders
# ---------------------------------------------------------------------------

def build_observations_table(
    ds,
    variables: Sequence[str],
    partitions: Optional[Sequence[Tuple[float, float]]] = None,
    leads: Optional[Sequence[int]] = None,
    store_valid_mask: bool = True,
    include_valid_time: bool = True,
    cache_path: Optional[str] = None,
) -> pd.DataFrame:
    """Builds the table of Ground Truth observations."""
    if cache_path:
        try: return pd.read_parquet(cache_path)
        except: pass

    rows = []
    want_E = "E" in variables
    want_Hs = "Hs" in variables
    want_periods = any(v in variables for v in ["Tm01", "Tm02", "Tp"])

    # 1. Energy
    if want_E:
        for args in _iter_ds_rows_E(ds, leads, want_obs=True, want_pred=False):
            issuance_dt, lead_h, f_idx, freq, y_true, _, is_valid = args
            rows.append({
                "issuance_time": pd.Timestamp(issuance_dt),
                "lead_h": int(lead_h),
                "variable": "E", "variable_label": "E",
                "x_kind": "freq", "x_value": float(freq), "x_label": f"{float(freq):.5g} Hz",
                "freq": np.float32(freq), "f_idx": int(f_idx),
                "y_true": np.float32(y_true) if is_valid and y_true is not None else np.nan,
                "is_valid": is_valid
            })

    # 2. Hs (Partitioned)
    if want_Hs:
        assert partitions, "Partitions required for Hs."
        for args in _iter_ds_rows_Hs(ds, partitions, leads, want_obs=True, want_pred=False):
            issuance_dt, lead_h, p_idx, plabel, hs_true, _, is_valid = args
            rows.append({
                "issuance_time": pd.Timestamp(issuance_dt),
                "lead_h": int(lead_h),
                "variable": "Hs", "variable_label": f"Hs[{plabel}]",
                "x_kind": "partition", "x_value": int(p_idx), "x_label": plabel,
                "f_idx": int(p_idx),
                "y_true": np.float32(hs_true) if is_valid and hs_true is not None else np.nan,
                "is_valid": is_valid
            })

    # 3. Periods (Partitioned)
    if want_periods:
        assert partitions, "Partitions required for Tm/Tp variables."
        for args in _iter_ds_rows_periods_partitioned(ds, partitions, leads, want_obs=True, want_pred=False):
            issuance_dt, lead_h, var_name, p_idx, label, y_true, _, is_valid = args
            if var_name not in variables: continue
            
            rows.append({
                "issuance_time": pd.Timestamp(issuance_dt),
                "lead_h": int(lead_h),
                "variable": var_name, "variable_label": label,
                "x_kind": "partition", "x_value": int(p_idx), "x_label": label.split("[")[-1].strip("]"),
                "f_idx": int(p_idx),
                "y_true": np.float32(y_true) if is_valid and y_true is not None else np.nan,
                "is_valid": is_valid
            })

    obs_df = pd.DataFrame.from_records(rows)
    if include_valid_time and not obs_df.empty:
        obs_df["valid_time"] = obs_df["issuance_time"] + pd.to_timedelta(obs_df["lead_h"], unit="h")
    
    if cache_path:
        try: obs_df.to_parquet(cache_path, index=False)
        except: pass
    return obs_df


def build_forecast_table(
    ds,
    model_id: str,
    variables: Sequence[str],
    partitions: Optional[Sequence[Tuple[float, float]]] = None,
    leads: Optional[Sequence[int]] = None,
    include_valid_time: bool = True,
    cache_path: Optional[str] = None,
) -> pd.DataFrame:
    """Builds the table of Forecast predictions for a specific model."""
    if cache_path:
        try: return pd.read_parquet(cache_path)
        except: pass

    rows = []
    want_E = "E" in variables
    want_Hs = "Hs" in variables
    want_periods = any(v in variables for v in ["Tm01", "Tm02", "Tp"])

    if want_E:
        for args in _iter_ds_rows_E(ds, leads, want_obs=False, want_pred=True):
            issuance_dt, lead_h, f_idx, freq, _, y_pred, _ = args
            rows.append({
                "issuance_time": pd.Timestamp(issuance_dt),
                "lead_h": int(lead_h), "model_id": str(model_id),
                "variable": "E", "variable_label": "E",
                "x_kind": "freq", "x_value": float(freq), "x_label": f"{float(freq):.5g} Hz",
                "freq": np.float32(freq), "f_idx": int(f_idx),
                "y_pred": np.float32(y_pred) if y_pred is not None else np.nan
            })

    if want_Hs:
        assert partitions, "Partitions required for Hs."
        for args in _iter_ds_rows_Hs(ds, partitions, leads, want_obs=False, want_pred=True):
            issuance_dt, lead_h, p_idx, plabel, _, hs_pred, _ = args
            rows.append({
                "issuance_time": pd.Timestamp(issuance_dt),
                "lead_h": int(lead_h), "model_id": str(model_id),
                "variable": "Hs", "variable_label": f"Hs[{plabel}]",
                "x_kind": "partition", "x_value": int(p_idx), "x_label": plabel,
                "f_idx": int(p_idx),
                "y_pred": np.float32(hs_pred) if hs_pred is not None else np.nan
            })

    if want_periods:
        assert partitions, "Partitions required for Tm/Tp."
        for args in _iter_ds_rows_periods_partitioned(ds, partitions, leads, want_obs=False, want_pred=True):
            issuance_dt, lead_h, var_name, p_idx, label, _, y_pred, _ = args
            if var_name not in variables: continue
            
            rows.append({
                "issuance_time": pd.Timestamp(issuance_dt),
                "lead_h": int(lead_h), "model_id": str(model_id),
                "variable": var_name, "variable_label": label,
                "x_kind": "partition", "x_value": int(p_idx), "x_label": label.split("[")[-1].strip("]"),
                "f_idx": int(p_idx),
                "y_pred": np.float32(y_pred) if y_pred is not None else np.nan
            })

    fc_df = pd.DataFrame.from_records(rows)
    if include_valid_time and not fc_df.empty:
        fc_df["valid_time"] = fc_df["issuance_time"] + pd.to_timedelta(fc_df["lead_h"], unit="h")
    
    if cache_path:
        try: fc_df.to_parquet(cache_path, index=False)
        except: pass
    return fc_df


def make_errors_view(
    forecast_df: pd.DataFrame, 
    obs_df: pd.DataFrame, 
    metrics: Sequence[str] = ("raw", "abs", "rel"), 
    eps: float = 1e-6, 
    mask_policy: Literal["obs"] = "obs"
) -> pd.DataFrame:
    """Joins Forecast and Observation tables and computes Error Metrics."""
    keys = ["issuance_time", "lead_h", "variable", "x_kind", "f_idx"]
    
    # Slim down obs before join to avoid column explosion
    obs_slim = obs_df[[c for c in obs_df.columns if c in keys + ["y_true", "is_valid", "freq", "x_label", "variable_label"]]]
    
    merged = forecast_df.merge(obs_slim, on=keys, how="left", suffixes=("", "_y"))
    
    # Coalesce metadata columns if duplicated (Option B)
    merged = _coalesce_columns_after_merge(merged, ["variable_label", "x_label", "freq"])
            
    if "raw" in metrics: merged["err"] = merged["y_pred"] - merged["y_true"]
    if "abs" in metrics: merged["abs_err"] = (merged["y_pred"] - merged["y_true"]).abs()
    if "rel" in metrics: merged["rel_err"] = (merged["y_pred"] - merged["y_true"]).abs() / merged["y_true"].abs().clip(lower=eps)
    
    if mask_policy == "obs" and "is_valid" in merged.columns:
        invalid = ~merged["is_valid"]
        # Set errors to NaN where observations are invalid
        cols_to_mask = ["y_true"] + [m for m in ["err", "abs_err", "rel_err"] if m in merged.columns]
        merged.loc[invalid, cols_to_mask] = np.nan
        
    return merged


def pivot_issuance_wide(
    df: pd.DataFrame,
    value: str,
    *,
    index: Sequence[str] = ("lead_h", "variable", "x_label"),
    columns: str = "issuance_time",
    agg: str = "first",
) -> pd.DataFrame:
    """
    Create a wide matrix with columns = issuance_time for plotting.
    """
    df_tmp = df.copy()
    if np.issubdtype(df_tmp[columns].dtype, np.datetime64):
        df_tmp["_col_iso"] = df_tmp[columns].dt.strftime("%Y-%m-%d %H:%M")
        col_key = "_col_iso"
    else:
        col_key = columns

    wide = (
        df_tmp
        .pivot_table(values=value, index=list(index), columns=col_key, aggfunc=agg)
        .sort_index(axis=1)
    )
    wide.columns.name = None
    return wide


# ---------------------------------------------------------------------------
# 4. Broadcast Filters (for masking tables)
# ---------------------------------------------------------------------------
@dataclass
class DFCondition:
    metric: Literal["obs", "model", "err", "abs_err", "rel_err"]
    op: Literal[">", ">=", "<", "<=", "==", "!="]
    thresh: float

_OP_MAP = {
    ">":  operator.gt, ">=": operator.ge,
    "<":  operator.lt, "<=": operator.le,
    "==": operator.eq, "!=": operator.ne,
}
_METRIC_ALIAS = {
    "raw": "err", "abs": "abs_err", "rel": "rel_err",
    "obs": "y_true", "model": "y_pred",
}

def filter_err_df_broadcast(
    err_df: pd.DataFrame,
    *,
    mask_variable: Literal["Hs", "Tm01", "Tm02", "Tp"],
    mask_part_idx: Optional[int] = None,
    model_ids: Optional[Sequence[str]] = None,
    conditions: List[DFCondition],
    cond_combine: Literal["all", "any"] = "all",
    keep_mode: Literal["lead", "forecast"] = "lead",
    lead_combine: Literal["any", "all"] = "any",
    propagate: Literal["all", "same_variable", "scope_only"] = "all",
    default_keep_for_missing_keys: bool = False,
) -> pd.DataFrame:
    """
    Builds a boolean mask on a specific scope (e.g. Hs partition 0) based on conditions,
    and broadcasts this mask to the entire dataframe.
    """
    if mask_variable not in {"Hs", "Tm01", "Tm02", "Tp"}:
        raise ValueError("mask_variable must be one of: 'Hs','Tm01','Tm02','Tp'.")

    df_all = err_df.copy()
    if model_ids is not None:
        df_all = df_all[df_all["model_id"].isin(model_ids)]
        if df_all.empty: raise ValueError("Requested model_ids not present.")

    # 1. Build Scope
    scope = df_all[df_all["variable"] == mask_variable].copy()
    if mask_variable == "Hs":
        if mask_part_idx is None: raise ValueError("mask_part_idx required for Hs.")
        scope = scope[(scope.get("x_kind", "") == "partition") & (scope["f_idx"] == int(mask_part_idx))]
    else:
        # For periods, we now use partitions, so f_idx is the partition index
        if mask_part_idx is not None:
            scope = scope[scope["f_idx"] == int(mask_part_idx)]
        # If mask_part_idx is None, we assume the user wants ALL partitions or scalar? 
        # Safer to require it if the variable is partitioned.

    if scope.empty: raise ValueError("No rows found in scope.")

    # 2. Evaluate Conditions
    row_masks = []
    for c in conditions:
        col = _METRIC_ALIAS.get(c.metric, c.metric)
        if col not in scope.columns: raise ValueError(f"Metric '{c.metric}' not found.")
        row_masks.append(_OP_MAP[c.op](scope[col].astype(float), float(c.thresh)))

    if not row_masks: scope_mask = pd.Series(True, index=scope.index)
    else:
        scope_mask = row_masks[0]
        for m in row_masks[1:]:
            scope_mask = (scope_mask & m) if cond_combine == "all" else (scope_mask | m)

    # 3. Broadcast
    out = df_all.copy()
    
    if keep_mode == "forecast":
        fkeys = ["issuance_time", "model_id"]
        tmp = scope.assign(__keep=scope_mask.values)
        agg_func = "all" if lead_combine == "all" else "any"
        keep_forecasts = set(tmp.groupby(fkeys)["__keep"].agg(agg_func)[lambda x: x].index.tolist())
        
        out_fkeys = list(zip(out["issuance_time"], out["model_id"]))
        out = out.loc[pd.Series(out_fkeys).isin(keep_forecasts)].copy()
        return out

    # keep_mode == 'lead'
    key_cols = ["issuance_time", "lead_h", "model_id"]
    per_lead_keep = pd.Series(scope_mask.values, index=scope[key_cols].apply(tuple, axis=1))
    
    def lookup_keep(row):
        return per_lead_keep.get((row["issuance_time"], row["lead_h"], row["model_id"]), default_keep_for_missing_keys)

    keep_flags = out[key_cols].apply(lookup_keep, axis=1)

    if propagate == "scope_only":
        to_nan = (~keep_flags) & (out.index.isin(scope.index)) # Approx logic, scope index match is safer
    elif propagate == "same_variable":
        to_nan = (~keep_flags) & (out["variable"] == mask_variable)
    else:
        to_nan = (~keep_flags)

    cols = [c for c in ["y_true", "y_pred", "err", "abs_err", "rel_err"] if c in out.columns]
    out.loc[to_nan, cols] = np.nan
    return out



# ==============================================================================
# 5. Reporting / LaTeX Generation
# ==============================================================================
def generate_latex_rmse_table(err_df: pd.DataFrame, output_dir: str):
    """
    Generates a LaTeX table (and text file) reporting RMSE for Total, Swell, 
    and Wind Sea components across specific lead times.
    
    Structure:
        - Filters for Hs, Tp, Tm02.
        - Maps partition indices 0->Total, 1->Swell, 2->WindSea.
        - computes RMSE = sqrt(mean(err^2)).
        - Saves 'metrics_table.tex' and 'metrics_table.txt'.
    """
    print(f"Generating RMSE Table (LaTeX/Text) in {output_dir}...")
    
    # 1. Configuration
    target_vars = ["Hs", "Tp", "Tm02"]
    leads_of_interest = [3, 24, 60, 120]
    lead_map_label = {3: "t+3 hours", 24: "t+1.0 days", 60: "t+2.5 days", 120: "t+5.0 days"}
    
    # Partition Mapping (Hardcoded based on train.py configuration)
    part_map = {0: "Total", 1: "Swell", 2: "WindSea"}
    
    model_labels = {"ecmwf_raw": "ECMWF", "ecmwf_ml": "SpecX"} # Customize labels here
    
    # 2. Filter & Prepare Data
    # Ensure we have the 'err' column
    if "err" not in err_df.columns:
        if "y_pred" in err_df.columns and "y_true" in err_df.columns:
            err_df["err"] = err_df["y_pred"] - err_df["y_true"]
        else:
            print("Error: Cannot calculate RMSE. Missing 'err', 'y_pred', or 'y_true'.")
            return

    # Filter for Partitioned variables (Hs, Tp, Tm02) and relevant indices
    mask = (
        (err_df["variable"].isin(target_vars)) & 
        (err_df["x_kind"] == "partition") &
        (err_df["f_idx"].isin(part_map.keys())) &
        (err_df["lead_h"].isin(leads_of_interest))
    )
    df = err_df[mask].copy()
    
    if df.empty:
        print("Warning: No matching data found for RMSE table generation.")
        return

    # Map indices to names
    df["part_name"] = df["f_idx"].map(part_map)

    # 3. Compute RMSE
    # Group by: Lead, Model, Partition, Variable
    rmse_df = (
        df.groupby(["lead_h", "model_id", "part_name", "variable"])["err"]
        .apply(lambda x: np.sqrt((x**2).mean()))
        .reset_index(name="RMSE")
    )

    # 4. Pivot for Lookup
    # Key: "Part_Var" (e.g., "Total_Hs")
    rmse_df["col_key"] = rmse_df["part_name"] + "_" + rmse_df["variable"]
    
    pivot = rmse_df.pivot(
        index=["lead_h", "model_id"], 
        columns="col_key", 
        values="RMSE"
    )

    # 5. Construct Table Content
    # Define Column Order
    col_order = [
        "Total_Hs", "Total_Tp", "Total_Tm02",
        "Swell_Hs", "Swell_Tp", "Swell_Tm02",
        "WindSea_Hs", "WindSea_Tp", "WindSea_Tm02"
    ]
    
    latex_rows = []
    txt_rows = []
    
    # Header for TXT
    txt_rows.append(f"{'LEAD':<15} | {'MODEL':<10} | " + " | ".join([f"{c:<10}" for c in col_order]))
    txt_rows.append("-" * 150)

    for lead_h in sorted(leads_of_interest):
        lead_name = lead_map_label.get(lead_h, f"t+{lead_h}h")
        
        # LaTeX Header for Lead
        latex_rows.append(f"\\multicolumn{{10}}{{c}}{{\\textbf{{LEAD: {lead_name}}}}} \\\\ \\midrule")
        
        # Models
        for m_id in ["ecmwf_raw", "ecmwf_ml"]:
            if (lead_h, m_id) not in pivot.index:
                continue
            
            row_data = pivot.loc[(lead_h, m_id)]
            name_tex = model_labels.get(m_id, m_id)
            
            # Extract values
            vals = []
            vals_txt = []
            for c in col_order:
                val = row_data.get(c, np.nan)
                if pd.notna(val):
                    vals.append(f"{val:.2f}")
                    vals_txt.append(f"{val:.2f}")
                else:
                    vals.append("-")
                    vals_txt.append("-")
            
            # LaTeX Row
            # Bold model name for LaTeX
            latex_row = f"\\textbf{{{name_tex}}} & " + " & ".join(vals) + " \\\\"
            latex_rows.append(latex_row)
            
            # TXT Row
            txt_row = f"{lead_name:<15} | {m_id:<10} | " + " | ".join([f"{v:<10}" for v in vals_txt])
            txt_rows.append(txt_row)
        
        latex_rows.append("\\midrule")
        txt_rows.append("-" * 150)

    # 6. Final Assembly
    # LaTeX Wrapper
    latex_content = r"""
\begin{table*}[t]
\centering
\caption{RMSE comparison for Total, Swell, and Wind Sea components.}
\label{table:wave_metrics_rmse_all}
% 1 left-aligned column (Model) + 9 centered columns (Metrics)
\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}lccccccccc@{}}
\toprule
 & \multicolumn{3}{@{}c@{}}{\textbf{Total}} & \multicolumn{3}{@{}c@{}}{\textbf{Swell}} & \multicolumn{3}{@{}c@{}}{\textbf{Wind Sea}} \\ \cmidrule{2-4}\cmidrule{5-7}\cmidrule{8-10}
\textbf{Model} & \textbf{Hs} & \textbf{Tp} & \textbf{Tm02} & \textbf{Hs} & \textbf{Tp} & \textbf{Tm02} & \textbf{Hs} & \textbf{Tp} & \textbf{Tm02} \\ \midrule
""" + "\n".join(latex_rows) + r"""
\bottomrule
\end{tabular*}
\end{table*}
"""

    # 7. Save Files
    os.makedirs(output_dir, exist_ok=True)
    
    tex_path = os.path.join(output_dir, "metrics_table.tex")
    with open(tex_path, "w") as f:
        f.write(latex_content)
        
    txt_path = os.path.join(output_dir, "metrics_table.txt")
    with open(txt_path, "w") as f:
        f.write("\n".join(txt_rows))
        
    print(f"✅ Saved tables to:\n   -> {tex_path}\n   -> {txt_path}")