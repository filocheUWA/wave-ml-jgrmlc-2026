#!/usr/bin/env python3
"""
src/scripts/train_ensemble.py

Standalone ensemble training script for the SpecX wave forecasting model.
Optimized for Setonix HPC with robust checkpointing and advanced ensemble diagnostics.
"""

# ==============================================================================
# SECTION A: BOOTSTRAP & ENVIRONMENT SETUP
# ==============================================================================
import os
import sys

# 1. Fix Matplotlib Disk Quota Issue
MYSCRATCH = os.environ.get('MYSCRATCH', '/scratch/pawsey0106/afiloche')
mpl_cache_dir = os.path.join(MYSCRATCH, '.cache', 'matplotlib')
os.makedirs(mpl_cache_dir, exist_ok=True)
os.environ['MPLCONFIGDIR'] = mpl_cache_dir
print(f"🔧 Environment: Redirected Matplotlib cache to {mpl_cache_dir}")

# 2. Project Path Resolution
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../../"))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# ==============================================================================
# SECTION B: IMPORTS
# ==============================================================================
import random
import hashlib
import json
import csv
import re
import warnings
import argparse
from datetime import datetime
import time

import numpy as np
import torch
import pandas as pd
from torch.utils.data import DataLoader
from torch.optim import AdamW
import matplotlib
import matplotlib.pyplot as plt

# Force Agg backend for headless server execution
matplotlib.use('Agg')

# Local Project Imports
from src.architecture.model_specx import SpecX
from src.training import *
from src.training.scalers import IdentityScaler, LogZScoreScaler, BinwiseZScoreScaler
from src.training.data_management import predict_dataset, MLCorrectedDataset, collate
from src.utils import *

# Clean Plotting Imports (Using modular architecture)
from src.plotting.dashboard import (
    plot_dashboard_hs_components, 
    plot_dashboard_bulk_params, 
    generate_forecast_video,
    generate_extreme_error_dashboards
)
from src.plotting.boxplots import create_summary_boxplots
from src.plotting.heatmap import plot_heatmap_bias_std
from src.plotting.qq import plot_qq_empirical
from src.plotting.mse_decomposition import plot_mse_decomposition
from src.plotting.pca_analysis import plot_pca_structure, plot_pca_mode_swell, get_worst_phase_error_dates
from src.plotting.reliability import plot_stratified_ranks, plot_spectral_uncertainty_diagnostics

warnings.filterwarnings("ignore", category=FutureWarning)

# ==============================================================================
# SECTION C: GLOBAL CONFIGURATION & PATHS
# ==============================================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Data Paths ---
ENV_DATA_ROOT = os.environ.get('DATA_ROOT')
DATA_ROOT = ENV_DATA_ROOT if ENV_DATA_ROOT else "/scratch/pawsey0106/afiloche/data/processed"

PATHS = {
    'forecast_dir': os.path.join(DATA_ROOT, "ecmwf_forecast_nc"),
    'buoy_dir':     os.path.join(DATA_ROOT, "buoy_nc"),
    'tides_path':   os.path.join(DATA_ROOT, "Preludes_tides", "processed_tides.nc")
}

# --- Output Paths ---
ENV_OUTPUT_ROOT = os.environ.get('OUTPUT_ROOT')
OUTPUT_ROOT = ENV_OUTPUT_ROOT if ENV_OUTPUT_ROOT else "/scratch/pawsey0106/afiloche/output"
ENSEMBLE_OUTPUT_ROOT = os.path.join(OUTPUT_ROOT, "SpecX_Ensemble")

# --- Fixed Experiment Parameters ---
TRAIN_DATE, VAL_DATE, TEST_DATE = ['20200101_00', '20220622_00'], ['20220701_00', '20221222_00'], ['20230101_00', '20231231_12']
PARTITIONS = [(0.034, 0.51), (0.034, 0.10), (0.10, 0.51)]
PART_MAP = {1: "Swell (< 0.1Hz)", 2: "Wind Sea (> 0.1Hz)"}

FIXED_PARAMS = {
    "history_vars": ['E'], 
    "num_workers": 8, 
    "weight_decay": 1e-2, 
    "grad_clip_norm": 1.0, 
    "loss_scale": 5.0, 
    "n_spec_ch": 1, 
    "n_coeff_ch": 1, 
    "n_output_ch": 1, 
    "warmup_epochs": 2, 
    "min_lr_factor": 0.1,
}

# ==============================================================================
# SECTION D: HELPER FUNCTIONS
# ==============================================================================
def parse_args():
    parser = argparse.ArgumentParser(description="Train SpecX Ensemble Architecture")
    # Ensemble Specs
    parser.add_argument("--n_ensemble", type=int, default=20, help="Number of ensemble members")
    parser.add_argument("--base_seed", type=int, default=1234, help="Starting seed for members")
    
    # Model Specs
    parser.add_argument("--model_name", type=str, default='specx_ens')
    parser.add_argument("--model_size", type=str, default='base', choices=['small', 'base', 'large'])
    parser.add_argument("--use_xattn", type=bool, default=False)
    
    # Input Config
    parser.add_argument("--input_mode", type=str, default="spectra", choices=['spectra', 'coeffs', 'fused'])
    parser.add_argument("--scale_input", action="store_true", default=True)
    parser.add_argument("--use_history", type=bool, default=False)
    parser.add_argument("--use_date", type=bool, default=True)
    parser.add_argument("--use_tide", type=bool, default=True)
    
    # Hyperparams
    parser.add_argument("--target_mode", type=str, default="direct", choices=['direct', 'residual'])
    parser.add_argument("--scale_output", action="store_true", default=True)
    parser.add_argument("--loss_type", type=str, default="mae")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=16)
    
    # Output Override
    parser.add_argument("--output_root", type=str, default=ENSEMBLE_OUTPUT_ROOT)
    parser.add_argument("--skip_plots", action="store_true", help="If set, skips inference and plotting.")
    parser.add_argument("--force_retrain", action="store_true", help="Force retrain existing ensemble members.")
    
    return parser.parse_args()

def setup_experiment(args, proposed_config):
    config_str = json.dumps(proposed_config, sort_keys=True)
    config_hash = hashlib.md5(config_str.encode("utf-8")).hexdigest()[:6]
    run_id = f"{args.model_name}_{config_hash}"
    print(f"🆔 Generated Run ID: {run_id}")

    root = args.output_root
    os.makedirs(root, exist_ok=True)
    existing_dirs = sorted([d for d in os.listdir(root) if re.match(r"Exp_\d+", d)])
    
    found_existing = False
    exp_id, base_dir = "", ""

    for d in existing_dirs:
        try:
            cfg_path = os.path.join(root, d, "config.json")
            if not os.path.exists(cfg_path): continue
            with open(cfg_path, "r") as f:
                saved_config = json.load(f)
            if saved_config.get("run_id_str") == run_id:
                print(f"🔍 Found matching ensemble config in: {d}")
                exp_id, base_dir = d, os.path.join(root, d)
                found_existing = True
                break
        except: continue

    if not found_existing:
        indices = [int(d.split("_")[1]) for d in existing_dirs]
        next_id = max(indices) + 1 if indices else 1
        exp_id = f"Exp_{next_id:03d}"
        base_dir = os.path.join(root, exp_id)
        print(f"🆕 Creating NEW ensemble experiment: {exp_id}")

    dirs = {
        "weights": os.path.join(base_dir, "weights"),
        "figures": os.path.join(base_dir, "figures"),
        "data":    os.path.join(base_dir, "loss_data"),
        "checkpoints": os.path.join(base_dir, "loss_data", "inference_checkpoints_csv")
    }
    for p in dirs.values(): os.makedirs(p, exist_ok=True)

    weights_pattern = os.path.join(dirs["weights"], f"{run_id}_m{{}}.pt")
    history_path = os.path.join(dirs["data"], "ensemble_history.json")

    full_config = proposed_config.copy()
    full_config.update({
        "run_id_str": run_id,
        "weights_pattern": weights_pattern,
        "history_path": history_path
    })

    if not found_existing:
        with open(os.path.join(base_dir, "config.json"), "w") as f:
            json.dump(full_config, f, indent=4)
        
        log_path = os.path.join(root, "experiment_log.csv")
        headers = ["Exp_ID", "Timestamp", "Run_ID"] + list(proposed_config.keys())
        row = [exp_id, datetime.now().strftime("%Y-%m-%d %H:%M"), run_id] + list(proposed_config.values())
        
        is_new = not os.path.exists(log_path)
        with open(log_path, "a", newline="") as f:
            writer = csv.writer(f)
            if is_new: writer.writerow(headers)
            writer.writerow(row)
        print(f"📝 Experiment registered in log.")
    else:
        print(f"✅ Resuming experiment {exp_id}")

    return full_config, dirs

def print_step_header(step_num, total_steps, title):
    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Step {step_num}/{total_steps}: {title}...")

def generate_ensemble_latex_rmse(pivot_table, save_path):
    """Helper to dump the Pandas Pivot Table into a publication-ready LaTeX format."""
    LEAD_MAP = {3: "t+3 hours", 24: "t+1.0 days", 60: "t+2.5 days", 120: "t+5.0 days"}
    model_labels = {"ecmwf_raw": "ECMWF", "ecmwf_ml": "ML-Ensemble"}
    col_order = ["Total_Hs", "Total_Tp", "Total_Tm02", "Swell_Hs", "Swell_Tp", "Swell_Tm02", "WindSea_Hs", "WindSea_Tp", "WindSea_Tm02"]

    latex_rows = []
    for lead_h in sorted(LEAD_MAP.keys()):
        if lead_h not in pivot_table.index.get_level_values('lead_h'): continue
        lead_name = LEAD_MAP[lead_h]
        latex_rows.append(f"\\multicolumn{{10}}{{c}}{{\\textbf{{LEAD: {lead_name}}}}} \\\\ \\midrule")
        for m_id in ["ecmwf_raw", "ecmwf_ml"]:
            if (lead_h, m_id) not in pivot_table.index: continue
            row = pivot_table.loc[(lead_h, m_id)]
            vals_str = [f"{row.get(c, np.nan):.2f}" if pd.notna(row.get(c, np.nan)) else "-" for c in col_order]
            latex_rows.append(f"\\textbf{{{model_labels.get(m_id, m_id)}}} & " + " & ".join(vals_str) + " \\\\")
        latex_rows.append("\\midrule")

    latex_table = r"""
\begin{table*}[t]
\centering
\caption{RMSE comparison for Total, Swell, and Wind Sea components.}
\label{table:wave_metrics_rmse_all}
\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}lccccccccc@{}}
\toprule
 & \multicolumn{3}{@{}c@{}}{\textbf{Total}} & \multicolumn{3}{@{}c@{}}{\textbf{Swell}} & \multicolumn{3}{@{}c@{}}{\textbf{Wind Sea}} \\ \cmidrule{2-4}\cmidrule{5-7}\cmidrule{8-10}
\textbf{Model} & \textbf{Hs} & \textbf{Tp} & \textbf{Tm02} & \textbf{Hs} & \textbf{Tp} & \textbf{Tm02} & \textbf{Hs} & \textbf{Tp} & \textbf{Tm02} \\ \midrule
""" + "\n".join(latex_rows) + r"""
\bottomrule
\end{tabular*}
\end{table*}
"""
    with open(save_path, "w") as f: f.write(latex_table)

# ==============================================================================
# SECTION E: MAIN EXECUTION
# ==============================================================================
def main():
    args = parse_args()
    
    # Global Seeding is omitted because each member gets BASE_SEED + idx
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    DO_PLOTS = not args.skip_plots

    print(f"⚙️  Using device: {DEVICE}")
    print(f"📂 Output Root: {args.output_root}")
    print(f"📈 Ensemble Members: {args.n_ensemble}")

    # 1. Setup Experiment
    arg_vars = vars(args).copy()
    for k in ['skip_plots', 'force_retrain']: arg_vars.pop(k, None)
    
    proposed_config = {**arg_vars, **FIXED_PARAMS}
    config, dirs = setup_experiment(args, proposed_config)

    # 2. Load Data
    print("\n--- Loading Datasets ---")
    ds_args = {
        'var_list': ['E'], 
        'history_vars': FIXED_PARAMS['history_vars'] if args.use_history else [], 
        'frequency_band': [0.034, 0.51], 
        **PATHS
    }
    train_set = ECMWFDataset(start_end=TRAIN_DATE, **ds_args)
    val_set   = ECMWFDataset(start_end=VAL_DATE, **ds_args)
    test_set  = ECMWFDataset(start_end=TEST_DATE, **ds_args)

    # 3. Scalers & DataLoaders
    input_scaler = LogZScoreScaler() if args.scale_input else IdentityScaler()
    if args.scale_input: input_scaler.fit(train_set, fit_on='X')
    
    output_scaler = IdentityScaler()
    if args.scale_output:
        scale_key = 'Y' if args.target_mode == 'direct' else 'Y-YM'
        output_scaler = LogZScoreScaler() if args.target_mode == 'direct' else BinwiseZScoreScaler()
        output_scaler.fit(train_set, fit_on=scale_key)

    loader_args = {"batch_size": args.batch_size, "num_workers": config['num_workers'], "pin_memory": True, "collate_fn": collate}
    loaders = {
        "train": DataLoader(train_set, shuffle=True, **loader_args),
        "val":   DataLoader(val_set,   shuffle=False, **loader_args),
        "test":  DataLoader(test_set,  shuffle=False, **loader_args)
    }
    
    freqs = torch.as_tensor(train_set.selected_freqs, dtype=torch.float32)

    def instantiate_member(seed):
        torch.manual_seed(seed)
        if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
        model = SpecX(
            input_size=(40, len(freqs), 36), model_size=config["model_size"], input_mode=config["input_mode"],
            n_spec_ch=config["n_spec_ch"], n_coeff_ch=config["n_coeff_ch"], n_output_ch=config["n_output_ch"],
            use_date_default=config["use_date"], use_tide_default=config["use_tide"], 
            use_xattn_default=config["use_xattn"], use_buoy_history_default=args.use_history,
            n_hist_vars=len(ds_args['history_vars'])
        ).to(DEVICE)
        criterion = make_criterion(config["loss_type"], freqs).to(DEVICE)
        optimizer = AdamW(build_param_groups_for_adamw(model, config["weight_decay"]), lr=config["lr"])
        return model, optimizer, criterion

    # ==========================================================================
    # 4. ENSEMBLE TRAINING LOOP
    # ==========================================================================
    ensemble_histories = []
    if os.path.exists(config["history_path"]):
        try:
            with open(config["history_path"], "r") as f: ensemble_histories = json.load(f)
        except: pass

    print_step_header(1, 4, f"Training Ensemble ({args.n_ensemble} Members)")
    
    for member_idx in range(args.n_ensemble):
        save_path = config["weights_pattern"].format(member_idx)
        current_seed = args.base_seed + member_idx
        
        weights_exist = os.path.exists(save_path)
        existing_entry = next((h for h in ensemble_histories if h.get("member_idx") == member_idx), None)
        history_is_valid = (existing_entry is not None) and ("train_loss" in existing_entry) and (len(existing_entry["train_loss"]) > 0)

        if weights_exist and history_is_valid and not args.force_retrain:
            print(f"   🤖 Mem {member_idx+1}/{args.n_ensemble} [Seed: {current_seed}] -> Cached weights/history found. Skipping.")
            continue
            
        print(f"\n   🤖 Mem {member_idx+1}/{args.n_ensemble} [Seed: {current_seed}] -> Training...")
        model, optimizer, criterion = instantiate_member(current_seed)
        runner = EpochRunner(model, optimizer, criterion, input_scaler=input_scaler, output_scaler=output_scaler,
                            target_mode=config["target_mode"], loss_scale_residual=config["loss_scale"],
                            grad_clip_norm=config["grad_clip_norm"], use_date=config["use_date"],
                            use_tide=config["use_tide"], use_history=args.use_history)
        
        scheduler = build_warmup_cosine_scheduler(optimizer, warmup_epochs=config["warmup_epochs"], total_epochs=config["epochs"], min_lr_factor=config["min_lr_factor"])
        
        history = {
            "member_idx": member_idx, "seed": current_seed,
            "train_loss": [], "val_loss": [], "test_loss": [], "lr": [],
            "train_hs_rmse": [], "val_hs_rmse": [], "test_hs_rmse": []
        }
        best_val_rmse = float('inf')

        # Epoch 0
        for split, loader in zip(["train", "val", "test"], [loaders["train"], loaders["val"], loaders["test"]]):
             rmse = compute_partition_Hs_rmse(model, loader, freqs=freqs, partition=PARTITIONS[0], input_scaler=input_scaler, output_scaler=output_scaler, target_mode=config["target_mode"], use_date=config["use_date"], use_tide=config["use_tide"])
             history[f"{split}_hs_rmse"].append(float(rmse))
             
        # Main Epoch Loop
        for epoch in range(1, config["epochs"] + 1):
            train_stats = runner.run_epoch(loaders["train"], mode="train")
            val_stats   = runner.run_epoch(loaders["val"],   mode="eval")
            test_stats  = runner.run_epoch(loaders["test"],  mode="eval")

            for split, loader in zip(["train", "val", "test"], [loaders["train"], loaders["val"], loaders["test"]]):
                 rmse = compute_partition_Hs_rmse(model, loader, freqs=freqs, partition=PARTITIONS[0], input_scaler=input_scaler, output_scaler=output_scaler, target_mode=config["target_mode"], use_date=config["use_date"], use_tide=config["use_tide"])
                 history[f"{split}_hs_rmse"].append(float(rmse))

            scheduler.step()
            history["lr"].append(optimizer.param_groups[0]["lr"])
            history["train_loss"].append(train_stats["loss_mean"])
            history["val_loss"].append(val_stats["loss_mean"])
            history["test_loss"].append(test_stats["loss_mean"])

            marker = ""
            if history['val_hs_rmse'][-1] < best_val_rmse:
                best_val_rmse = history['val_hs_rmse'][-1]
                torch.save(model.state_dict(), save_path)
                marker = "<- BEST"
            print(f"      [Ep {epoch:03d}] Loss: {train_stats['loss_mean']:.4f}/{val_stats['loss_mean']:.4f} | RMSE_Hs: {history['val_hs_rmse'][-1]:.3f}m {marker}")

        # Update ensemble history file
        ensemble_histories = [h for h in ensemble_histories if h.get("member_idx") != member_idx]
        ensemble_histories.append(history)
        with open(config["history_path"], "w") as f: json.dump(ensemble_histories, f, indent=4)

    # ==========================================================================
    # 5. ENSEMBLE INFERENCE LOOP
    # ==========================================================================
    if DO_PLOTS:
        print_step_header(2, 4, "Ensemble Inference (Checkpointing CSVs)")
        eval_cache_path = os.path.join(dirs["data"], "ensemble_test_err_df.csv")
        variables = ["E", "Hs", "Tm01", "Tm02", "Tp"]
        group_keys = ["issuance_time", "lead_h", "variable", "variable_label", "x_kind", "x_value", "x_label", "f_idx"]
        
        obs_df_test = build_observations_table(test_set, variables=variables, partitions=PARTITIONS)
        fc_df_raw   = build_forecast_table(test_set, model_id="ecmwf_raw", variables=variables, partitions=PARTITIONS)
        
        member_dfs = []
        dummy_model, _, _ = instantiate_member(0) # Ensure memory struct
        
        for member_idx in range(args.n_ensemble):
            ckpt_path = os.path.join(dirs["checkpoints"], f"member_{member_idx:02d}.csv")
            if os.path.exists(ckpt_path) and not args.force_retrain:
                df_member = pd.read_csv(ckpt_path)
                if 'issuance_time' in df_member.columns: df_member['issuance_time'] = pd.to_datetime(df_member['issuance_time'])
                member_dfs.append(df_member)
                continue
            
            path = config["weights_pattern"].format(member_idx)
            if not os.path.exists(path): continue
            
            dummy_model.load_state_dict(torch.load(path, map_location=DEVICE, weights_only=True))
            dummy_model.eval()
            
            y_pred, _ = predict_dataset(loaders["test"], dummy_model, input_scaler=input_scaler, output_scaler=output_scaler, target_mode=config["target_mode"], use_date=config["use_date"], use_tide=config["use_tide"], use_history=args.use_history)
            
            ds_member = MLCorrectedDataset(test_set, y_pred)
            df_member = build_forecast_table(ds_member, model_id=f"mem_{member_idx}", variables=variables, partitions=PARTITIONS)
            df_final = df_member[[k for k in group_keys if k in df_member.columns] + ["y_pred"]]
            df_final.to_csv(ckpt_path, index=False)
            member_dfs.append(df_final)

        # Aggregate Bulk Stats
        all_members_df = pd.concat(member_dfs, axis=0)
        stats_df = all_members_df.groupby(group_keys)["y_pred"].agg(["mean", "std"]).reset_index()
        fc_df_ml = stats_df.copy()
        fc_df_ml["model_id"] = "ecmwf_ml" 
        fc_df_ml = fc_df_ml.rename(columns={"mean": "y_pred", "std": "y_std"}) 

        fc_df_all = pd.concat([fc_df_raw, fc_df_ml], ignore_index=True)
        err_df_all = make_errors_view(forecast_df=fc_df_all, obs_df=obs_df_test)
        err_df_all.to_csv(eval_cache_path, index=False)
        print(f"   -> Aggregation complete. Error table saved to {eval_cache_path}")

        # ==========================================================================
        # 6. ENSEMBLE DIAGNOSTICS & PLOTS
        # ==========================================================================
        print_step_header(3, 4, "Standard Evaluation Plots")
        
        # A. Errorbar Loss/RMSE
        if len(ensemble_histories) > 0:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
            epochs_loss = np.arange(1, len(ensemble_histories[0]["train_loss"]) + 1)
            colors = {'train': 'blue', 'val': 'orange', 'test': 'green'}
            for split in ['train', 'val', 'test']:
                mu_loss = np.mean([h[f"{split}_loss"] for h in ensemble_histories], axis=0)
                std_loss = np.std([h[f"{split}_loss"] for h in ensemble_histories], axis=0)
                ax1.errorbar(epochs_loss, mu_loss, yerr=std_loss, fmt='o-', label=f"{split.title()}", color=colors[split], capsize=4)
                
                mu_rmse = np.mean([h[f"{split}_hs_rmse"] for h in ensemble_histories], axis=0)
                std_rmse = np.std([h[f"{split}_hs_rmse"] for h in ensemble_histories], axis=0)
                ax2.errorbar(np.arange(len(mu_rmse)), mu_rmse, yerr=std_rmse, fmt='o-', label=f"{split.title()}", color=colors[split], capsize=4)
                
            ax1.set(xlabel="Epoch", ylabel=f"Loss ({config['loss_type'].upper()})", title="Ensemble Loss")
            ax2.set(xlabel="Epoch", ylabel="RMSE Hs (m)", title="Ensemble Hs RMSE")
            for ax in [ax1, ax2]: ax.legend(); ax.grid(True, alpha=0.3)
            plt.savefig(os.path.join(dirs["figures"], "ensemble_loss_rmse_curves.png"), dpi=150)
            plt.close()

        # B. Videos
        video_dir = os.path.join(dirs["figures"], "videos")
        os.makedirs(video_dir, exist_ok=True)
        generate_forecast_video(plot_dashboard_hs_components, err_df_all, save_dir=video_dir, output_filename="ens_hs_components.gif", is_ensemble=True)
        generate_forecast_video(plot_dashboard_bulk_params, err_df_all, save_dir=video_dir, output_filename="ens_bulk_params.gif", is_ensemble=True)

        # C. Boxplots & standard plots
        create_summary_boxplots(err_df=err_df_all, save_path=os.path.join(dirs["figures"], "boxplots"), hs_partitions=range(len(PARTITIONS)), scalar_leads=(3, 24, 72, 120), freq_leads=(24, 120), show_plots=False, model_list=["ecmwf_raw", "ecmwf_ml"], labels=["ECMWF", "ML-Ensemble"])
        plot_heatmap_bias_std(err_df_all, save_path=os.path.join(dirs["figures"], "heatmap_bias_std.png"), ref_model="ecmwf_raw", ref_label="ECMWF", exp_model="ecmwf_ml", exp_label="ML-Ensemble")
        plot_mse_decomposition(err_df_all, save_path=os.path.join(dirs["figures"], "Hs_mse_decomposition.png"), ref_model="ecmwf_raw", ref_label="ECMWF", exp_model="ecmwf_ml", exp_label="ML-Ensemble")
        
        for lead in [3, 24, 72, 120]:
            for var in ["Hs", "Tm02", "Tp"]:
                plot_qq_empirical(err_df_all, lead_time=lead, variable=var, save_dir=dirs["figures"], show_plot=False, ref_model="ecmwf_raw", ref_label="ECMWF", exp_model="ecmwf_ml", exp_label="ML-Ensemble")
        
        plot_pca_structure(err_df_all, save_dir=dirs["figures"], ref_model="ecmwf_raw", ref_label="ECMWF", exp_model="ecmwf_ml", exp_label="ML-Ensemble", variable="E", n_modes=10)

        # D. RMSE LaTeX Table Generation
        target_vars = ["Hs", "Tp", "Tm02"]
        part_map = {0: "Total", 1: "Swell", 2: "WindSea"}
        df_metrics = err_df_all[(err_df_all["variable"].isin(target_vars)) & (err_df_all["x_kind"] == "partition") & (err_df_all["f_idx"].isin(part_map.keys()))].copy()
        df_metrics["part_name"] = df_metrics["f_idx"].map(part_map)
        rmse_df = df_metrics.groupby(["lead_h", "model_id", "part_name", "variable"])["err"].apply(lambda x: np.sqrt((x**2).mean())).reset_index(name="RMSE")
        rmse_df["col_key"] = rmse_df["part_name"] + "_" + rmse_df["variable"]
        pivot_table = rmse_df.pivot(index=["lead_h", "model_id"], columns="col_key", values="RMSE")
        generate_ensemble_latex_rmse(pivot_table, os.path.join(dirs["figures"], "ensemble_rmse_table.tex"))

        # ==========================================================================
        # 7. ADVANCED ENSEMBLE DIAGNOSTICS 
        # ==========================================================================
        print_step_header(4, 4, "Advanced Ensemble Diagnostics")
        
        # PCA Phase Portraits
        plot_pca_mode_swell(err_df_all, save_dir=dirs["figures"], mode_x=8, mode_y=9)
        
        # Dashboards for worst phase error predictions
        worst_ml_dates, worst_ecmwf_dates = get_worst_phase_error_dates(err_df_all, n_extremes=15)
        generate_extreme_error_dashboards(err_df_all, worst_ml_dates, save_dir=dirs["figures"], prefix="ML-Ensemble")
        generate_extreme_error_dashboards(err_df_all, worst_ecmwf_dates, save_dir=dirs["figures"], prefix="ECMWF")

        # Reliability & Spectral Uncertainty Diagnostics
        PLOT_CONFIG = [
            {"var": "Hs",   "f_idx": 1, "label": "Hs swell"},
            {"var": "Tm02", "f_idx": 1, "label": "Tm02 swell"},
            {"var": "Tp",   "f_idx": 0, "label": "Tp"}
        ]
        LEAD_TIMES = [24, 72, 120]
        
        plot_stratified_ranks(checkpoint_dir=dirs["checkpoints"], obs_df=err_df_all, config=PLOT_CONFIG, leads_to_plot=LEAD_TIMES, save_dir=dirs["figures"])
        plot_spectral_uncertainty_diagnostics(checkpoint_dir=dirs["checkpoints"], obs_df=err_df_all, save_dir=dirs["figures"])

    print(f"\n✅ Ensemble Run Complete. Results saved in {os.path.dirname(dirs['weights'])}")

if __name__ == "__main__":
    main()