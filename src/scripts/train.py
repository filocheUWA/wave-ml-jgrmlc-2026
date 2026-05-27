#!/usr/bin/env python3
"""
src/scripts/train.py

Standalone training script for the SpecX wave forecasting model.
Optimized for Setonix HPC with hardcoded scratch paths and notebook-style logging.
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
from src.utils.table_builder import generate_latex_rmse_table

# Plotting Imports
from src.plotting.dashboard import (
    plot_dashboard_hs_components, 
    plot_dashboard_bulk_params, 
    generate_forecast_video
)
from src.plotting.boxplots import create_summary_boxplots
from src.plotting.heatmap import plot_heatmap_bias_std
from src.plotting.qq import plot_qq_empirical
from src.plotting.pca_analysis import plot_pca_structure
from src.plotting.mse_decomposition import plot_mse_decomposition

warnings.filterwarnings("ignore", category=FutureWarning)

# ==============================================================================
# SECTION C: GLOBAL CONFIGURATION & PATHS
# ==============================================================================
# --- Hardware ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Data Paths ---
ENV_DATA_ROOT = os.environ.get('DATA_ROOT')
if ENV_DATA_ROOT:
    DATA_ROOT = ENV_DATA_ROOT
    print(f"📂 Data Root: Using env var DATA_ROOT={DATA_ROOT}")
else:
    DATA_ROOT = "/scratch/pawsey0106/afiloche/data/processed"
    print(f"📂 Data Root: Using hardcoded scratch path {DATA_ROOT}")

PATHS = {
    'forecast_dir': os.path.join(DATA_ROOT, "ecmwf_forecast_nc"),
    'buoy_dir':     os.path.join(DATA_ROOT, "buoy_nc"),
    'tides_path':   os.path.join(DATA_ROOT, "Preludes_tides", "processed_tides.nc")
}

# --- Output Paths ---
ENV_OUTPUT_ROOT = os.environ.get('OUTPUT_ROOT')
if ENV_OUTPUT_ROOT:
    OUTPUT_ROOT = ENV_OUTPUT_ROOT
else:
    OUTPUT_ROOT = "/scratch/pawsey0106/afiloche/output/SpecX"

# --- Fixed Experiment Parameters ---
TRAIN_DATE, VAL_DATE, TEST_DATE = ['20200101_00', '20220621_00'], ['20220701_00', '20221222_00'], ['20230101_00', '20231231_12']

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
    parser = argparse.ArgumentParser(description="Train SpecX Modular Architecture")
    # Model Specs
    parser.add_argument("--model_name", type=str, default='specx')
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
    parser.add_argument("--seed", type=int, default=1234)
    # Output Override
    parser.add_argument("--output_root", type=str, default=OUTPUT_ROOT)
    
    # Workflow Control
    parser.add_argument("--skip_plots", action="store_true", help="If set, skips the inference and plotting phase.")
    
    return parser.parse_args()

def set_reproducibility(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def setup_experiment(args, proposed_config):
    # 1. Generate Run ID
    config_str = json.dumps(proposed_config, sort_keys=True)
    config_hash = hashlib.md5(config_str.encode("utf-8")).hexdigest()[:6]
    run_id = f"{args.model_name}_{config_hash}"
    print(f"🆔 Generated Run ID: {run_id}")

    # 2. Directory Management
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
                print(f"🔍 Found matching config in: {d}")
                exp_id, base_dir = d, os.path.join(root, d)
                found_existing = True
                break
        except: continue

    if not found_existing:
        indices = [int(d.split("_")[1]) for d in existing_dirs]
        next_id = max(indices) + 1 if indices else 1
        exp_id = f"Exp_{next_id:03d}"
        base_dir = os.path.join(root, exp_id)
        print(f"🆕 Creating NEW experiment: {exp_id}")

    dirs = {
        "weights": os.path.join(base_dir, "weights"),
        "figures": os.path.join(base_dir, "figures"),
        "data":    os.path.join(base_dir, "loss_data"),
    }
    for p in dirs.values(): os.makedirs(p, exist_ok=True)

    full_config = proposed_config.copy()
    full_config.update({
        "run_id_str": run_id,
        "save_path": os.path.join(dirs["weights"], f"{run_id}.pt"),
        "history_path": os.path.join(dirs["data"], "history.csv")
    })

    # Incomplete-run protection: restart if metadata exists without weights.
    if found_existing and not os.path.exists(full_config["save_path"]):
        print(f"⚠️  ZOMBIE DETECTED: Folder {exp_id} exists but weights are missing.")
        print(f"   -> Mode: OVERWRITE (Restarting training for {run_id})")
        found_existing = False 

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

    return full_config, dirs, found_existing

def print_step_header(step_num, total_steps, title):
    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Step {step_num}/{total_steps}: {title}...")

# ==============================================================================
# SECTION E: MAIN EXECUTION
# ==============================================================================
def main():
    args = parse_args()
    set_reproducibility(args.seed)
    
    DO_PLOTS = not args.skip_plots

    print(f"⚙️  Using device: {DEVICE}")
    print(f"📂 Output Root: {args.output_root}")
    print(f"📊 Visualization Mode: {'ENABLED' if DO_PLOTS else 'DISABLED'}")

    # 1. Setup Experiment
    arg_vars = vars(args).copy()
    arg_vars.pop('skip_plots', None)
    
    proposed_config = {**arg_vars, **FIXED_PARAMS}
    config, dirs, resumed = setup_experiment(args, proposed_config)

    # 2. Load Data
    print("\n--- Loading Datasets ---")
    print(f"   -> Forecast: {PATHS['forecast_dir']}")
    print(f"   -> Tides:    {PATHS['tides_path']}")
    
    ds_args = {
        'var_list': ['E'], 
        'history_vars': FIXED_PARAMS['history_vars'] if args.use_history else [], 
        'frequency_band': [0.034, 0.51], 
        **PATHS
    }
    train_set = ECMWFDataset(start_end=TRAIN_DATE, **ds_args)
    val_set   = ECMWFDataset(start_end=VAL_DATE, **ds_args)
    test_set  = ECMWFDataset(start_end=TEST_DATE, **ds_args)

    # 3. Scalers & Model Setup
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
    # Partitions updated to 0.10 Hz split
    partitions = [(0.034, 0.51), (0.034, 0.10), (0.10, 0.51)]

    model = SpecX(
        input_size=(40, len(freqs), 36),
        model_size=config["model_size"], input_mode=config["input_mode"],
        n_spec_ch=config["n_spec_ch"], n_coeff_ch=config["n_coeff_ch"], n_output_ch=config["n_output_ch"],
        use_date_default=config["use_date"], use_tide_default=config["use_tide"], 
        use_xattn_default=config["use_xattn"], use_buoy_history_default=args.use_history,
        n_hist_vars=len(ds_args['history_vars'])
    ).to(DEVICE)

    criterion = make_criterion(config["loss_type"], freqs).to(DEVICE)
    optimizer = AdamW(build_param_groups_for_adamw(model, config["weight_decay"]), lr=config["lr"])

    # 4. Training Phase
    if resumed:
        print(f"🚀 Loading weights from {config['save_path']}...")
        model.load_state_dict(torch.load(config["save_path"], map_location=DEVICE))
        try:
            history = pd.read_csv(config["history_path"]).to_dict(orient='list')
            for k in ["train_loss", "val_loss", "test_loss", "lr"]: 
                history[k] = [x for x in history[k] if not np.isnan(x)]
        except FileNotFoundError:
             history = {}
    else:
        print(f"🚀 No existing weights found. Starting training for {config['epochs']} epochs...")
        
        runner = EpochRunner(model, optimizer, criterion, input_scaler=input_scaler, output_scaler=output_scaler,
                            target_mode=config["target_mode"], loss_scale_residual=config["loss_scale"],
                            grad_clip_norm=config["grad_clip_norm"], use_date=config["use_date"],
                            use_tide=config["use_tide"], use_history=args.use_history)
        
        scheduler = build_warmup_cosine_scheduler(optimizer, warmup_epochs=config["warmup_epochs"], total_epochs=config["epochs"], min_lr_factor=config["min_lr_factor"])
        
        history = {"train_loss": [], "val_loss": [], "test_loss": [], "lr": [], "train_hs_rmse": [], "val_hs_rmse": [], "test_hs_rmse": []}
        best_val_rmse = float('inf')

        # --- 4. Epoch 0 Metrics (Pre-Training) ---
        print("Calculating initial metrics before training (Epoch 0)...")
        target_partition = partitions[0] 
        
        for split, loader in zip(["train", "val", "test"], [loaders["train"], loaders["val"], loaders["test"]]):
             rmse = compute_partition_Hs_rmse(model, loader, freqs=freqs, partition=target_partition, input_scaler=input_scaler, output_scaler=output_scaler, target_mode=config["target_mode"], use_date=config["use_date"], use_tide=config["use_tide"])
             history[f"{split}_hs_rmse"].append(float(rmse))
             
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Ep 000/{config['epochs']} | "
              f"RMSE_Hs: tr={history['train_hs_rmse'][0]:.3f}m, val={history['val_hs_rmse'][0]:.3f}m, te={history['test_hs_rmse'][0]:.3f}m")

        # --- 5. Main Loop ---
        for epoch in range(1, config["epochs"] + 1):
            train_stats = runner.run_epoch(loaders["train"], mode="train")
            val_stats   = runner.run_epoch(loaders["val"],   mode="eval")
            test_stats  = runner.run_epoch(loaders["test"],  mode="eval")

            # Compute Physical Metrics
            for split, loader in zip(["train", "val", "test"], [loaders["train"], loaders["val"], loaders["test"]]):
                 rmse = compute_partition_Hs_rmse(model, loader, freqs=freqs, partition=target_partition, input_scaler=input_scaler, output_scaler=output_scaler, target_mode=config["target_mode"], use_date=config["use_date"], use_tide=config["use_tide"])
                 history[f"{split}_hs_rmse"].append(float(rmse))

            # Update History
            scheduler.step()
            current_lr = optimizer.param_groups[0]["lr"]
            history["lr"].append(current_lr)
            history["train_loss"].append(train_stats["loss_mean"])
            history["val_loss"].append(val_stats["loss_mean"])
            history["test_loss"].append(test_stats["loss_mean"])

            # Save Best based on Validation Hs RMSE
            saved_marker = ""
            if history['val_hs_rmse'][-1] < best_val_rmse:
                best_val_rmse = history['val_hs_rmse'][-1]
                torch.save(model.state_dict(), config["save_path"])
                saved_marker = "<- BEST (Hs RMSE)"

            print(f"[{datetime.now().strftime('%H:%M:%S')}] Ep {epoch:03d}/{config['epochs']} | "
                  f"LR={current_lr:.1e} | "
                  f"Loss: tr={history['train_loss'][-1]:.4f}, val={history['val_loss'][-1]:.4f}, te={history['test_loss'][-1]:.4f} | "
                  f"RMSE_Hs: tr={history['train_hs_rmse'][-1]:.3f}m, val={history['val_hs_rmse'][-1]:.3f}m, te={history['test_hs_rmse'][-1]:.3f}m {saved_marker}")

        # --- 6. Save History ---
        print(f"\nTraining finished. Saving history to {config['history_path']}...")
        save_hist = history.copy()
        for k in ["train_loss", "val_loss", "test_loss", "lr"]: 
            save_hist[k] = [np.nan] + save_hist[k]
            
        pd.DataFrame(save_hist).to_csv(config['history_path'], index=False)
        print("✅ History saved.")

    # ==============================================================================
    # SECTION 5: INFERENCE & VISUALIZATION (CONDITIONAL)
    # ==============================================================================
    if DO_PLOTS:
        print("\n========================================")
        print("📊 STARTING POST-TRAINING ANALYSIS")
        print("========================================")
        
        # --- Step 1: Inference & Error Table ---
        print_step_header(1, 8, "Inference & Error Table Generation")
        cache_path = os.path.join(dirs["data"], "test_err_df.csv")
        
        if not os.path.exists(cache_path):
            print("   -> Cache not found. Running model inference on Test Set...")
            model.load_state_dict(torch.load(config["save_path"], map_location=DEVICE))
            model.eval()
            
            y_pred, _ = predict_dataset(loaders["test"], model, input_scaler=input_scaler, output_scaler=output_scaler, target_mode=config["target_mode"], use_date=config["use_date"], use_tide=config["use_tide"], use_history=args.use_history)
            
            test_set_ml = MLCorrectedDataset(test_set, y_pred)
            vars_an = ["E", "Hs", "Tm01", "Tm02", "Tp"]
            
            print("   -> Building Forecast Tables...")
            obs_tbl = build_observations_table(test_set, variables=vars_an, partitions=partitions)
            raw_tbl = build_forecast_table(test_set, model_id="ecmwf_raw", variables=vars_an, partitions=partitions)
            ml_tbl  = build_forecast_table(test_set_ml, model_id="ecmwf_ml", variables=vars_an, partitions=partitions)
            
            err_df_all = make_errors_view(pd.concat([raw_tbl, ml_tbl], ignore_index=True), obs_tbl)
            err_df_all.to_csv(cache_path, index=False)
            print(f"   -> Saved error table to: {cache_path}")
        else:
            print(f"   -> Loading cached error table from: {cache_path}")
            err_df_all = pd.read_csv(cache_path)

        # --- Step 2: Forecast Videos ---
        print_step_header(2, 8, "Generating Forecast Videos")
        video_dir = os.path.join(dirs["figures"], "videos")
        os.makedirs(video_dir, exist_ok=True)
        
        print("   -> Rendering Hs Components video...")
        generate_forecast_video(plot_dashboard_hs_components, err_df_all, save_dir=video_dir, fps=2, output_filename="dashboard_hs_components.gif")
        
        print("   -> Rendering Bulk Parameters video...")
        generate_forecast_video(plot_dashboard_bulk_params, err_df_all, save_dir=video_dir, fps=2, output_filename="dashboard_bulk_params.gif")

        # --- Step 3: Boxplots ---
        print_step_header(3, 8, "Statistical Boxplots")
        create_summary_boxplots(err_df_all, save_path=dirs["figures"], show_plots=False)

        # --- Step 4: Heatmap ---
        print_step_header(4, 8, "Error Heatmap (Bias & Std)")
        plot_heatmap_bias_std(err_df_all, save_path=os.path.join(dirs["figures"], "heatmap_bias_std.png"), variable="E")

        # --- Step 5: MSE Decomposition ---
        print_step_header(5, 8, "MSE Decomposition Analysis")
        plot_mse_decomposition(
            err_df_all, 
            save_path=os.path.join(dirs["figures"], "Hs_mse_decomposition.png"),
            ref_model="ecmwf_raw", ref_label="ECMWF",
            exp_model="ecmwf_ml",  exp_label="ML"
        )

        # --- Step 6: QQ Plots ---
        print_step_header(6, 8, "Empirical Q-Q Plots")
        for lead in [24, 48, 72, 120]:
            for var in ["Hs", "Tp"]: 
                plot_qq_empirical(err_df_all, lead_time=lead, variable=var, save_dir=dirs["figures"], show_plot=False)

        # --- Step 7: PCA Analysis ---
        print_step_header(7, 8, "PCA Error Structure")
        plot_pca_structure(err_df_all, save_dir=dirs["figures"], ref_model="ecmwf_raw", ref_label="ECMWF", exp_model="ecmwf_ml", exp_label="ML", variable="E")

        # --- Step 8: LaTeX RMSE Table ---
        print_step_header(8, 8, "LaTeX RMSE Table")
        try:
            generate_latex_rmse_table(err_df_all, output_dir=dirs["figures"])
        except Exception as e:
            print(f"⚠️ Failed to generate LaTeX table: {e}")

    else:
        print("\n🚫 Skipping Post-Training Analysis (--skip_plots set)")

    print(f"\n✅ Run Complete. Results saved in {os.path.dirname(dirs['weights'])}")

if __name__ == "__main__":
    main()
