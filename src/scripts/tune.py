#!/usr/bin/env python3
"""
src/scripts/tune.py

Robust Hyperparameter Optimization (HPO) script using Hyperopt.
Features:
- Resume capability (pickled Trials)
- Top-5 model checkpointing (saves disk space)
- Full history logging per trial
- Atomic updates for HPC fault tolerance
"""

# ==============================================================================
# SECTION A: BOOTSTRAP & ENVIRONMENT SETUP (MATCHING TRAIN.PY)
# ==============================================================================
import os
import sys

# 1. Fix Matplotlib Disk Quota Issue
# Redirect font cache to scratch to prevent [Errno 122] Disk quota exceeded
MYSCRATCH = os.environ.get('MYSCRATCH', '/scratch/pawsey0106/afiloche')
mpl_cache_dir = os.path.join(MYSCRATCH, '.cache', 'matplotlib')
os.makedirs(mpl_cache_dir, exist_ok=True)
os.environ['MPLCONFIGDIR'] = mpl_cache_dir
print(f"🔧 Environment: Redirected Matplotlib cache to {mpl_cache_dir}")

# 2. Project Path Resolution (for local imports)
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../../"))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# ==============================================================================
# SECTION B: IMPORTS
# ==============================================================================
import pickle
import csv
import json
import logging
import argparse
import shutil
import numpy as np
import pandas as pd
import torch
import warnings
import random
from datetime import datetime
from torch.utils.data import DataLoader
from torch.optim import AdamW

# Hyperopt Imports
# Added generate_trials_to_calculate for expert warm-start
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK, STATUS_FAIL
from hyperopt.fmin import generate_trials_to_calculate

# Local Project Imports (Aligned with train.py)
from src.architecture.model_specx import SpecX
from src.training import * # Imports EpochRunner, ECMWFDataset, make_criterion, etc.
from src.training.scalers import IdentityScaler, LogZScoreScaler, BinwiseZScoreScaler
from src.training.data_management import collate
from src.utils import * # Imports compute_partition_Hs_rmse, etc.

warnings.filterwarnings("ignore")

# ==============================================================================
# CONFIGURATION
# ==============================================================================
N_TRIALS = 500        
EPOCHS_PER_TRIAL = 15 
TOP_K_MODELS = 10     # Number of best models to keep weights for

# Paths
ENV_DATA_ROOT = os.environ.get('DATA_ROOT')
DATA_ROOT = ENV_DATA_ROOT if ENV_DATA_ROOT else "/scratch/pawsey0106/afiloche/data/processed"

ENV_OUTPUT_ROOT = os.environ.get('OUTPUT_ROOT')
OUTPUT_ROOT = ENV_OUTPUT_ROOT if ENV_OUTPUT_ROOT else "/scratch/pawsey0106/afiloche/output_tuning"

PATHS = {
    'forecast_dir': os.path.join(DATA_ROOT, "ecmwf_forecast_nc"),
    'buoy_dir':     os.path.join(DATA_ROOT, "buoy_nc"),
    'tides_path':   os.path.join(DATA_ROOT, "Preludes_tides", "processed_tides.nc")
}

# Fixed Params (Not tuned)
FIXED_PARAMS = {
    "history_vars": ['E'], 
    "num_workers": 6,
    "weight_decay": 1e-2, 
    "grad_clip_norm": 1.0, 
    "loss_scale": 5.0, 
    "n_spec_ch": 1, 
    "n_coeff_ch": 1, 
    "n_output_ch": 1, 
    "warmup_epochs": 2,
    "min_lr_factor": 0.1,
}

# Global Data Cache (Populated in main)
DATASETS = {}
LOADERS = {}
FREQS = None

# ==============================================================================
# SEARCH SPACE
# ==============================================================================
SPACE = {
    # Architecture
    'model_name': 'specx',
    'model_size': hp.choice('model_size', ['small', 'base', 'large']),
    'use_xattn': hp.choice('use_xattn', [True, False]),
    
    # Input
    'input_mode': hp.choice('input_mode', ["spectra", "coeffs"]), 
    'scale_input': hp.choice('scale_input', [True, False]),
    'use_history': hp.choice('use_history', [True, False]),
    'use_date': hp.choice('use_date', [True, False]),
    'use_tide': hp.choice('use_tide', [True, False]),
    
    # Output
    'target_mode': hp.choice('target_mode', ["direct", "residual"]),
    'scale_output': hp.choice('scale_output', [True, False]),
    
    # Training
    'loss_type': hp.choice('loss_type', ["mse", "mae", "binweighted_mse", "binweighted_mae"]),
    'lr': hp.choice('lr', [1e-3, 5e-4, 1e-4]),
    'batch_size': hp.choice('batch_size', [8, 16, 32, 64])
}

# ==============================================================================
# UTILS
# ==============================================================================
def get_trial_dir(trial_id):
    return os.path.join(OUTPUT_ROOT, "trials", f"{trial_id:04d}")

def manage_top_k_models(summary_path):
    """
    Scans summary.csv, identifies the top K models, and deletes .pt files
    for any model NOT in the top K.
    """
    if not os.path.exists(summary_path): return

    df = pd.read_csv(summary_path)
    if df.empty: return

    # Sort by score (Ascending RMSE is better)
    df_sorted = df.sort_values("val_rmse", ascending=True)
    
    # Identify Keepers
    top_ids = df_sorted.head(TOP_K_MODELS)["trial_id"].astype(int).tolist()
    
    # Scan directory for all saved weights
    trials_dir = os.path.join(OUTPUT_ROOT, "trials")
    for d in os.listdir(trials_dir):
        try:
            tid = int(d)
        except ValueError: continue
            
        weight_path = os.path.join(trials_dir, d, "model.pt")
        
        if os.path.exists(weight_path):
            if tid not in top_ids:
                print(f"🗑️  Cleaning: Deleting weights for Trial {tid} (Not in Top {TOP_K_MODELS})")
                os.remove(weight_path)

def save_trial_result(trial_id, params, history, best_epoch_idx, model_state):
    """Saves logs and optionally weights."""
    t_dir = get_trial_dir(trial_id)
    os.makedirs(t_dir, exist_ok=True)

    # 1. Save Config
    with open(os.path.join(t_dir, "config.json"), "w") as f:
        json.dump(params, f, indent=4)

    # 2. Save History CSV
    pd.DataFrame(history).to_csv(os.path.join(t_dir, "history.csv"), index=False)

    # 3. Update Summary CSV
    summary_path = os.path.join(OUTPUT_ROOT, "summary.csv")
    best_rmse = history["val_hs_rmse"][best_epoch_idx]
    
    row = {
        "trial_id": trial_id,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "val_rmse": best_rmse,
        "best_epoch": best_epoch_idx,
        **params
    }
    
    # Atomic append
    df_row = pd.DataFrame([row])
    if not os.path.exists(summary_path):
        df_row.to_csv(summary_path, index=False)
    else:
        df_row.to_csv(summary_path, mode='a', header=False, index=False)

    # 4. Save Weights (Temporarily save ALL, then clean up)
    torch.save(model_state, os.path.join(t_dir, "model.pt"))
    
    # 5. Clean up old weights
    manage_top_k_models(summary_path)

# ==============================================================================
# OBJECTIVE FUNCTION
# ==============================================================================
def objective(params):
    # 1. Setup ID
    trial_id = params.pop('trial_id_injected') 
    
    print(f"\n=== Starting Trial {trial_id:04d} ===")
    print(f"Params: {json.dumps(params, indent=2)}")
    
    try:
        # 2. Set Seed
        seed = 1234
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        # 3. Prepare Scalers
        train_set = DATASETS['train']
        
        input_scaler = LogZScoreScaler() if params['scale_input'] else IdentityScaler()
        if params['scale_input']: input_scaler.fit(train_set, fit_on='X')
        
        output_scaler = IdentityScaler()
        if params['scale_output']:
            scale_key = 'Y' if params['target_mode'] == 'direct' else 'Y-YM'
            ScalerClass = LogZScoreScaler if params['target_mode'] == 'direct' else BinwiseZScoreScaler
            output_scaler = ScalerClass()
            output_scaler.fit(train_set, fit_on=scale_key)

        # 4. Loaders
        loader_args = {
            "batch_size": params['batch_size'], 
            "num_workers": FIXED_PARAMS['num_workers'], 
            "pin_memory": True, 
            "collate_fn": collate
        }
        train_loader = DataLoader(DATASETS['train'], shuffle=True, **loader_args)
        val_loader   = DataLoader(DATASETS['val'],   shuffle=False, **loader_args)

        # 5. Model
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        freqs = FREQS
        
        model = SpecX(
            input_size=(40, len(freqs), 36),
            model_size=params["model_size"], input_mode=params["input_mode"],
            n_spec_ch=FIXED_PARAMS["n_spec_ch"], n_coeff_ch=FIXED_PARAMS["n_coeff_ch"], n_output_ch=FIXED_PARAMS["n_output_ch"],
            use_date_default=params["use_date"], use_tide_default=params["use_tide"], 
            use_xattn_default=params["use_xattn"], use_buoy_history_default=params["use_history"],
            n_hist_vars=1
        ).to(device)

        criterion = make_criterion(params["loss_type"], freqs).to(device)
        optimizer = AdamW(build_param_groups_for_adamw(model, FIXED_PARAMS["weight_decay"]), lr=params["lr"])
        
        runner = EpochRunner(
            model, optimizer, criterion, input_scaler=input_scaler, output_scaler=output_scaler,
            target_mode=params["target_mode"], loss_scale_residual=FIXED_PARAMS["loss_scale"],
            grad_clip_norm=FIXED_PARAMS["grad_clip_norm"], use_date=params["use_date"],
            use_tide=params["use_tide"], use_history=params["use_history"]
        )
        
        scheduler = build_warmup_cosine_scheduler(
            optimizer, warmup_epochs=FIXED_PARAMS["warmup_epochs"], 
            total_epochs=EPOCHS_PER_TRIAL, min_lr_factor=FIXED_PARAMS["min_lr_factor"]
        )

        # 6. Training Loop
        history = {"train_loss": [], "val_loss": [], "val_hs_rmse": []}
        best_val_rmse = float('inf')
        best_model_state = None
        best_epoch = 0

        target_partition = (0.034, 0.51) # Total Hs
        
        for epoch in range(1, EPOCHS_PER_TRIAL + 1):
            train_stats = runner.run_epoch(train_loader, mode="train")
            val_stats   = runner.run_epoch(val_loader,   mode="eval")
            
            # Physical Metric
            val_rmse = compute_partition_Hs_rmse(
                model, val_loader, freqs=freqs, partition=target_partition, 
                input_scaler=input_scaler, output_scaler=output_scaler, 
                target_mode=params["target_mode"], use_date=params["use_date"], use_tide=params["use_tide"]
            )
            
            history["train_loss"].append(train_stats["loss_mean"])
            history["val_loss"].append(val_stats["loss_mean"])
            history["val_hs_rmse"].append(float(val_rmse))
            
            scheduler.step()

            # Track Best
            if val_rmse < best_val_rmse:
                best_val_rmse = val_rmse
                best_model_state = model.state_dict() 
                best_epoch = epoch

            # Check divergence
            if np.isnan(train_stats["loss_mean"]):
                print("💥 Loss diverged (NaN). Pruning trial.")
                return {'loss': 9999.0, 'status': STATUS_FAIL}

        # 7. Save Results
        print(f"✅ Trial {trial_id} Finished. Best Val RMSE: {best_val_rmse:.4f} (Ep {best_epoch})")
        save_trial_result(trial_id, params, history, best_epoch_idx=best_epoch-1, model_state=best_model_state)

        return {
            'loss': best_val_rmse,
            'status': STATUS_OK,
            'trial_id': trial_id
        }

    except Exception as e:
        print(f"❌ Trial {trial_id} Failed: {str(e)}")
        return {'status': STATUS_FAIL}

# ==============================================================================
# MAIN (Orchestrator)
# ==============================================================================
def main():
    print(f"🔧 Starting Robust HPO for {N_TRIALS} trials")
    print(f"📂 Output Dir: {OUTPUT_ROOT}")
    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_ROOT, "trials"), exist_ok=True)

    # 1. Load Data Globally (Once)
    print("--- Loading Datasets into Memory ---")
    TRAIN_DATE, VAL_DATE = ['20200101_00', '20220622_00'], ['20220701_00', '20221222_00']
    
    ds_args = {
        'var_list': ['E'], 
        'history_vars': FIXED_PARAMS['history_vars'],
        'frequency_band': [0.034, 0.51], 
        **PATHS
    }
    
    global FREQS
    DATASETS['train'] = ECMWFDataset(start_end=TRAIN_DATE, **ds_args)
    DATASETS['val']   = ECMWFDataset(start_end=VAL_DATE,   **ds_args)
    FREQS = torch.as_tensor(DATASETS['train'].selected_freqs, dtype=torch.float32)
    print(f"✅ Data Loaded. Train: {len(DATASETS['train'])}, Val: {len(DATASETS['val'])}")

    # 2. Load/Init Trials
    trials_path = os.path.join(OUTPUT_ROOT, "trials.pkl")
    if os.path.exists(trials_path):
        print(f"🔄 Resuming from {trials_path}")
        with open(trials_path, "rb") as f:
            trials = pickle.load(f)
    else:
        print("🆕 Starting fresh trials database with Expert Initial Guess")
        # Define Expert starting point
        initial_guesses = [
            {
                'model_size': 1,      # 'base' is index 1
                'use_xattn': 1,       # False is index 1
                'input_mode': 0,      # 'spectra' is index 0
                'scale_input': 0,     # True is index 0
                'use_history': 1,     # False is index 1
                'use_date': 0,        # True is index 0
                'use_tide': 0,        # True is index 0
                'target_mode': 0,     # 'direct' is index 0
                'scale_output': 0,    # True is index 0
                'loss_type': 1,       # 'mae' is index 1
                'lr': 2,              # 0.0001 (1e-4) is index 2
                'batch_size': 1,      # 16 is index 1
            }
        ]
        # Initialize with expert guess
        trials = generate_trials_to_calculate(initial_guesses)

    # 3. Robust Loop
    for i in range(len(trials), N_TRIALS):
        print(f"\n--- Global Progress: {i+1}/{N_TRIALS} ---")
        
        current_trial_id = i + 1
        
        def objective_wrapper(params):
            params['trial_id_injected'] = current_trial_id
            return objective(params)
        
        fmin(
            fn=objective_wrapper,
            space=SPACE,
            algo=tpe.suggest,
            max_evals=i + 1,
            trials=trials,
            rstate=np.random.default_rng(1234+i),
            show_progressbar=False
        )
        
        with open(trials_path + ".tmp", "wb") as f:
            pickle.dump(trials, f)
        os.replace(trials_path + ".tmp", trials_path)
        print(f"💾 Checkpoint saved: {len(trials)} trials complete.")

    print("\n🎉 Hyperparameter Optimization Complete!")

if __name__ == "__main__":
    main()