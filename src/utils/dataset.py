"""Dataset utilities for paired ECMWF forecasts, buoy spectra, and context."""

import os
import json
import torch
import numpy as np
import pandas as pd
import xarray as xr
import netCDF4 as nc
from datetime import datetime, timedelta
from torch.utils.data import Dataset

class ECMWFDataset(Dataset):
    """
    Forecast-issuance dataset for supervised spectral post-processing.

    Each sample contains one ECMWF forecast sequence, the aligned buoy target,
    masks for missing observations, and optional environmental context.
    """
    def __init__(self,
                 forecast_dir='../data/processed/ecmwf_forecast_nc',
                 buoy_dir='../data/processed/buoy_nc',
                 tides_path='../data/processed/Preludes_tides/processed_tides.nc',
                 var_list=['E'],
                 history_vars=['E'],
                 start_end=['20210101_00', '20220101_00'],
                 history_days=5,       
                 history_step_hours=3, 
                 frequency_band=None,
                 model_freqs=None,
                 scaler_path: str | None = None):

        # Match all spectra to the selected ECMWF frequency band.
        if model_freqs is None:
            f0 = 0.0345
            model_freqs = f0 * torch.logspace(0, 35, steps=36, base=1.1)
        self.model_freqs = model_freqs.cpu().numpy()

        if frequency_band is not None:
            fmin, fmax = frequency_band
            mask = (self.model_freqs >= fmin) & (self.model_freqs <= fmax)
            self.freq_idx = np.where(mask)[0]
        else:
            self.freq_idx = np.arange(len(self.model_freqs))
        self.F = len(self.freq_idx)
        self.selected_freqs = self.model_freqs[self.freq_idx]

        # Load deterministic tide context indexed by valid time.
        ds_tide = xr.open_dataset(tides_path)
        tide_df = ds_tide.to_dataframe()
        if not isinstance(tide_df.index, pd.DatetimeIndex):
            tide_df.index = pd.to_datetime(tide_df.index)
        self.tide_series = tide_df['etaMSL'].astype(np.float32)

        # Optional legacy scaler support for pre-scaled dataset access.
        self.scaling_enabled = False
        if scaler_path:
            if os.path.exists(scaler_path):
                with open(scaler_path, 'r') as f:
                    stats = json.load(f)
                self.mean_log = float(stats['mean_log'])
                self.std_log = float(stats['std_log'])
                self.scaling_enabled = True
                self.epsilon = 1e-6
                print(f"Scaler loaded from {scaler_path}. Scaling is ENABLED.")
            else:
                print(f"Warning: Scaler path provided but not found. Scaling is DISABLED.")

        # Store file locations and context-variable configuration.
        self.forecast_dir = forecast_dir
        self.buoy_dir     = buoy_dir
        self.var_list     = var_list
        self.history_vars = history_vars
        self.n_hist_vars  = len(history_vars)

        self.start_time = datetime.strptime(start_end[0], "%Y%m%d_%H")
        self.end_time   = datetime.strptime(start_end[1], "%Y%m%d_%H")

        self.forecast_files = sorted([
            f for f in os.listdir(forecast_dir)
            if f.endswith('.nc') and self.start_time <= self._fname_to_dt(f) <= self.end_time
        ])

        # Preload buoy-history context for efficient random access.
        self.history_window_hours = history_days * 24
        self.history_step = history_step_hours
        self.n_history_steps = self.history_window_hours // self.history_step
        
        print(f"Building Generic History Buffer ({history_days} days, vars: {history_vars})...")
        self._preload_history_buffer()

    def _fname_to_dt(self, fname):
        return datetime.strptime(fname.split('.')[0], "%Y%m%d_%H")

    def get_issuance_time(self, idx):
        """Helper to safely get issuance time for the sample at idx."""
        return self._fname_to_dt(self.forecast_files[idx])

    def _preload_history_buffer(self):
        buffer_start = self.start_time - timedelta(days=5)
        buffer_end = self.end_time
        # History timeline extends before the split to cover early issuances.
        self.history_timeline = pd.date_range(start=buffer_start, end=buffer_end, freq=f'{self.history_step}h')
        
        T_total = len(self.history_timeline)
        # Buffer shape: `(time, frequency, history_variable)`.
        self.history_buffer = torch.zeros((T_total, self.F, self.n_hist_vars), dtype=torch.float32)
        self.history_mask = torch.zeros((T_total, self.F, self.n_hist_vars), dtype=torch.float32)
        
        self.dt_to_idx = {dt: i for i, dt in enumerate(self.history_timeline)}
        available_files = set(os.listdir(self.buoy_dir))
        
        for i, dt in enumerate(self.history_timeline):
            fname = dt.strftime("%Y%m%d_%H.nc")
            if fname in available_files:
                try:
                    with nc.Dataset(os.path.join(self.buoy_dir, fname), 'r') as ds:
                        for v_idx, var in enumerate(self.history_vars):
                            if var in ds.variables:
                                val = ds.variables[var][self.freq_idx]
                                self.history_buffer[i, :, v_idx] = torch.from_numpy(val)
                                self.history_mask[i, :, v_idx] = 1.0
                except: pass

    # -------------------------------------------------------------------------
    # Transforms and spectral helpers
    # -------------------------------------------------------------------------
    def _apply_transform(self, tensor: torch.Tensor) -> torch.Tensor:
        tensor_log = torch.log10(tensor + self.epsilon)
        return (tensor_log - self.mean_log) / self.std_log

    def inverse_transform(self, tensor: torch.Tensor) -> torch.Tensor:
        tensor_unscaled_log = (tensor * self.std_log) + self.mean_log
        return torch.pow(10, tensor_unscaled_log) - self.epsilon

    @staticmethod
    def compute_energy_spectrum(X, dtheta=10 * torch.pi / 180, dir_axis=2):
        """Integrate directional energy density to a 1D frequency spectrum."""
        return X.sum(dim=dir_axis) * dtheta

    def __len__(self):
        return len(self.forecast_files)

    # -------------------------------------------------------------------------
    # Sample assembly
    # -------------------------------------------------------------------------
    def __getitem__(self, idx):
        forecast_file = self.forecast_files[idx]
        issuance_dt   = self._fname_to_dt(forecast_file)

        # ECMWF directional spectrum, subset to the selected frequency band.
        nc_path = os.path.join(self.forecast_dir, forecast_file)
        with nc.Dataset(nc_path, 'r') as ds:
            X_np = ds.variables['efth'][:] 
        X_np = X_np[:, :, self.freq_idx]
        X = torch.from_numpy(np.expand_dims(X_np, 0).astype(np.float32))

        # Buoy target spectrum and validity mask over the forecast horizon.
        V = len(self.var_list)
        Y_np   = np.full((V, 40, 1, self.F), 0, dtype=np.float32)
        mask_Y = np.zeros_like(Y_np, dtype=np.float32)
        lead_hours_list = [3 * (i + 1) for i in range(40)]

        # Normalized temporal coordinates for lead time and calendar phase.
        lead_norm = np.empty(40, np.float32)
        week_norm = np.empty(40, np.float32)
        doy_norm  = np.empty(40, np.float32)
        dom_norm  = np.empty(40, np.float32)
        dow_norm  = np.empty(40, np.float32)
        hour_norm = np.empty(40, np.float32)

        for t_idx, lead_h in enumerate(lead_hours_list):
            valid_dt = issuance_dt + timedelta(hours=lead_h)
            buoy_fname = valid_dt.strftime("%Y%m%d_%H.nc")
            buoy_path  = os.path.join(self.buoy_dir, buoy_fname)
            
            if os.path.exists(buoy_path):
                try:
                    with nc.Dataset(buoy_path, 'r') as ds_buoy:
                        for v_idx, var in enumerate(self.var_list):
                            if var in ds_buoy.variables:
                                val = ds_buoy.variables[var][self.freq_idx]
                                Y_np[v_idx, t_idx, 0, :] = val
                                mask_Y[v_idx, t_idx, 0, :] = 1.0
                except: pass

            lead_norm[t_idx] = (lead_h - 3) / 120.0
            week_norm[t_idx] = (valid_dt.isocalendar().week % 52) / 52.0
            doy_norm[t_idx]  = (valid_dt.timetuple().tm_yday - 1) / 364.0
            dom_norm[t_idx]  = (valid_dt.day - 1) / 31.0
            dow_norm[t_idx]  = valid_dt.weekday() / 7.0
            hour_norm[t_idx] = valid_dt.hour / 24.0

        date_encoding = torch.tensor(
            np.stack([lead_norm, week_norm, doy_norm, dom_norm, dow_norm, hour_norm], axis=0),
            dtype=torch.float32
        )

        # Tide context is aligned to the forecast valid-time horizon.
        start_tide = issuance_dt + timedelta(hours=3)
        end_tide   = issuance_dt + timedelta(hours=122, minutes=45)
        rng = pd.date_range(start=start_tide, end=end_tide, freq='15min', inclusive='both')
        tide = torch.tensor(self.tide_series.loc[rng].to_numpy(np.float32).reshape(1, -1))

        # Raw ECMWF 1D spectrum used as the baseline forecast.
        Y_model = self.compute_energy_spectrum(X, dir_axis=2)
        Y = torch.from_numpy(Y_np)
        
        # Recent buoy-history context ending at the forecast issuance time.
        if issuance_dt in self.dt_to_idx:
            idx_0 = self.dt_to_idx[issuance_dt]
            start = idx_0 - (self.n_history_steps - 1)
            end = idx_0 + 1
            if start >= 0:
                H = self.history_buffer[start : end]      # (40, F, n_vars)
                H_mask = self.history_mask[start : end]   # (40, F, n_vars)
            else:
                H = torch.zeros((self.n_history_steps, self.F, self.n_hist_vars))
                H_mask = torch.zeros((self.n_history_steps, self.F, self.n_hist_vars))
        else:
            H = torch.zeros((self.n_history_steps, self.F, self.n_hist_vars))
            H_mask = torch.zeros((self.n_history_steps, self.F, self.n_hist_vars))

        # Optional on-dataset scaling for legacy workflows.
        if self.scaling_enabled:
            Y = self._apply_transform(Y)
            Y_model = self._apply_transform(Y_model)
            H = self._apply_transform(H)
        Y_model = Y_model.unsqueeze(2)

        return X, Y, torch.from_numpy(mask_Y), date_encoding, tide, Y_model, H, H_mask
