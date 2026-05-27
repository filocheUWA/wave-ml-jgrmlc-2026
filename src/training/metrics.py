"""Physical forecast-skill metrics derived from corrected spectra."""

import torch
from src.training.scalers import Scaler

@torch.no_grad()
def compute_partition_Hs_rmse(
    model,
    loader,
    *,
    freqs: torch.Tensor,
    input_scaler: Scaler,
    output_scaler: Scaler,
    target_mode: str,
    partition: tuple[float, float] = (0.034, 0.51),
    use_date: bool = False,
    use_tide: bool = False,
) -> float:
    """
    Compute RMSE of significant wave height over a frequency partition.

    Predictions and observations are integrated over the same valid frequency
    bins so missing buoy records do not contribute to the score.
    """
    device = next(model.parameters()).device
    model.eval()

    freqs = freqs.to(device)
    fmin, fmax = partition
    pmask = ((freqs >= fmin) & (freqs < fmax))
    pmask_f = pmask.to(torch.float32)

    se_sum = 0.0
    n_eff  = 0

    for X, Y, M, D, T, YM, _, _ in loader:
        X_raw  = X.to(device, non_blocking=True)
        Y_raw  = Y.to(device, non_blocking=True)
        M_raw  = M.to(device, non_blocking=True)
        YM_raw = YM.to(device, non_blocking=True)

        X_in = input_scaler.transform(X_raw)
        YM_in = input_scaler.transform(YM_raw)

        fwd_kwargs = {}
        if use_date:
            fwd_kwargs["date_encoding"] = D.to(device, non_blocking=True)
        if use_tide:
            fwd_kwargs["tide"] = T.to(device, non_blocking=True)

        if model.input_mode == 'spectra':
            x_s, x_c = X_in, None
        elif model.input_mode == 'coeffs':
            x_s, x_c = None, YM_in
        else:
            x_s, x_c = X_in, YM_in

        out_scaled = model(x_spec=x_s, x_coeff=x_c, **fwd_kwargs)
        out_physical = output_scaler.inverse_transform(out_scaled)

        out_physical = out_physical.squeeze(-1).squeeze(1)
        Y_tgt  = Y_raw.squeeze(-1).squeeze(1)
        M_mask = M_raw.squeeze(-1).squeeze(1)
        YM_res = YM_raw.squeeze(-1).squeeze(1)

        # Compose the physical corrected spectrum for the selected target mode.
        Y_pred = out_physical if target_mode == "direct" else (YM_res + out_physical)
        
        # Wave energy density is non-negative before spectral integration.
        Y_pred = Y_pred.clamp_min(0.0)

        # Integrate observations only where buoy spectral bins are valid.
        obs_mask = (M_mask > 0.5).to(torch.float32) * pmask_f
        m0_true = torch.trapz(Y_tgt * obs_mask, freqs, dim=-1).clamp_min(0.0)
        hs_true = 4.0 * torch.sqrt(m0_true)
        valid   = (obs_mask.sum(dim=-1) > 0)

        # Use the identical mask for the corrected forecast.
        m0_pred = torch.trapz(Y_pred * obs_mask, freqs, dim=-1).clamp_min(0.0)
        hs_pred = 4.0 * torch.sqrt(m0_pred)

        diff = (hs_pred - hs_true)[valid]
        se_sum += float((diff * diff).sum().item())
        n_eff  += int(valid.sum().item())

    return (se_sum / max(n_eff, 1)) ** 0.5


@torch.no_grad()
def compute_baseline_hs_rmse(
    loader,
    *,
    freqs: torch.Tensor,
    output_scaler: Scaler,
    partition: tuple[float, float] = (0.034, 0.51),
) -> float:
    """
    Compute baseline Hs RMSE for the raw ECMWF 1D spectrum.
    """
    device = freqs.device

    fmin, fmax = partition
    pmask = ((freqs >= fmin) & (freqs < fmax))
    pmask_f = pmask.to(torch.float32)

    se_sum = 0.0
    n_eff  = 0

    for _, Y, M, _, _, YM, _, _ in loader:
        Y_tgt  = Y.to(device, non_blocking=True).squeeze(-1).squeeze(1)
        M_mask = M.to(device, non_blocking=True).squeeze(-1).squeeze(1)
        YM_raw = YM.to(device, non_blocking=True).squeeze(-1).squeeze(1)
        
        # Keep the baseline physically non-negative before integration.
        YM_raw = YM_raw.clamp_min(0.0)

        # Integrate observations only where buoy spectral bins are valid.
        obs_mask = (M_mask > 0.5).to(torch.float32) * pmask_f
        m0_true = torch.trapz(Y_tgt * obs_mask, freqs, dim=-1).clamp_min(0.0)
        hs_true = 4.0 * torch.sqrt(m0_true)
        valid   = (obs_mask.sum(dim=-1) > 0)

        # Use the raw forecast with the identical observation mask.
        m0_pred = torch.trapz(YM_raw * obs_mask, freqs, dim=-1).clamp_min(0.0)
        hs_pred = 4.0 * torch.sqrt(m0_pred)

        diff = (hs_pred - hs_true)[valid]
        se_sum += float((diff * diff).sum().item())
        n_eff  += int(valid.sum().item())

    return (se_sum / max(n_eff, 1)) ** 0.5
