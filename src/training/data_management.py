"""Batch collation and inference helpers for spectral forecast post-processing."""

import torch
from torch.utils.data import Dataset
from src.training.scalers import Scaler

def collate(batch):
    """
    Convert dataset samples to the model tensor layout.

    The dataset stores spectra in file-oriented order; SpecX consumes tensors as
    `(B, C, T, F, Theta)` with singleton `Theta=1` for 1D spectra.
    """
    X, Y, M, D, T, YM, H, H_mask = zip(*batch)

    # Permute spectra for (B,C,T,F,Θ) layout
    X    = torch.stack(X).permute(0, 1, 2, 4, 3)

    # Permute 1D tensors for (B,C,T,F,1) layout
    Y    = torch.stack(Y).permute(0, 1, 2, 4, 3)
    M    = torch.stack(M).permute(0, 1, 2, 4, 3)
    YM   = torch.stack(YM).permute(0, 1, 2, 4, 3)

    # Context tensors keep their native sequence layouts.
    D      = torch.stack(D)
    T      = torch.stack(T)
    H      = torch.stack(H)       # (B, 40, F, n_vars)
    H_mask = torch.stack(H_mask)  # (B, 40, F, n_vars)

    return X, Y, M, D, T, YM, H, H_mask


@torch.no_grad()
def predict_dataset(
    loader,
    model,
    *,
    input_scaler: Scaler,
    output_scaler: Scaler,
    target_mode: str,
    use_date: bool,
    use_tide: bool,
    use_history: bool = False
):
    """
    Return ML-corrected spectra and raw ECMWF spectra for a full loader.
    """
    device = next(model.parameters()).device
    model.eval()
    preds, raws = [], []

    for X, Y, M, D, T, YM, H, H_mask in loader:
        X_raw  = X.to(device, non_blocking=True)
        YM_raw = YM.to(device, non_blocking=True)

        X_in = input_scaler.transform(X_raw)
        YM_in = input_scaler.transform(YM_raw)

        fwd_kwargs = {}
        if use_date:
            fwd_kwargs["date_encoding"] = D.to(device, non_blocking=True)
        if use_tide:
            fwd_kwargs["tide"] = T.to(device, non_blocking=True)
        if use_history:
            H_raw = H.to(device, non_blocking=True)
            if target_mode == "direct":
                fwd_kwargs["history"] = output_scaler.transform(H_raw)
            else:
                fwd_kwargs["history"] = input_scaler.transform(H_raw)
            fwd_kwargs["history_mask"] = H_mask.to(device, non_blocking=True)

        if model.input_mode == 'spectra':
            x_s, x_c = X_in, None
        elif model.input_mode == 'coeffs':
            x_s, x_c = None, YM_in
        else:
            x_s, x_c = X_in, YM_in

        out_scaled = model(x_spec=x_s, x_coeff=x_c, **fwd_kwargs)
        out_physical = output_scaler.inverse_transform(out_scaled)

        out_physical = out_physical.squeeze(-1).squeeze(1)
        YM_physical  = YM_raw.squeeze(-1).squeeze(1)

        if target_mode == "direct":
            Y_corr = out_physical
        elif target_mode == "residual":
            Y_corr = YM_physical + out_physical
        else:
            raise ValueError(f"Unknown target_mode={target_mode}")

        Y_corr.clamp_min_(0.0)

        preds.append(Y_corr.cpu())
        raws.append(YM_physical.cpu())

    return torch.cat(preds, 0), torch.cat(raws, 0)


class MLCorrectedDataset(Dataset):
    """
    Wrap an `ECMWFDataset` and replace the raw forecast with ML output.

    This lets the existing table-building utilities evaluate the corrected
    spectrum with the same observation metadata and masks as the baseline.
    """
    def __init__(self, base_ds, y_pred_40F: torch.Tensor):
        self.base = base_ds
        if y_pred_40F.ndim != 3:
            raise AssertionError(f"y_pred_40F must be (N,40,F), got {tuple(y_pred_40F.shape)}")
        if y_pred_40F.shape[0] != len(base_ds):
            raise AssertionError("Prediction dataset size mismatch")
        self.y_pred = y_pred_40F.to(torch.float32).cpu()

    def __getattr__(self, name): return getattr(self.base, name)
    def __len__(self): return len(self.base)

    def __getitem__(self, idx):
        # Preserve all metadata; replace only the model spectrum used downstream.
        X, Y, mask_Y, date_enc, tide, Y_model_raw, H, H_mask = self.base[idx]
        
        yhat_physical = self.y_pred[idx].unsqueeze(0).unsqueeze(2)
        yhat_physical = yhat_physical.to(dtype=Y_model_raw.dtype)

        Y_model_corr = yhat_physical
            
        return X, Y, mask_Y, date_enc, tide, Y_model_corr, H, H_mask
