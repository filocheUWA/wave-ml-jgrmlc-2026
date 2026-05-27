"""Training and evaluation loop for the spectral post-processing model."""

import torch
from torch.nn.utils import clip_grad_norm_
from src.training.scalers import Scaler

# --- batch preparation --------------------------------------------------------
def build_context(batch, device, *, use_date: bool, use_tide: bool, use_history: bool = False):
    """
    Move a batch to the target device and assemble optional context inputs.
    """
    X, Y, M, D, T, YM, H, HM = batch
    
    # Keep entry tensors contiguous for stable convolution kernels on GPU backends.
    X  = X.to(device, non_blocking=True).contiguous()
    Y  = Y.to(device, non_blocking=True).contiguous()
    M  = M.to(device, non_blocking=True).float().contiguous()
    YM = YM.to(device, non_blocking=True).contiguous()

    fwd_kwargs = {}
    if use_date:
        fwd_kwargs["date_encoding"] = D.to(device, non_blocking=True).contiguous()
    if use_tide:
        fwd_kwargs["tide"] = T.to(device, non_blocking=True).contiguous()
    if use_history:
        fwd_kwargs["history"] = H.to(device, non_blocking=True).contiguous()
        fwd_kwargs["history_mask"] = HM.to(device, non_blocking=True).contiguous()
        
    return X, Y, M, YM, fwd_kwargs


def compute_loss(criterion, y_pred, y_tgt, mask):
    """Delegate masked spectral loss computation to the selected criterion."""
    return criterion(y_pred, y_tgt, mask)


def apply_update(model, optimizer, loss, grad_clip_norm):
    """Apply one optimizer update with optional gradient clipping."""
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    if grad_clip_norm is not None:
        clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
    optimizer.step()


# --- epoch runner -------------------------------------------------------------
class EpochRunner:
    def __init__(
        self,
        model,
        optimizer,
        criterion,
        input_scaler: Scaler,
        output_scaler: Scaler,
        target_mode: str,
        *,
        loss_scale_residual: float = 5.0,
        grad_clip_norm: float | None = 1.0,
        use_date: bool = False,
        use_tide: bool = False,
        use_history: bool = False,
    ):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.input_scaler = input_scaler
        self.output_scaler = output_scaler
        self.target_mode = target_mode
        self.loss_scale_residual = loss_scale_residual
        self.grad_clip_norm = grad_clip_norm
        self.use_date = use_date
        self.use_tide = use_tide
        self.use_history = use_history

    @property
    def device(self):
        return next(self.model.parameters()).device

    def run_epoch(self, loader, *, mode: str) -> dict:
        is_train = (mode == "train")
        self.model.train(is_train)

        loss_sum, n_batches, n_samples = 0.0, 0, 0

        with torch.set_grad_enabled(is_train):
            for batch in loader:
                # Prepare forecast tensors, observation target, mask, and context.
                X, Y, M, YM, fwd_kwargs = build_context(
                    batch, self.device, 
                    use_date=self.use_date, 
                    use_tide=self.use_tide,
                    use_history=self.use_history
                )

                # Scale model inputs and restore contiguity after tensor transforms.
                x_s_scaled = self.input_scaler.transform(X).contiguous()
                x_c_scaled = self.input_scaler.transform(YM).contiguous()

                if self.use_history and "history" in fwd_kwargs:
                    if self.target_mode == "direct":
                        fwd_kwargs["history"] = self.output_scaler.transform(fwd_kwargs["history"]).contiguous()
                    else:
                        fwd_kwargs["history"] = self.input_scaler.transform(fwd_kwargs["history"]).contiguous()

                if self.model.input_mode == 'spectra':
                    x_s, x_c = x_s_scaled, None
                elif self.model.input_mode == 'coeffs':
                    x_s, x_c = None, x_c_scaled
                else:
                    x_s, x_c = x_s_scaled, x_c_scaled

                # Predict the scaled target spectrum or residual correction.
                model_pred_scaled = self.model(x_spec=x_s, x_coeff=x_c, **fwd_kwargs)

                # Match the supervised target to the selected regression task.
                if self.target_mode == "direct":
                    raw_target = Y
                elif self.target_mode == "residual":
                    raw_target = Y - YM
                else:
                    raise ValueError(f"Unknown target_mode={self.target_mode!r}")

                target_scaled = self.output_scaler.transform(raw_target)

                # Loss is always reported before residual-mode training rescaling.
                base_loss = compute_loss(self.criterion, model_pred_scaled, target_scaled, M)

                loss = (
                    base_loss * self.loss_scale_residual
                    if (is_train and self.target_mode == "residual")
                    else base_loss
                )

                if is_train:
                    apply_update(self.model, self.optimizer, loss, self.grad_clip_norm)

                loss_sum += float(base_loss.item())
                n_batches += 1
                n_samples += X.size(0)

        return {
            "loss_mean": loss_sum / max(n_batches, 1),
            "n_batches": n_batches,
            "n_samples": n_samples,
            "mode": mode,
            "target_mode": self.target_mode,
        }
