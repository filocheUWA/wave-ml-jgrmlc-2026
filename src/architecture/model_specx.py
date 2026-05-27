"""SpecX: context-conditioned encoder-decoder for spectral forecast correction."""
from __future__ import annotations
import torch
import torch.nn as nn

from .conv_theta import (
    theta_mixing_conv3x3,
    no_theta_conv3x3,
    conv3d_1x1,
    batch_norm3d,
    DirectionDownsampleAA,
    DirectionSoftCollapse,
    init_xavier_conv,
)

from .gates import TimeSeqGateBlock, ResidualGateStack, TimeSeqGateBlockIO
from .context import DateEncoderFF, TideEncoderResNet1D, HistoryEncoder, ContextFusion
from .film import FiLMT, FiLMF, apply_film
from .xattn import CrossAttentionBlock


# ───────────────────────── filters presets ─────────────────────────
def _filters_from_size(model_size: str | None, filters_tuple: tuple[int, int, int] | None):
    if filters_tuple is not None:
        return filters_tuple
    if model_size is None or model_size == "base":
        return (12, 16, 24)
    if model_size == "small":
        return (8, 12, 16)
    if model_size == "large":
        return (16, 24, 32)
    raise ValueError("model_size must be one of {'small','base','large'} or provide filters=...")


class SpecX(nn.Module):
    """
    Spectral encoder-decoder with optional date, tide, and buoy-history context.
    """
    def __init__(self, input_size=(40, 29, 36), *,
                 input_mode: str = "fused",
                 n_spec_ch: int = 1, n_coeff_ch: int = 1, n_output_ch: int = 1,
                 model_size: str | None = "base", filters: tuple[int, int, int] | None = None,
                 expand_decoder: bool = False, dec_channels: int | None = None,
                 activation: str = "ReLU", theta_temperature: float = 1.0,
                 ctx_width: int | None = None, use_film_F: bool = True,
                 
                 # Flags
                 use_date_default: bool = False, 
                 use_tide_default: bool = False, 
                 use_buoy_history_default: bool = False,
                 use_xattn_default: bool = False,
                 
                 # New Params
                 n_hist_vars: int = 1,
                 
                 xattn_heads: int = 2, xattn_dropout: float = 0.1):
        super().__init__()
        assert input_mode in {"fused", "spectra", "coeffs"}
        self.input_mode = input_mode
        T, Fbins, Th = input_size
        self.T, self.Fbins, self.ThetaBins = T, Fbins, Th

        # filters & widths
        C0, C1, C2 = _filters_from_size(model_size, filters)
        self.C2_latent = C2

        if expand_decoder:
            C_dec = dec_channels if dec_channels is not None else max(32, C2)
        else:
            C_dec = C2
        self.C_dec = C_dec

        act = nn.ReLU() if activation == "ReLU" else nn.SiLU()

        # ────────────── Encoder (2D spectra) ──────────────
        self.bn_in_X  = batch_norm3d(n_spec_ch)
        self.conv_in_X = theta_mixing_conv3x3(n_spec_ch, C0)
        self.down1_X   = DirectionDownsampleAA(C0, C1, pool=True,  act=act)
        self.down2_X   = DirectionDownsampleAA(C1, C2, pool=True,  act=act)
        Th9 = Th // 4
        self.enc_gate_pre_X = TimeSeqGateBlock(C2, (T, Fbins, Th9), ktheta=3)

        # ────────────── Encoder (1D coeffs) ──────────────
        self.bn_in_C   = batch_norm3d(n_coeff_ch)
        self.conv_in_C = no_theta_conv3x3(n_coeff_ch, C0)
        self.block1_C  = TimeSeqGateBlockIO(C0, C1, (T, Fbins, 1), ktheta=1)
        self.block2_C  = TimeSeqGateBlockIO(C1, C2, (T, Fbins, 1), ktheta=1)
        self.enc_gate_pre_C = TimeSeqGateBlock(C2, (T, Fbins, 1), ktheta=1)

        # ────────────── Shared collapse ──────────────
        self.theta_collapse = DirectionSoftCollapse(C2, temperature=theta_temperature)
        self.expand_after_latent = None
        if C_dec != C2:
            self.expand_after_latent = conv3d_1x1(C2, C_dec)

        # ────────────── Context encoders & fusion ──────────────
        ctx_dim_default = C_dec if ctx_width is None else ctx_width
        self.ctx_width = ctx_dim_default
        
        self.date_enc = DateEncoderFF(d_ctx=ctx_dim_default, n_freqs=8)
        self.tide_enc = TideEncoderResNet1D(d_ctx=ctx_dim_default, T=T)
        self.hist_enc = HistoryEncoder(d_ctx=ctx_dim_default, F_bins=Fbins, n_vars=n_hist_vars)
        
        self.ctx_fuse = ContextFusion(d_ctx=ctx_dim_default)

        # ────────────── FiLM Generators ──────────────
        self.film_T_enc_c1 = FiLMT(d_ctx=ctx_dim_default, C=C1)
        self.film_T_enc_c2 = FiLMT(d_ctx=ctx_dim_default, C=C2)
        self.film_T_dec    = FiLMT(d_ctx=ctx_dim_default, C=C_dec)
        if use_film_F:
            self.film_F_enc_c1 = FiLMF(d_global=ctx_dim_default, C=C1, Fbins=Fbins, d_fpos=16)
            self.film_F_enc_c2 = FiLMF(d_global=ctx_dim_default, C=C2, Fbins=Fbins, d_fpos=16)
            self.film_F_dec    = FiLMF(d_global=ctx_dim_default, C=C_dec, Fbins=Fbins, d_fpos=16)
        else:
            self.film_F_enc_c1, self.film_F_enc_c2, self.film_F_dec = None, None, None
        self.use_film_F = use_film_F

        self.xattn = CrossAttentionBlock(C=C_dec, ctx_dim=ctx_dim_default, n_heads=xattn_heads, dropout=xattn_dropout)

        # ────────────── Decoder ──────────────
        self.dec_gate1 = ResidualGateStack(1, C_dec, (T, Fbins, 1), ktheta=1)
        self.dec_gate2 = ResidualGateStack(1, C_dec, (T, Fbins, 1), ktheta=1)
        self.dec_refine = TimeSeqGateBlock(C_dec, (T, Fbins, 1), ktheta=1)

        self.proj_to_C2 = None
        if C_dec != C2:
            self.proj_to_C2 = conv3d_1x1(C_dec, C2)

        self.conv_out = conv3d_1x1(C2, n_output_ch)

        # defaults
        self.use_date_default = use_date_default
        self.use_tide_default = use_tide_default
        self.use_buoy_history_default = use_buoy_history_default
        self.use_xattn_default = use_xattn_default

        self.apply(init_xavier_conv)

    # ─────────────────────────── forward ───────────────────────────
    def forward(self, x_spec=None, x_coeff=None, *,
                date_encoding=None,      
                tide=None,
                history=None, history_mask=None,
                
                use_date: bool | None = None,
                use_tide: bool | None = None,
                use_buoy_history: bool | None = None,
                use_xattn: bool | None = None,
                verbose: bool = False):
        
        def p(tag, t, tail=""):
            if verbose:
                print(f"{tag:<24} {tuple(t.shape)}{tail}")

        if self.input_mode in {"spectra", "fused"}: assert x_spec is not None
        if self.input_mode in {"coeffs", "fused"}: assert x_coeff is not None

        # Resolve flags
        if use_date is None:  use_date  = self.use_date_default
        if use_tide is None:  use_tide  = self.use_tide_default
        if use_buoy_history is None: use_buoy_history = self.use_buoy_history_default
        if use_xattn is None: use_xattn = self.use_xattn_default

        # Re-assert memory contiguity for all entry-point tensors to ensure ROCm kernel safety
        if x_spec is not None: x_spec = x_spec.contiguous()
        if x_coeff is not None: x_coeff = x_coeff.contiguous()

        # ────────────── Context Encoding ──────────────
        perT_date = self.date_enc(date_encoding) if (use_date and (date_encoding is not None)) else None
        perT_tide = self.tide_enc(tide)          if (use_tide and (tide is not None))          else None
        
        # Encode recent buoy spectra when history context is enabled.
        perT_hist = None
        if use_buoy_history and history is not None:
            if history_mask is None:
                history_mask = torch.ones_like(history)
            perT_hist = self.hist_enc(history, history_mask)
            if verbose: p("[History Encoded]", perT_hist)

        # Fuse 3 potential tokens
        perT_ctx, global_ctx = self.ctx_fuse(perT_date, perT_tide, perT_hist)

        # ────────────── FiLM Generation ──────────────
        gt1, bt1, gt2, bt2, gtd, btd = [None] * 6
        gf1, bf1, gf2, bf2, gfd, bfd = [None] * 6
        if perT_ctx is not None:
            gt1, bt1 = self.film_T_enc_c1(perT_ctx)
            gt2, bt2 = self.film_T_enc_c2(perT_ctx)
            gtd, btd = self.film_T_dec(perT_ctx)
        if self.use_film_F and global_ctx is not None:
            gf1, bf1 = self.film_F_enc_c1(global_ctx)
            gf2, bf2 = self.film_F_enc_c2(global_ctx)
            gfd, bfd = self.film_F_dec(global_ctx)

        # ────────────── Encoder Paths ──────────────
        s_path, c_path = None, None
        if self.input_mode in {"spectra", "fused"}:
            p("[input_spec]", x_spec)
            s_path = self.bn_in_X(x_spec)
            s_path = self.conv_in_X(s_path)
            s_path = self.down1_X(s_path, gt1, bt1, gf1, bf1)
            s_path = self.down2_X(s_path, gt2, bt2, gf2, bf2)
            s_path = self.enc_gate_pre_X(s_path)

        if self.input_mode in {"coeffs", "fused"}:
            # The unsqueeze operation creates a non-contiguous view. 
            # We must fix it immediately before it reaches the convolution layers.
            if x_coeff.dim() == 4: x_coeff = x_coeff.unsqueeze(-1).contiguous()
            p("[input_coeff]", x_coeff)
            c_path = self.bn_in_C(x_coeff)
            c_path = self.conv_in_C(c_path)
            c_path = self.block1_C(c_path, gt1, bt1, gf1, bf1)
            c_path = self.block2_C(c_path, gt2, bt2, gf2, bf2)
            c_path = self.enc_gate_pre_C(c_path)

        # ────────────── Fusion / Path Selection ──────────────
        if self.input_mode == 'fused':
            s_path_collapsed = self.theta_collapse(s_path)
            x = s_path_collapsed + c_path
            p("[LATENT z=fused]", x)
        elif self.input_mode == 'spectra':
            x = self.theta_collapse(s_path)
            p("[LATENT z=spectra]", x)
        else: 
            x = c_path
            p("[LATENT z=coeffs]", x)

        if self.expand_after_latent is not None:
            x = self.expand_after_latent(x)

        # ────────────── Cross-Attention ──────────────
        if use_xattn and (perT_ctx is not None):
            x = self.xattn(x, perT_ctx); p("[xattn latent]", x)
        elif verbose and use_xattn:
            print("[xattn latent]           skipped (no context)")

        x = apply_film(x, gtd, btd, gfd, bfd); p("[FiLM_TF @ dec1]", x)

        # ────────────── Decoder ──────────────
        x = self.dec_gate1(x)
        x = self.dec_gate2(x)
        x = self.dec_refine(x)

        if self.proj_to_C2 is not None:
            x = self.proj_to_C2(x)

        x = self.conv_out(x); p("[conv_out → n_out]", x)
        return x
