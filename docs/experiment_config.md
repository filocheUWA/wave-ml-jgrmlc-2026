# Experiment Configuration

This document records the manuscript-style experiment assumptions encoded in the
current scripts.

## Dates

Default split in `src/scripts/train.py`:

| Split | Date range |
| --- | --- |
| Train | `20200101_00` to `20220621_00` |
| Validation | `20220701_00` to `20221222_00` |
| Test | `20230101_00` to `20231231_12` |

The manuscript describes chronological splitting with buffer periods to avoid
temporal leakage between training, validation, and test windows.

## Fixed Parameters

Default fixed parameters in `train.py` include:

```text
history_vars = ["E"]
num_workers = 8
weight_decay = 1e-2
grad_clip_norm = 1.0
loss_scale = 5.0
n_spec_ch = 1
n_coeff_ch = 1
n_output_ch = 1
warmup_epochs = 2
min_lr_factor = 0.1
```

## Main CLI Options

| Option | Meaning |
| --- | --- |
| `--model_size` | `small`, `base`, or `large` SpecX channel preset |
| `--input_mode` | `spectra`, `coeffs`, or `fused` |
| `--use_xattn` | Enable optional cross-attention |
| `--use_history` | Include 5-day buoy-history context |
| `--use_date` | Include date/lead-time context |
| `--use_tide` | Include tide context |
| `--target_mode` | `direct` or `residual` |
| `--loss_type` | `mse`, `mae`, `binweighted_mse`, `binweighted_mae` |
| `--scale_input` | Apply input scaler |
| `--scale_output` | Apply target/output scaler |
| `--skip_plots` | Train/evaluate without post-training figures |

## Hyperparameter Search Space

`src/scripts/tune.py` explores:

- model size: `small`, `base`, `large`
- cross-attention: enabled/disabled
- input mode: `spectra`, `coeffs`
- input scaling: enabled/disabled
- context: history/date/tide enabled/disabled
- target mode: `direct`, `residual`
- output scaling: enabled/disabled
- loss: MSE, MAE, bin-weighted MSE, bin-weighted MAE
- learning rate: `1e-3`, `5e-4`, `1e-4`
- batch size: `8`, `16`, `32`, `64`

The manuscript-selected configuration favors 2D spectral input, date context,
tide context, no buoy-history context, no cross-attention, direct prediction, and
MAE-style training.

## Output Organization

Each experiment uses a hashed run id derived from its configuration and writes to:

```text
OUTPUT_ROOT/
  Exp_XXX/
    config.json
    weights/
      <run_id>.pt
    loss_data/
      history.csv
      test_err_df.csv
    figures/
```

`experiment_log.csv` at `OUTPUT_ROOT` records experiment metadata across runs.
