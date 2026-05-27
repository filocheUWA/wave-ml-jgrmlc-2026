# Data Schema

This document describes the data layout expected by the current scripts. It is a
developer-facing map of the assumptions encoded in `src/utils/dataset.py` and
`src/training/data_management.py`.

## Directory Layout

Set `DATA_ROOT` to a processed data directory with:

```text
DATA_ROOT/
  ecmwf_forecast_nc/
    YYYYMMDD_HH.nc
  buoy_nc/
    YYYYMMDD_HH.nc
  <tide_directory>/
    processed_tides.nc
```

The scripts default to system-specific scratch paths when `DATA_ROOT` is not set.

## Required NetCDF Variables

| Source | Variable | Meaning |
| --- | --- | --- |
| ECMWF forecast files | `efth` | Directional wave energy spectrum |
| Buoy files | `E` | 1D wave energy spectrum |
| Tide file | `etaMSL` | Sea-level anomaly / tidal water level |

The current training scripts use `E` as the target variable. Other variables may
exist in the processed files, but they are not required by the default pipeline.

## Time Convention

- Forecast issuances are named `YYYYMMDD_HH.nc`.
- Forecast horizon is 5 days.
- Lead-time step is 3 hours.
- Each sample contains 40 lead times: `+3 h` through `+120 h`.
- Tide context covers the same 120-hour forecast horizon at 15-minute
  resolution, giving 480 samples.
- Optional buoy history covers the 5 days before issuance at 3-hour resolution,
  giving 40 history steps.

## Frequency Convention

The default ECMWF model frequency grid is generated as:

```text
f0 = 0.0345
freqs = f0 * logspace(0, 35, steps=36, base=1.1)
```

The default selected band is approximately:

```text
0.034 Hz <= f <= 0.51 Hz
```

This usually yields 29 selected frequency bins in the manuscript experiments.

## Dataset Sample

`ECMWFDataset.__getitem__` returns:

```text
X, Y, mask_Y, date_encoding, tide, Y_model, H, H_mask
```

| Name | Meaning | Shape before collation |
| --- | --- | --- |
| `X` | ECMWF directional spectrum | `(1, 40, 36, F)` |
| `Y` | Buoy target spectrum | `(V, 40, 1, F)` |
| `mask_Y` | Valid-observation mask | `(V, 40, 1, F)` |
| `date_encoding` | Lead/date features | `(6, 40)` |
| `tide` | Tide context | `(1, 480)` |
| `Y_model` | ECMWF 1D spectrum from directional integration | `(1, 40, 1, F)` |
| `H` | Optional buoy-history spectra | `(40, F, n_history_vars)` |
| `H_mask` | Valid-history mask | `(40, F, n_history_vars)` |

`V` is the number of target variables; default training uses `V=1` for `E`.

## Batch Layout

The custom `collate` function converts spectra to model layout:

```text
(B, C, T, F, Theta)
```

Important batch tensors:

| Name | Shape after collation |
| --- | --- |
| `X` | `(B, 1, 40, F, 36)` |
| `Y` | `(B, 1, 40, F, 1)` |
| `M` | `(B, 1, 40, F, 1)` |
| `YM` | `(B, 1, 40, F, 1)` |
| `D` | `(B, 6, 40)` |
| `T` | `(B, 1, 480)` |
| `H` | `(B, 40, F, n_history_vars)` |
| `H_mask` | `(B, 40, F, n_history_vars)` |

## Target Modes

The training runner supports:

- `direct`: model predicts the physical target spectrum.
- `residual`: model predicts `Y - YM`, later added back to the ECMWF spectrum.

The manuscript-selected workflow uses direct spectral prediction.
