# Scientific Context

This repository implements the machine-learning pipeline associated with a
manuscript on site-specific post-processing of spectral wave forecasts using
buoy measurements.

## Problem

Operational ECMWF wave forecasts provide directional spectral wave energy, but
at the Australian North West Shelf study site they exhibit local, structured
errors. The manuscript focuses especially on premature swell-arrival behavior
and errors in low-frequency swell energy.

The repository formulates post-processing as supervised regression:

```text
ECMWF forecast spectrum + local context -> buoy-like 1D energy spectrum
```

The corrected spectrum is then integrated into bulk parameters such as
significant wave height (`Hs`), peak period (`Tp`), and mean zero-crossing period
(`Tm02`).

## Paper-To-Code Map

| Manuscript concept | Repository location |
| --- | --- |
| ECMWF/buoy/tide data loading | `src/utils/dataset.py` |
| Batch layout and prediction helpers | `src/training/data_management.py` |
| SpecX model | `src/architecture/model_specx.py` |
| Direction-aware convolutions | `src/architecture/conv_theta.py` |
| Context encoders for date, tide, and history | `src/architecture/context.py` |
| FiLM conditioning | `src/architecture/film.py` |
| Optional cross-attention | `src/architecture/xattn.py` |
| Training loop | `src/training/engine.py` |
| Losses and Hs RMSE metrics | `src/training/losses.py`, `src/training/metrics.py` |
| Single-run orchestration | `src/scripts/train.py` |
| Hyperparameter search | `src/scripts/tune.py` |
| Ensemble workflow | `src/scripts/train_ensemble.py` |
| Manuscript-style plots | `src/plotting/` |

## Main Scientific Findings

- The selected model uses 2D directional spectral input, date context, and tide
  context.
- Date and tide context improve validation skill.
- Five-day buoy history was tested but was not selected by the hyperparameter
  search.
- The strongest improvements occur in the swell band (`f < 0.1 Hz`).
- Swell `Hs`, `Tp`, and `Tm02` improve across lead times.
- Wind-sea energy correction is mixed and can degrade `Hs`.
- PCA diagnostics suggest the model reduces structured time-frequency swell
  error modes, including arrival-time-related modes.
- Ensemble averaging improves deterministic predictions, but rank histograms
  show the ensemble is underdispersive and not calibrated uncertainty.

## Frequency Partitions

The manuscript and plotting code commonly use:

| Partition | Frequency range |
| --- | --- |
| Total | approximately `0.034-0.51 Hz` |
| Swell | `0.034-0.10 Hz` |
| Wind sea | `0.10-0.51 Hz` |

The `0.10 Hz` cut-off is based on the observed local spectral climatology.

## Interpretation Caution

The model is a local statistical correction. It should not be interpreted as a
general wave model or as a globally transferable ECMWF correction without new
site data and validation.
