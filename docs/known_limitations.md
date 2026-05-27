# Known Limitations

This file documents scientific and engineering limitations that a new
contributor should understand before extending the project.

## Scientific Limitations

### Short Record Length

The model is trained and evaluated on a limited multi-year record. This restricts
coverage of rare extremes, interannual climate variability, and changing
seasonal regimes.

### Non-Stationary Forecast System

ECMWF operational cycles change over time. Changes in model physics, forcing,
assimilation, or resolution can alter the baseline forecast-error distribution
that the ML correction is trying to learn.

### Site Specificity

The current model is specialized to one Australian North West Shelf site.
Applying it elsewhere requires new site data, validation, and likely retraining.

### Wind-Sea Skill

The strongest improvements are in the swell band. Wind-sea energy correction is
less reliable and can degrade `Hs` in the wind-sea partition.

### Buoy History

The architecture supports recent buoy-history context, but the manuscript
hyperparameter search did not select it. This suggests the current history
encoder or conditioning strategy does not yet extract useful persistence
information.

### Uncertainty

The ensemble spread is underdispersive. It should be understood as a rough
epistemic spread from random initialization, not a calibrated probabilistic
forecast.

## Engineering Limitations

### Hardcoded HPC Defaults

The scripts still contain system-specific default paths and SLURM assumptions.
Use `DATA_ROOT` and `OUTPUT_ROOT` environment variables when running elsewhere.

### Script-Based Workflow

The project is not yet packaged as an installable Python module. Entry points are
plain scripts under `src/scripts/`.

### Limited Automated Tests

The repository currently relies on experiment runs and plotting outputs rather
than unit tests or CI.

### Private Data

The buoy observations are commercially restricted. Full scientific
reproducibility requires access to the processed data or an equivalent local
dataset with matching schema.

### Baseline Module Cleanliness

`src/architecture/model_baseline.py` should be reviewed before relying on the
baseline models. The main manuscript workflow uses `SpecX`.
