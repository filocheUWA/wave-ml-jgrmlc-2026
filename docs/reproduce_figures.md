# Reproducing Figures

The plotting workflow is driven by the post-training phase of
`src/scripts/train.py` and `src/scripts/train_ensemble.py`. Most figures are
created from a cached error table produced after model inference on the test set.

## Standard Post-Training Analysis

Run training without `--skip_plots`:

```bash
python3 -u src/scripts/train.py
```

or, for the ensemble workflow:

```bash
python3 -u src/scripts/train_ensemble.py
```

Outputs are written under:

```text
OUTPUT_ROOT/
  Exp_XXX/
    figures/
    loss_data/
```

The key cached table is:

```text
Exp_XXX/loss_data/test_err_df.csv
```

If this file exists, plotting can reuse it without rerunning inference.

## Plot Function Map

| Manuscript figure type | Function / module |
| --- | --- |
| Forecast dashboards and GIFs | `src/plotting/dashboard.py` |
| Summary boxplots | `src/plotting/boxplots.py` |
| Bias/std frequency-lead heatmaps | `src/plotting/heatmap.py` |
| Hs MSE decomposition | `src/plotting/mse_decomposition.py` |
| Empirical QQ plots | `src/plotting/qq.py` |
| PCA structure comparison | `src/plotting/pca_analysis.py::plot_pca_structure` |
| Swell arrival-time PCA mode analysis | `src/plotting/pca_analysis.py::plot_pca_mode_swell` |
| Ensemble rank histograms / uncertainty diagnostics | `src/plotting/reliability.py` |

## Dashboard Figures

The dashboard plots compare:

- observations
- raw ECMWF baseline
- ML or ML-Ensemble correction

The three spectrum panels use lead times:

```text
+3 h, +60 h, +120 h
```

The same lead times are marked on the time-series panels with vertical guide
lines.

## PCA Figures

`plot_pca_structure`:

1. Builds time-frequency error matrices for ECMWF and ML.
2. Fits PCA on ECMWF baseline errors.
3. Projects ML errors onto the same ECMWF error modes.
4. Compares modal error energy in a scree plot.
5. Displays the leading PCA mode shapes as frequency-lead heatmaps.

Interpretation:

- Green downward annotations indicate reduced modal error energy.
- Red upward annotations indicate increased modal error energy.
- The heatmaps show ECMWF error eigenvectors, not ML predictions.

## Common Failure Modes

- If a plot uses stale data, delete the relevant cached CSV in `loss_data/`.
- If figures look cramped after layout edits, regenerate both PNG and PDF before
  judging; `save_figure(..., bbox_inches="tight")` can change final whitespace.
- If imports fail on a login node, check that the same environment used for
  training is active for plotting.
