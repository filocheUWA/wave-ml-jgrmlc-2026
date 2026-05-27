# Data Availability

This file provides draft Open Research language for the data objects associated
with the manuscript.  Final repository identifiers and access conditions should
be inserted before public release.

The analysis uses preprocessed NetCDF files containing ECMWF directional wave
forecasts, local buoy-derived one-dimensional spectra, and tide information.
The expected schema is documented in `docs/data_schema.md`.  The source buoy
observations are commercially restricted and are not redistributed in this
repository unless permission is granted by the data provider.  The ECMWF
forecast data should be accessed through the applicable ECMWF archive and terms
of use.

Draft statement:

The processed data supporting this study are available from [repository or DOI]
under [access conditions].  The source buoy observations were provided for this
study by [data provider] and are subject to [access restriction].  ECMWF
forecast data are available from ECMWF through [archive or access route] under
the applicable ECMWF terms of use.  The software used to construct the training
dataset, train the post-processing model, and generate the manuscript
diagnostics is archived at [software DOI] and developed at [GitHub URL].

Items to complete before release:

- Confirm whether the processed buoy-derived NetCDF files can be redistributed.
- Record the final data repository and DOI, if applicable.
- Record the ECMWF access route and formal data citation.
- Add formal data citations to the manuscript References section.
