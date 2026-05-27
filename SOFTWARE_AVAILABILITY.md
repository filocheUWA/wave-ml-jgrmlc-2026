# Software Availability

This repository contains the software used to train and evaluate the
site-specific spectral wave forecast post-processing model described in the
associated manuscript.

The main software components are:

```text
src/architecture/      SpecX architecture and supporting neural-network blocks.
src/training/          Data handling, losses, metrics, optimization, and epoch runner.
src/utils/             Dataset loading, interpolation, scalers, and table builders.
src/plotting/          Manuscript-style diagnostics and result visualization.
src/scripts/           Training, ensemble training, and hyperparameter-search entry points.
```

Draft statement:

The software used in this study is available at [GitHub URL] and preserved at
[software DOI] under the license provided in this repository.  The archived
release contains the source code, configuration files, and retained diagnostic
outputs required to inspect the model architecture and reproduce the manuscript
figures, subject to the data access conditions described in
`DATA_AVAILABILITY.md`.

Items to complete before release:

- Record the final GitHub URL.
- Archive a versioned release and record the DOI.
- Update `CITATION.cff` if the title, author list, or DOI changes.
- Add the software citation to the manuscript References section.
