<!--
#
# SPDX-FileCopyrightText: Copyright © 2025 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileCopyrightText: Copyright © 2026 University of Bern
#
# SPDX-FileContributor: Lisa Fournier <lisa.fournier@idiap.ch>
# SPDX-FileContributor: Lisa Fournier <lisa.fournier@unibe.ch>
#
# SPDX-License-Identifier: GPL-3.0-only
#
-->
# Project

This project gathers generic tools for omics data analysis.



## Installation

To install this library in another Python project, execute simply:

```bash
pip install git+https://github.com/AI-for-RNA-Biology/omicstools.git
```
or
```bash
git clone https://github.com/AI-for-RNA-Biology/omicstools.git
cd omicstools
pip install .
```

To use `omicstools.GMM`, install the optional `gmm` extra (`pip install ".[gmm]"`), which brings `graspologic`.

`omicstools.OmicsDataset` also requires the `scanalysis` package ([AI-for-RNA-Biology/scanalysis](https://github.com/AI-for-RNA-Biology/scanalysis), not on PyPI), installed with `omicstools`.

`omicstools.LMM_helper` (and the LMM methods of `OmicsDataset`) call R through `rpy2`: R must be available with the `lme4` package installed.

## Modules
- `OmicsDataset`: container for an omics matrix and its metadata (SVD / PCA, correlation of components with covariates, linear mixed models per component, clustering, plots).
- `LMM_helper`: linear mixed models through R `lme4` (`lmer_general`, `lmer_per_PC_re`, ...).
- `GMM`: Gaussian mixture model plotting helpers.
- `helper_plot`: generic plotting helpers (`plot_boxplot_significance`, `boxplot_pca_significance`).
