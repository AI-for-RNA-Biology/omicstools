#
# SPDX-FileCopyrightText: Copyright © 2025 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileCopyrightText: Copyright © 2026 University of Bern
#
# SPDX-FileContributor: Lisa Fournier <lisa.fournier@idiap.ch>
# SPDX-FileContributor: Lisa Fournier <lisa.fournier@unibe.ch>
#
# SPDX-License-Identifier: GPL-3.0-only
#
from graspologic.cluster.autogmm import AutoGMMCluster
from scipy.stats import norm
import seaborn as sns
import matplotlib.pyplot as plt

def plot_GMMclusters(X, min_components=1, max_components=2, xmin=0, xmax=20):
    """Fit a Gaussian mixture (graspologic AutoGMMCluster, `min_components` to `max_components` components) to `X` and
    plot its histogram (probability) with the fitted Gaussians.

    Args:
        X: values (2D array, one column).
        min_components: minimal number of components. Defaults to 1.
        max_components: maximal number of components. Defaults to 2.
        xmin: lower x limit of the Gaussians. Defaults to 0.
        xmax: upper x limit of the Gaussians. Defaults to 20.
    """
    autogmm_ = AutoGMMCluster(min_components=min_components, max_components=max_components)
    c_hat_autogmm = autogmm_.fit_predict(X)
    sns.histplot(X, stat='probability')
    plot_gaussian(autogmm_.model_.means_, autogmm_.model_.covariances_, xmin, xmax)
    plt.ylim(0, 0.2)

def plot_gaussian(x_means, x_vars, xmin=0, xmax=20):
    """Plot the normal densities of means `x_means` and variances `x_vars` on [xmin, xmax] (black lines)."""
    x = np.linspace(start=xmin, stop=xmax, num=200)
    for m, v in zip(x_means, x_vars):
        y = norm(m, np.sqrt(v)).pdf(x)
        plt.plot(x, y.T, linewidth=3, color='black')