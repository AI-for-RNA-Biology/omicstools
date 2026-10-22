#
# SPDX-FileCopyrightText: Copyright © 2025 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileCopyrightText: Copyright © 2026 University of Bern
#
# SPDX-FileContributor: Lisa Fournier <lisa.fournier@idiap.ch>
# SPDX-FileContributor: Lisa Fournier <lisa.fournier@unibe.ch>
#
# SPDX-License-Identifier: GPL-3.0-only
#
import seaborn as sns
from itertools import combinations
from scipy.stats import ttest_ind, mannwhitneyu, wilcoxon, levene
from statannotations.Annotator import Annotator
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


from itertools import combinations
from statannotations.Annotator import Annotator

def plot_boxplot_significance(df, x, y, log_scale=False, stripplot=True, palette=None, fontsize=12, showfliers=True,
                              stat='t-test', pairs=None, p_values=None, boxplot=True):
    """
    Plot boxplots with significance annotations.

    Args:
        df (pandas.DataFrame): The dataframe containing the data.
        x (str): The column name for the x-axis.
        y (str): The column name for the y-axis.
        log_scale (bool, optional): Whether to use a logarithmic scale for the y-axis. Defaults to False.
    """

    if pairs is None or p_values is None:
        pairs = list(combinations(list(df[x].unique()), 2))
        p_values = []
        for p1, p2 in pairs:
            if stat == 't-test':
                p_values.append(
                    ttest_ind(
                        list(df[df[x] == p1][y].values),
                        list(df[df[x] == p2][y].values),
                        equal_var=False,
                        alternative="two-sided",
                        nan_policy='omit')[1])
            elif stat == 'mannwhitneyu':
                p_values.append(
                    mannwhitneyu(
                        list(df[df[x] == p1][y].values),
                        list(df[df[x] == p2][y].values),
                        alternative="two-sided")[1])
            elif stat == 'wilcoxon':
                p_values.append(
                    wilcoxon(
                        list(df[df[x] == p1][y].values),
                        list(df[df[x] == p2][y].values),
                        alternative="two-sided")[1])
            elif stat == 'levene':
                p_values.append(
                    levene(
                        list(df[df[x] == p1][y].values),
                        list(df[df[x] == p2][y].values))[1])
            else:
                raise ValueError(f"Stat test {stat} not supported")
            
    formatted_pvalues = [f"P={pvalue:.2e}" for pvalue in p_values]
    plotting_parameters = {"data": df, "x": x, "y": y, "linewidth": 2.3, 'palette': palette}

    if boxplot:
        ax = sns.boxplot(**plotting_parameters, showfliers=showfliers)
    else:
        ax = sns.violinplot(**plotting_parameters)
    if stripplot:
        sns.stripplot(**plotting_parameters, ax=ax, hue=x)

    if log_scale:
        ax.set_yscale('log')

    annotator = Annotator(ax, pairs, **plotting_parameters);
    annotator.configure(fontsize=fontsize);

    annotator.set_custom_annotations(formatted_pvalues);
    annotator.annotate();
    plt.ylabel(y)
    sns.despine()
    plt.xticks(rotation=30)

def boxplot_pca_significance(df_pca, treatment_col=None, treatment_order=None, PCs=None, palette=None, fontsize=12, stripplot=False):
    """Box plots of the principal component values per treatment (10 panels per row), with the significance of the
    differences between treatments (`plot_boxplot_significance`).

    Args:
        df_pca: one row per sample, with `PC1`, `PC2`, ... columns and a treatment column.
        treatment_col: treatment column.
        treatment_order: order of the treatments (default: order of appearance).
        PCs: 0-based indices of the components to plot (default: every PC column).
        palette: colours per treatment.
        fontsize: font size of the significance annotations. Defaults to 12.
        stripplot: also plot the individual samples. Defaults to False.
    """
    if PCs is None:
        PCs = range(len([col for col in df_pca.columns if col.startswith('PC')]))
    
    nb_pcs = len(PCs)

    dfs=[]
    if treatment_order is not None:
        for treatment in treatment_order:
            df = df_pca[df_pca[treatment_col] == treatment]
            dfs.append(df)
        df_pca = pd.concat(dfs)

    plt.figure(figsize=(24, 6*int(np.ceil(nb_pcs/10))))
    for ax_nb, i in enumerate(PCs):
        plt.subplot(int(np.ceil(nb_pcs/10)), 10, ax_nb+1)
        plot_boxplot_significance(df=df_pca, x=treatment_col, y='PC'+str(i+1), stripplot=stripplot, palette=palette, fontsize=fontsize)

        plt.legend().remove()
    plt.tight_layout()

