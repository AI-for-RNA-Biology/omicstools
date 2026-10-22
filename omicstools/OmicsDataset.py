#
# SPDX-FileCopyrightText: Copyright © 2025 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileCopyrightText: Copyright © 2026 University of Bern
#
# SPDX-FileContributor: Lisa Fournier <lisa.fournier@idiap.ch>
# SPDX-FileContributor: Lisa Fournier <lisa.fournier@unibe.ch>
#
# SPDX-License-Identifier: GPL-3.0-only
#
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from omicstools.helper_plot import plot_boxplot_significance
from scanalysis.dimred import compute_svd_correlation, select_correlation_kmeans
from scipy.stats import spearmanr
import itertools
import rpy2.robjects as robjects
from rpy2.robjects import pandas2ri
from rpy2.robjects.packages import importr
import os



class OmicsDataset:
    """Omics dataset (features in rows, samples in columns) grouped by treatment, with SVD/PCA analyses of its log2
    values and their association with the treatments or with sample covariates.
    """
    def __init__(self, df, df_log=None, treatments=None, treatment_name="treatment", feature_type="feature",
                 nb_comp=8, metadata=None):
        """
        Initialize an instance of the OmicsDataset class.

        Parameters:
        - df (pandas.DataFrame): The input DataFrame containing the dataset.
        Usually takes features in rows and samples in columns.
        - treatments (list or None): Optional list of treatments applied to the dataset.
        - metadata (pandas.DataFrame): Optional metadata DataFrame. The samples should be in the index and match the columns of the input DataFrame.

        Returns:
        None
        """
        if treatments is None:
            samples = df.columns
        else:
            all_samples = list(itertools.chain.from_iterable(treatments.values()))
            samples = [sample for sample in all_samples if sample in df.columns]

        self.df = df[samples]
        if df_log is not None:
            self.df_log = df_log[samples]
        else:   
            self.df_log = np.log2(self.df+1)
        self.treatments = treatments
        self.treatment_name = treatment_name
        self.treatment_order = self.treatments.keys()
        self.feature_type = feature_type
        self.pca = None
        self.nb_comp = nb_comp
        self.metadata = metadata

    
    def perform_svd_analysis(self, top_n=8, remove_first_comp=True, save_path=None, prefix=""):
        """
        Perform Singular Value Decomposition (SVD) analysis on the dataset.

        Args:
            top_n (int): Number of top components to keep (default is 8).

        Returns:
            None
        """

        from numpy.linalg import svd

        U, s, Vt = svd(self.df_log.T, full_matrices=False)
        self.U = U
        self.s = s
        self.Vt = Vt

        explained_variance = s*100 / s.sum()

        plt.figure(figsize=(6, 6))
        sns.scatterplot(data=pd.DataFrame(explained_variance, index=['PC'+str(i+1) for i in range(len(explained_variance))])[:top_n])
        plt.xlabel('PC component', fontsize=15)
        plt.ylabel('Percentage of variance explained [%]', fontsize=15)
        plt.title('Variance explained by each PC', weight='bold', fontsize=15)
        sns.despine()
        plt.xticks(rotation=90, fontsize=8)
        if save_path is not None:
            plt.savefig(os.path.join(save_path, f"{prefix}_explained_variance_raw.pdf"))

        if remove_first_comp:
            # Remove the first component
            X_without_PC1 = self.df_log.T - np.dot(np.array([U[:, 0]*s[0]]).T, np.array([Vt[0]]))

            # Do the SVD again
            U_w1, s_w1, Vt_w1 = svd(X_without_PC1, full_matrices=False)

            explained_variance_w1 = s_w1*100 / s_w1.sum()

            plt.figure(figsize=(6, 6))
            sns.scatterplot(data=pd.DataFrame(explained_variance_w1, index=['PC'+str(i+1) for i in range(len(explained_variance_w1))])[:top_n])
            plt.xlabel('PC component', fontsize=15)
            plt.ylabel('Percentage of variance explained [%]', fontsize=15)
            plt.title('Variance explained by each PC', weight='bold', fontsize=15)
            sns.despine()
            plt.xticks(rotation=90, fontsize=8)
            if save_path is not None:
                plt.savefig(os.path.join(save_path, f"{prefix}_explained_variance_denoised.pdf"))

            df_pca = pd.DataFrame((U_w1[:, :len(s_w1)] * s_w1)[:, :top_n], index=self.df_log.columns, columns=['PC'+str(i+1) for i in range(top_n)])
            self.pca = df_pca.merge(self.from_treatments_dict_to_treatments_df(), left_index=True, right_index=True)
            self.U_w1 = U_w1
            self.Vt_w1 = Vt_w1
            self.X_without_PC1 = X_without_PC1
            self.s_w1 = s_w1
        else:
            df_pca = pd.DataFrame((U[:, :len(s)] * s)[:, :top_n], index=self.df_log.columns, columns=['PC'+str(i+1) for i in range(top_n)])
            self.pca = df_pca.merge(self.from_treatments_dict_to_treatments_df(), left_index=True, right_index=True)

    

    def remove_first_component(self):   
        """Remove the first component of the data already denoised by `perform_svd_analysis(remove_first_comp=True)`
        (second removal), plot the variance explained after the removal and store the new principal components in
        `self.pca`.
        """
        # TODO: should be modified: this method fails as is: the PC columns are hard-coded to range(47) (number
        # of samples of one dataset), the figure path ./Figures/variance_explained_PUD_denoised_debatched.pdf is
        # hard-coded, and all_treatments / all_samples (returned df_treatment) are undefined.
        from numpy.linalg import svd


        X_without_PC1_bis = self.X_without_PC1 - np.dot(np.array([self.U_w1[:, 0]*self.s_w1[0]]).T, np.array([self.Vt_w1[0]])) 

        # Do the SVD again
        U_w1_bis, s_w1_bis, Vt_w1_bis = svd(X_without_PC1_bis, full_matrices=False)

        explained_variance_w1_bis = s_w1_bis*100 / s_w1_bis.sum()

        plt.figure(figsize=(6, 6))
        sns.scatterplot(data=pd.DataFrame(explained_variance_w1_bis, index=['PC'+str(i+1) for i in range(len(explained_variance_w1_bis))]))
        plt.xlabel('PC component', fontsize=15)
        plt.ylabel('Percentage of variance explained [%]', fontsize=15)
        plt.title('Variance explained by each PC', weight='bold', fontsize=15)
        sns.despine()
        plt.xticks(rotation=90, fontsize=8)
        plt.tight_layout()
        plt.savefig("./Figures/variance_explained_PUD_denoised_debatched.pdf", bbox_inches='tight')

        self.pca = pd.DataFrame((U_w1_bis[:, :len(s_w1_bis)] * s_w1_bis), index=self.df_log.columns, columns=['PC'+str(i+1) for i in range(47)])
        if self.metadata is not None:
            self.pca = self.pca.merge(self.metadata, left_index=True, right_index=True)
        self.U_w1 = U_w1_bis
        self.Vt_w1 = Vt_w1_bis
        self.X_without_PC1 = X_without_PC1_bis
        self.s_w1 = s_w1_bis


    
    def projection_correlation_one_PC(self, PC=1, min_extreme='Sensitive extreme', max_extreme='Resistant extreme', k_clusters=10, sort_by="projection",
                                      min_extreme_color='royalblue', max_extreme_color='orange', corr_on_original_X=False, scaling=False):
            """
            Computes the projection correlation for one principal component (PC) of the dataset.

            Args:
                PC (int): The principal component to consider. Default is 1.
                min_extreme (str): The label for the minimum extreme. Default is 'Sensitive extreme'.
                max_extreme (str): The label for the maximum extreme. Default is 'Resistant extreme'.
                k_clusters (int): The number of clusters to use for k-means. Default is 10.

            Returns:
                tuple: A tuple containing two lists. The first list contains the features specific to the minimum extreme,
                       and the second list contains the features specific to the maximum extreme.
            """
            if corr_on_original_X:
                corr_svd = compute_svd_correlation(U=self.U_w1, X=np.array(self.df_log.T), nb_comp=self.nb_comp)
            else:
                corr_svd = compute_svd_correlation(U=self.U_w1, X=np.array(self.X_without_PC1), nb_comp=self.nb_comp)

            features_min_PC1, features_max_PC1 = select_correlation_kmeans(corr_svd=corr_svd, 
                                                                Vt=self.Vt_w1, 
                                                                df=self.df_log.reset_index(drop=True).T, 
                                                                pc_name=['PC'+str(PC)], 
                                                                k_clusters=k_clusters, 
                                                                color_extreme=[min_extreme, max_extreme], 
                                                                top_genes_max=1, 
                                                                top_genes_min=1, 
                                                                var_names=list(self.df_log.index), 
                                                                random_state=42,
                                                                sort_by=sort_by,
                                                                min_extreme_color=min_extreme_color,
                                                                max_extreme_color=max_extreme_color,
                                                                scaling=scaling,)

            index_column_name = self.df_log.index.name

            if index_column_name is None:
                index_column_name = 'index'
            PC_min_features = list(set(list(self.df_log.reset_index().loc[features_min_PC1][index_column_name])))
            # pd.DataFrame(PC_min_features).to_csv(f"{self.df_normalized_log_index}/PC1_sensitive_specific_genes.csv", index=False, header=False)


            PC_max_features = list(set(list(self.df_log.reset_index().loc[features_max_PC1][index_column_name])))
            # pd.DataFrame(PC_max_features).to_csv(f"{self.df_normalized_log_index}/PC1_resistant_specific_genes.csv", index=False, header=False)
            return PC_min_features, PC_max_features
    
    def distribution_one_feature_across_treatments(self, feature_name, use_log=False, stat='t-test'):
        """
        Plot the distribution of one feature across the samples.

        Parameters:
        - feature_name (str): Name of the feature.
        """
        if use_log:
            df = self.df_log
        else:        
            df = self.df

        df_treatment = self.from_treatments_dict_to_treatments_df()
        if feature_name not in df.index:
            print(f"{feature_name} not in {self.feature_type} data")
            return
        
        df_feature = pd.DataFrame(df.loc[feature_name])
        df = df_feature.merge(df_treatment, left_index=True, right_index=True)

        if self.treatment_order is not None:
            ordered_dfs = []
            for treatment in self.treatment_order:
                ordered_dfs.append(df[df[self.treatment_name] == treatment])
            df = pd.concat(ordered_dfs)

        plot_boxplot_significance(df, self.treatment_name, feature_name, stat=stat)
        plt.title(feature_name, weight='bold')
        plt.ylabel(f"{self.feature_type} expression")

    def from_treatments_dict_to_treatments_df(self):
        """
        Convert treatments dictionary to treatments DataFrame.

        Returns:
        - pd.DataFrame: DataFrame containing treatments information.
        """
        all_samples = []
        all_treatments = []

        for treatment, samples in self.treatments.items():
            all_samples += samples
            all_treatments += [treatment]*len(samples)

        df_treatment = pd.DataFrame(all_treatments, index=all_samples, columns=[self.treatment_name])
        return df_treatment
    
    def correlation_PCs_and_X(self, X, top_n=8, name="IC50", abs=False):
        """Spearman correlation between each of the `top_n` first principal components (`self.pca`) and the sample
        values `X`, plotted as bar plots (correlation and -log10(p-value)).

        Args:
            X: one value per sample, in the order of `self.pca`.
            top_n: number of components. Defaults to 8.
            name: name of `X` in the plot titles. Defaults to "IC50".
            abs: use the absolute correlation. Defaults to False.

        Returns:
            tuple: ({PC: correlation}, {PC: -log10(p-value)}).
        """

        spearman_corr = {}
        p_vals = {}
        for i in range(top_n):
            if abs:
                spearman_corr[f"PC{i+1}"] = np.abs(spearmanr(self.pca[f"PC{i+1}"], X)[0])
            else:
                spearman_corr[f"PC{i+1}"] = spearmanr(self.pca[f"PC{i+1}"], X)[0]
            p_vals[f"PC{i+1}"] = -np.log10(spearmanr(self.pca[f"PC{i+1}"], X)[1])

        plt.figure(figsize=(15, 5))
        plt.subplot(1, 2, 1)
        sns.barplot(data=pd.DataFrame.from_dict(spearman_corr, orient='index').T, color='gray')
        plt.xlabel("PC")
        plt.ylabel("Spearman correlation")
        plt.title(f"Spearman correlation between PCs and {name} values", weight='bold')
        plt.xticks(rotation=90, fontsize=8)

        plt.subplot(1, 2, 2)
        sns.barplot(data=pd.DataFrame.from_dict(p_vals, orient='index').T, color='gray')
        plt.xlabel("PC")
        plt.ylabel("-log10(p-value)")
        plt.title(f"p-value for the spearman correlation \n between PCs and {name} values", weight='bold')
        plt.xticks(rotation=90, fontsize=8)

        return spearman_corr, p_vals

    def correlation_PCs_and_treatment(self, top_n=8):
        """One-way ANOVA of each of the `top_n` first principal components (`self.pca`) between the treatments, plotted
        as bar plots (F value and -log10(p-value)).

        Args:
            top_n: number of components. Defaults to 8.
        """
        from scipy.stats import f_oneway
        f_values = {}
        p_values = {}
        treatments = list(self.treatments.keys())
        for i in range(top_n):
            pca_values = [self.pca[self.pca[self.treatment_name] == treatment][f"PC{i+1}"] for treatment in treatments]
            f_value, p_value = f_oneway(*pca_values)
            f_values[f"PC{i+1}"] = f_value
            p_values[f"PC{i+1}"] = -np.log10(p_value)
        
        fig, ax = plt.subplots(1, 2, figsize=(15, 5))
        sns.barplot(data=pd.DataFrame.from_dict(f_values, orient='index').T, color='gray', ax=ax[0])
        ax[0].set_xlabel("PC")
        ax[0].set_ylabel("f-value")
        ax[0].set_title(f"ANOVA between PCs and treatment", weight='bold')

        sns.barplot(data=pd.DataFrame.from_dict(p_values, orient='index').T, color='gray', ax=ax[1])
        ax[1].set_xlabel("PC")
        ax[1].set_ylabel("-log10(p-value)")
        ax[1].set_title(f"ANOVA between PCs and treatment", weight='bold')


    def point_biserial_correlation(self, treatment1, treatment2, top_n=8, abs=False):
        """
        Compute the point biserial correlation between each of the PC components with two given treatments.

        Parameters:
        - treatment1 (str): The first treatment.
        - treatment2 (str): The second treatment.

        Returns:
        - dict: A dictionary containing the point biserial correlation coefficient for each PC component.
        """
        from scipy.stats import pointbiserialr

        pc_correlation = {}
        p_vals = {}
        for i in range(top_n):
            pc_component = self.pca[f"PC{i+1}"]
            treatment1_values = pc_component[self.pca[self.treatment_name] == treatment1]
            treatment2_values = pc_component[self.pca[self.treatment_name] == treatment2]
            bools = [False] * len(treatment1_values) + [True] * len(treatment2_values)
            correlation, p_val = pointbiserialr(bools, list(treatment1_values) + list(treatment2_values))
            pc_correlation[f"PC{i+1}"] = np.abs(correlation)
            p_vals[f"PC{i+1}"] = -np.log10(p_val)

        plt.figure(figsize=(15, 5))
        plt.subplot(1, 2, 1)
        sns.barplot(data=pd.DataFrame.from_dict(pc_correlation, orient='index').T, color='gray')
        plt.xlabel("PC")
        plt.ylabel("correlation")
        plt.title(f"Point biserial correlation between PCs and the two treatments: \n {treatment1} & {treatment2}", weight='bold')
        plt.xticks(rotation=90, fontsize=8)

        plt.subplot(1, 2, 2)
        sns.barplot(data=pd.DataFrame.from_dict(p_vals, orient='index').T, color='gray')
        plt.xlabel("PC")
        plt.ylabel("-log10(p-value)")
        plt.title(f"Point biserial correlation between PCs and the two treatments: \n {treatment1} & {treatment2}", weight='bold')
        plt.xticks(rotation=90, fontsize=8)

        return pc_correlation, p_vals
        
    def lmer_per_PC(self, fixed_effect, random_effect, PC=1,):
        """
        Comparison of LMMs with and without fixed effect. Compute the pvalue from the anova resulting from the comparison 
        between the full modelPC ~ 1 + fixed_effect + (1 | random_effect) and the 
        reduced one PC ~ 1 + (1 | random_effect).

        Args:
            comp (int): SVD component to test. Start from 1.
            cluster_name (str, optional): Cluster name from predicted_label column to test.

        Returns:
            float: pvalue resulting from the anova between the full and the reduced linear mixed model
        """
        lme4 = importr("lme4")

        with (robjects.default_converter + pandas2ri.converter).context():
            lmer_df_r = robjects.conversion.get_conversion().py2rpy(self.pca)

        full_model = lme4.lmer(formula=f"PC{PC} ~ 1 + (1 | {random_effect})", data=lmer_df_r)
        reduced_model = lme4.lmer(formula=f"PC{PC} ~ 1 + {fixed_effect} + (1 | {random_effect})", data=lmer_df_r)

        anova_result = robjects.r["anova"](reduced_model, full_model)

        with (robjects.default_converter + pandas2ri.converter).context():
            anova_result = robjects.conversion.get_conversion().rpy2py(anova_result)

        print(anova_result)
        p_val = anova_result["Pr(>Chisq)"].values[1]
        return anova_result, p_val
    
    def hierarchical_clustering(self, 
                                palette,
                                log=False, 
                                treatments=None, 
                                method='spearman', color_threshold=1,
                                level='sample'):
        """
        Hierarchical clustering of samples (level='sample') or of treatments (level='treatment').
        With level='treatment', the samples of each treatment are first averaged so that
        the leaves of the dendrogram are the treatments.
        """
        from scipy.cluster import hierarchy

        if log:
            mat = self.df_log
        else:
            mat = self.df

        if treatments is None:
            treatments = self.treatments
        
        order_treatments = []
        if treatments is None:
            for treatment in self.treatment_order:
                order_treatments += self.treatments[treatment]
        else:
            for treatment in treatments:
                order_treatments += self.treatments[treatment]
        
        df = mat[order_treatments]
        if level == 'treatment':
            sample_to_treatment = {sample: treatment for treatment in treatments for sample in treatments[treatment]}
            df = df.T.groupby(sample_to_treatment).mean().T[list(treatments.keys())]
        correlation_matrix = df.corr(method=method)

        fig, ax = plt.subplots(1, 1, figsize=(7, 5))
        # Perform hierarchical clustering
        dendrogram = hierarchy.dendrogram(hierarchy.linkage(correlation_matrix, method='average'), 
                                        labels=df.columns, 
                                        color_threshold=color_threshold,
                                        above_threshold_color='k')

        # Show the dendrogram
        plt.xlabel('Treatments' if level == 'treatment' else 'Samples')
        plt.ylabel('Distance')
        plt.title('Hierarchical Clustering Dendrogram', weight='bold')


        for label in ax.get_xticklabels():
            for i, treatment in enumerate(treatments.keys()):
                leaves = [treatment] if level == 'treatment' else treatments[treatment]
                for leaf in leaves:
                    if label.get_text() == leaf:
                        label.set_color(palette[i])

        plt.xticks(rotation=90);