#
# SPDX-FileCopyrightText: Copyright © 2025 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Lisa Fournier <lisa.fournier@idiap.ch>
#
# SPDX-License-Identifier: GPL-3.0-only
#
import rpy2.robjects as robjects

from rpy2.robjects import pandas2ri
from rpy2.robjects.packages import importr

def lmer_per_PC_2re(df_pca, fixed_effect, random_effect1, random_effect2, PC=1,):
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
        lmer_df_r = robjects.conversion.get_conversion().py2rpy(df_pca)

    full_model = lme4.lmer(formula=f"PC{PC} ~ 1 + (1 | {random_effect1}) + (1 | {random_effect2})", data=lmer_df_r)
    reduced_model = lme4.lmer(formula=f"PC{PC} ~ 1 + {fixed_effect} + (1 | {random_effect1}) + (1 | {random_effect2})", data=lmer_df_r)

    anova_result = robjects.r["anova"](reduced_model, full_model)

    with (robjects.default_converter + pandas2ri.converter).context():
        anova_result = robjects.conversion.get_conversion().rpy2py(anova_result)

    print(anova_result)
    p_val = anova_result["Pr(>Chisq)"].values[1]
    return anova_result, p_val

def lmer_per_PC_1re(df_pca, fixed_effect, random_effect, PC=1,):
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
        lmer_df_r = robjects.conversion.get_conversion().py2rpy(df_pca)

    full_model = lme4.lmer(formula=f"PC{PC} ~ 1 + (1 | {random_effect})", data=lmer_df_r)
    reduced_model = lme4.lmer(formula=f"PC{PC} ~ 1 + {fixed_effect} + (1 | {random_effect})", data=lmer_df_r)

    anova_result = robjects.r["anova"](reduced_model, full_model)

    with (robjects.default_converter + pandas2ri.converter).context():
        anova_result = robjects.conversion.get_conversion().rpy2py(anova_result)

    print(anova_result)
    p_val = anova_result["Pr(>Chisq)"].values[1]
    return anova_result, p_val


def lmer_per_PC_re(df_pca, fixed_effect, random_effects, PC=1,):
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
        lmer_df_r = robjects.conversion.get_conversion().py2rpy(df_pca)

    stri = ""
    for random_effect in random_effects:
        stri = f" + (1 | {random_effect})"
    full_model = lme4.lmer(formula=f"PC{PC} ~ 1 {stri}", data=lmer_df_r)
    reduced_model = lme4.lmer(formula=f"PC{PC} ~ 1 + {fixed_effect} {stri}", data=lmer_df_r)

    anova_result = robjects.r["anova"](reduced_model, full_model)

    with (robjects.default_converter + pandas2ri.converter).context():
        anova_result = robjects.conversion.get_conversion().rpy2py(anova_result)

    print(anova_result)
    p_val = anova_result["Pr(>Chisq)"].values[1]
    return anova_result, p_val

def lmer_general(df, predicted_variable, fixed_effect, random_effects):
    """
    Comparison of LMMs with and without fixed effect. Compute the pvalue from the anova resulting from the comparison 
    between the full predicted_variable ~ 1 + fixed_effect + (1 | random_effect) and the 
    reduced one predicted_variable ~ 1 + (1 | random_effect).

    Args:
        df (pandas.DataFrame): The input dataframe containing the data for the linear mixed model.
        predicted_variable (str): The name of the predicted variable in the dataframe.
        fixed_effect (str): The name of the fixed effect in the linear mixed model.
        random_effects (list): A list of names of the random effects in the linear mixed model.

    Returns:
        tuple: A tuple containing the anova result and the p-value resulting from the anova between the full and the reduced linear mixed model.
    """
    lme4 = importr("lme4")

    with (robjects.default_converter + pandas2ri.converter).context():
        lmer_df_r = robjects.conversion.get_conversion().py2rpy(df)

    stri = ""
    for random_effect in random_effects:
        stri = f" + (1 | {random_effect})"
    full_model = lme4.lmer(formula=f"{predicted_variable} ~ 1 {stri}", data=lmer_df_r)
    reduced_model = lme4.lmer(formula=f"{predicted_variable} ~ 1 + {fixed_effect} {stri}", data=lmer_df_r)

    anova_result = robjects.r["anova"](reduced_model, full_model)

    with (robjects.default_converter + pandas2ri.converter).context():
        anova_result = robjects.conversion.get_conversion().rpy2py(anova_result)

    print(anova_result)
    p_val = anova_result["Pr(>Chisq)"].values[1]
    return anova_result, p_val