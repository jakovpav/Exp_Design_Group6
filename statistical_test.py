import numpy as np
import pandas as pd
import scipy.stats as stats
from statsmodels.stats.multitest import multipletests
from collections import defaultdict

dict_tuned = {'ConvNeuMF': np.array([0.02026, 0.02202, 0.02384, 0.02279, 0.02742]),
 'DMF': np.array([0.01203, 0.01412, 0.00907, 0.00725, 0.00765]),
 'MultiDAE': np.array([0.03238, 0.03149, 0.03044, 0.01042, 0.02924]),
 'MultiVAE': np.array([0.02746, 0.02431, 0.01797, 0.01751, 0.00536]),
 'NeuMF': np.array([0.02194, 0.01542, 0.02203, 0.0121 , 0.01001]),
 'GMF': np.array([0.02472, 0.00878, 0.02248, 0.00532, 0.0124 ]),
 'NGCF': np.array([0.01844, 0.00222, 0.00091, 0.002  , 0.01358]),
 'ConvMf': np.array([0.02658, 0.028  , 0.02797, 0.02951, 0.02942])}

dict_untuned = {'ConvNeuMF': np.array([0.00331, 0.00593, 0.00329, 0.00618, 0.00536]),
 'DMF': np.array([0.00312, 0.00593, 0.0101 , 0.00875, 0.00224]),
 'MultiDAE': np.array([0.01466, 0.00733, 0.00354, 0.00373, 0.00384]),
 'MultiVAE': np.array([0.00596, 0.0059 , 0.00584, 0.00616, 0.00536]),
 'NeuMF': np.array([0.00598, 0.00593, 0.00585, 0.00617, 0.00536]),
 'GMF': np.array([0.00442, 0.00418, 0.00444, 0.00376, 0.00318]),
 'NGCF': np.array([0.00101, 0.00106, 0.00079, 0.00111, 0.0013 ]),
 'ConvMf': np.array([0.00598, 0.00593, 0.00585, 0.00617, 0.00536])}

def statistical_significance_test_mixed_labeled(dict_tuned, dict_untuned, alpha = 0.05):
   
    """
    Performs a Wilcoxon signed-rank test for paired comparisons (same model tuned vs. untuned)
    and a Mann-Whitney U test for independent comparisons (tuned vs. different untuned models).
    Presents the results as a labeled p-value matrix.

    Args:
        dict_tuned (dict): Model names as keys, lists of 5 CV scores as values (tuned models).
        dict_untuned (dict): Model names as keys, lists of 5 CV scores as values (untuned models).
        alpha (float): Significance level for statistical testing.

    Returns:
        pd.DataFrame: A matrix where rows (Tuned Models) and columns (Untuned Models)
                      are labeled, and values are the p-values from the appropriate test.
    """
    
    comparisons = []
    results = []

    # Create labeled matrix
    model1_names = [f"{m} (Tuned)" for m in dict_tuned.keys()]
    model2_names = [f"{m} (Untuned)" for m in dict_untuned.keys()]
    p_value_matrix = pd.DataFrame(index=model1_names, columns=model2_names, dtype=float)

    # Iterate over all model pairs
    for model1 in dict_tuned:
        for model2 in dict_untuned:
            if model1 == model2:  # Paired test for same model tuned vs. untuned
                stat, p_value = stats.wilcoxon(
                    dict_tuned[model1], dict_untuned[model2], alternative="greater", method="exact"
                )
            else:  # Independent test for different models (same data, different algorithms)
                stat, p_value = stats.mannwhitneyu(
                    dict_tuned[model1], dict_untuned[model2], alternative="greater"
                )
            print(dict_tuned[model1], dict_untuned[model2], p_value)

            # Store results
            results.append(p_value)
            comparisons.append((f"{model1} (Tuned)", f"{model2} (Untuned)"))
            p_value_matrix.loc[f"{model1} (Tuned)", f"{model2} (Untuned)"] = p_value

    # Apply Benjamini-Hochberg (FDR) correction to control Type I error
    corrected_p_values = multipletests(results, method='fdr_bh')[1]

    # Update the matrix with corrected p-values
    for i, (model1, model2) in enumerate(comparisons):
        p_value_matrix.loc[model1, model2] = corrected_p_values[i]

    return p_value_matrix

# Remove 'EASER' from the dictionaries
dict_tuned.pop('EASER', None)
dict_untuned.pop('EASER', None)

# Run statistical test with labeled matrix
p_value_matrix_mixed_labeled = statistical_significance_test_mixed_labeled(dict_tuned, dict_untuned)

# Display the DataFrame
print(p_value_matrix_mixed_labeled)
