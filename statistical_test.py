import numpy as np
import pandas as pd
import scipy.stats as stats
from statsmodels.stats.multitest import multipletests
from collections import defaultdict
import json

def statistical_significance_test(dict_tuned: dict[str, np.ndarray], dict_untuned: dict[str, np.ndarray], alpha: float = 0.05) -> pd.DataFrame:
   
    """
    Performs a Mann-Whitney U test to compare each tuned model against each untuned model.
    Applies Benjamini-Hochberg correction to control the false discovery rate.

    Args:
        dict_tuned (dict[str, np.ndarray]): A dictionary where keys are model names, 
            and values are NumPy arrays containing 5 cross-validation scores (tuned models).
        dict_untuned (dict[str, np.ndarray]): A dictionary where keys are model names, 
            and values are NumPy arrays containing 5 cross-validation scores (untuned models).
        alpha (float, optional): The significance level for statistical testing. Defaults to 0.05.

    Returns:
        pd.DataFrame: A p-value matrix where rows represent tuned models, columns represent 
            untuned models, and values correspond to the corrected p-values from the Mann-Whitney U test.
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
           stat, p_value = stats.mannwhitneyu(dict_tuned[model1], dict_untuned[model2], alternative="greater")
           
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

# Load the tuned data
with open("tuned_results.json", "r") as file:
    dict_tuned = json.load(file)
# Convert lists to NumPy arrays
dict_tuned = {key: np.array(value) for key, value in dict_tuned.items()}

# Load the untuned data
with open("untuned_results.json", "r") as file:
    dict_untuned = json.load(file)
# Convert lists to NumPy arrays
dict_untuned = {key: np.array(value) for key, value in dict_untuned.items()}

# Run statistical test with labeled matrix
p_value_matrix = statistical_significance_test(dict_tuned, dict_untuned)

# Save DataFrame
p_value_matrix.to_csv("p_value_matrix.csv", index=True)

# Display the DataFrame
print(p_value_matrix)

