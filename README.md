# Reproduction of Paper: *"Everyone’s a Winner! On Hyperparameter Tuning of Recommendation Models"*

This repository contains the code for the **Experiment Design Course (Winter Term 2024)** from **Group 6**.

## Overview
To validate the findings of the original paper, **statistical tests** were conducted to examine whether the proposed hypothesis holds.

## Statistical Testing Process
1. The script **`statistical_test.py`** loads the **tuned** and **untuned** cross-validation results stored as JSON files.
2. These results are used to perform **statistical significance tests**.
3. The output is a **DataFrame** containing the p-values (adjusted for multiple testing) for each **tuned vs. untuned model** comparison.
4. The final results are saved as a **CSV file** for further analysis.

## Experiment Environment
The following software versions were used:

| Package        | Version  |
|---------------|----------|
| **NumPy**     | 1.26.4   |
| **Pandas**    | 2.2.3    |
| **SciPy**     | 1.13.1   |
| **Statsmodels** | 0.14.2 |
| **Python**    | 3.9.7    |
