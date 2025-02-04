Exp_Design_Group6

This repository contains the code for the Experiment Design Course (Winter Term 2024) from Group 6.

As part of the project, we aim to reproduce the results of the paper:
📄 "Everyone’s a Winner! On Hyperparameter Tuning of Recommendation Models".

To validate the findings, statistical tests were conducted to examine whether the hypothesis holds.
Statistical Testing Process

    The script statistical_test.py contains the cross-validation results stored as a dictionary.
    These results are used to perform statistical significance tests.
    The output is a DataFrame containing the p-values (adjusted for multiple testing) for each tuned model vs. untuned model comparison.
