# Policy Analysis Module  
## author: julian gutierrez 
## Dartmouth College 25S

This module prepares the core policy-level data structures used throughout the interpretability and clustering pipeline. It contains the implementation of two primary components: `dps.py`, which generates Pareto-optimal policies using Direct Policy Search (DPS), and `shap_analysis.py`, which computes the marginal contribution of each objective to each policy action using SHAP.

The `dps.py` script formulates the shallow lake problem following the structure in Quinn et al. (2017). Policies are defined as radial basis function (RBF) rules mapping observed phosphorus levels to release decisions. A cubic RBF is used with 3 basis functions per policy. The model is optimized using the NSGA-II evolutionary algorithm implemented in the Platypus library, accessed via Rhodium. The output includes both performance objectives (utility, reliability, inertia, max_P) and policy parameters.

The output is saved as `pareto_frontier.csv` with the following header:

max_P,utility,inertia,reliability,center_0,radius_0,weight_0,center_1,radius_1,weight_1,center_2,radius_2,weight_2

The `shap_analysis.py` script explains policy behavior. For each lake state in a fixed grid, the script collects all policy actions and fits a regression model `f(objectives) → action`. SHAP is applied to compute the marginal influence of each objective on the predicted action at that state. This is repeated for all lake states, generating a SHAP tensor of shape `(n_states, n_policies, n_objectives)`. Results are saved to `shap_values.npy` and `actions_matrix.npy`.

These files serve as the input to later stages of the project: both inter-policy clustering (based on action or SHAP similarity) and intra-policy analysis (grouping different behavioral modes within a policy). The lake state grid used for evaluation spans `[0.0, 2.0]` in 201 steps. Actions are clipped to `[0.01, 0.1]` to conform to model bounds.

The module assumes that the RBF structure and objective ordering are consistent across scripts. The regression model used for SHAP is a `RandomForestRegressor` from `sklearn`, chosen for its interpretability and robustness to feature interactions. SHAP values are computed using the TreeExplainer interface from the `shap` package.

All Outputs are saved in the `../data/` directory relative to the module, excluding the cache files generated when running `dps.py`. Both scripts are validated under realistic workloads and are suitable for batch experimentation.
