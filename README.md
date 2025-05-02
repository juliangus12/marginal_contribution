# marginal_contribution



## gen_shap
This directory contains the full workflow for generating SHAP explinations of Pareto-optimal policies for a the shallow lake problem. These explinations estimate the marginal contribution of each objective to the action taken by each policy at every lake state. 

`shap_analysis.py` builds a dataset where, the input is are the objective performances of each policy and the target is to predict the action for each policy, fixed at each state. 

The file outputs `shap_values.npy` a (n_states, n_polices, 4) array where each slice contains SHAP values for utility, reliability, intertia, and max_P at a specific lake state X_t


`visualize.py` Plots a selected policy's action curve alongside stacked SHAP values over lake states. This visual shows the action take at each lake state along with the relative contribution of each objective to that actin.
