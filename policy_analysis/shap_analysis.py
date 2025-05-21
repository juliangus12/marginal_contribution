# shap_analysis.py - compute SHAP values for policy actions from objectives
# author: julian gutierrez, dartmouth college, spring 2025
#
# this module estimates the marginal contribution of each objective
# (e.g., utility, reliability) to the action a policy takes at a given lake state.
#
# for each observation x_t, we fit a regression model: f(objectives) -> action
# then apply shap to estimate local feature attributions.
#
# outputs:
#   - shap_values.npy: array of shape (n_states, n_policies, n_objectives)
#   - actions_matrix.npy: policy actions at each lake state
#
# dependencies: pandas, numpy, shap, sklearn, matplotlib, tqdm

import pandas as pd
import numpy as np
import shap
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
from tqdm import tqdm  # progress bar for long loops

# load pareto frontier results with both objective values and RBF parameters
df = pd.read_csv("pareto_frontier.csv")

# define the four performance objectives used as features
objective_names = ["utility", "reliability", "inertia", "max_P"]
X = df[objective_names].values  # (n_policies, 4)

# define the lake observation space to evaluate policies over
lake_states = np.linspace(0.0, 2.0, 201)  # 201 evenly spaced states

# evaluates RBF policy for a given lake state sequence
def RBFpolicy(Xvals, vars, n_rbfs=3):
    actions = []

    # normalize weights to sum to 1
    weight_sum = sum(vars[f"weight_{i}"] for i in range(n_rbfs))
    weights = [vars[f"weight_{i}"] / weight_sum for i in range(n_rbfs)]

    for x in Xvals:
        y = 0
        for i in range(n_rbfs):
            c = vars[f"center_{i}"]
            r = vars[f"radius_{i}"]
            w = weights[i]
            if r != 0:
                y += w * (abs((x - c) / r) ** 3)  # cubic basis
        y = min(0.1, max(y, 0.01))  # clip to allowed action bounds
        actions.append(y)

    return actions

# build a matrix of actions for each policy at every lake state
actions_matrix = []
for _, row in df.iterrows():
    actions = RBFpolicy(lake_states, row)
    actions_matrix.append(actions)
actions_matrix = np.array(actions_matrix)  # shape: (n_policies, n_states)

# initialize SHAP result container
n_policies, n_states = actions_matrix.shape
shap_results = np.zeros((n_states, n_policies, len(objective_names)))

print("Computing SHAP values for all lake states...")

# for each lake state, learn f(objectives) -> action and explain with SHAP
for state_idx in tqdm(range(n_states)):
    Y = actions_matrix[:, state_idx]  # target: action taken at this state
    model = RandomForestRegressor().fit(X, Y)
    explainer = shap.Explainer(model)
    shap_values = explainer(X).values  # (n_policies, n_objectives)
    shap_results[state_idx] = shap_values

# save SHAP values and actions for downstream clustering/visualization
np.save("../data/shap_values.npy", shap_results)
np.save("../data/actions_matrix.npy", actions_matrix)

print("Saved shap_values.npy and actions_matrix.npy to ../data/")
