import pandas as pd
import numpy as np
import shap
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
from tqdm import tqdm  # for progress bar

# Load Pareto frontier data
df = pd.read_csv("pareto_frontier.csv")

# Extract objectives
objective_names = ["utility", "reliability", "inertia", "max_P"]
X = df[objective_names].values  # shape: (n_policies, 4)

# Lake state values (observations)
lake_states = np.linspace(0.0, 2.0, 201)

# Define RBF policy evaluator
def RBFpolicy(Xvals, vars, n_rbfs=3):
    actions = []
    weight_sum = sum([vars[f"weight_{i}"] for i in range(n_rbfs)])
    weights = [vars[f"weight_{i}"]/weight_sum for i in range(n_rbfs)]

    for x in Xvals:
        y = 0
        for i in range(n_rbfs):
            c = vars[f"center_{i}"]
            r = vars[f"radius_{i}"]
            w = weights[i]
            if r != 0:
                y += w * (abs((x - c)/r) ** 3)
        y = min(0.1, max(y, 0.01))
        actions.append(y)

    return actions

# Step 1: Build actions matrix: shape (n_policies, n_states)
actions_matrix = []
for _, row in df.iterrows():
    actions = RBFpolicy(lake_states, row)
    actions_matrix.append(actions)
actions_matrix = np.array(actions_matrix)  # shape: (n_policies, n_states)

# Step 2: Compute SHAP values across all lake states
n_policies, n_states = actions_matrix.shape
shap_results = np.zeros((n_states, n_policies, len(objective_names)))

print("Computing SHAP values for all lake states...")
for state_idx in tqdm(range(n_states)):
    Y = actions_matrix[:, state_idx]  # action taken at X_t
    model = RandomForestRegressor().fit(X, Y)
    explainer = shap.Explainer(model)
    shap_values = explainer(X).values  # shape: (n_policies, 4)
    shap_results[state_idx] = shap_values

# Save both SHAP and actions
np.save("../data/shap_values.npy", shap_results)
np.save("../data/actions_matrix.npy", actions_matrix)

print("Saved shap_values.npy and actions_matrix.npy to ../data/")
