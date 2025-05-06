import numpy as np
from scipy.spatial.distance import pdist, squareform
import os

# Load data
actions = np.load("data/actions_matrix.npy")         # (n_policies, n_states)
shap = np.load("data/shap_values.npy")               # (n_states, n_policies, n_objectives)

# Transpose SHAP to (n_policies, n_states, n_objectives)
shap = np.transpose(shap, (1, 0, 2))  # (n_policies, n_states, n_objectives)

n_policies, n_states = actions.shape
n_objectives = shap.shape[2]
print(f"Loaded {n_policies} policies with {n_states} lake states and {n_objectives} objectives")

# Normalize SHAP per state (sum over objectives = 1 for each policy at each X_t)
shap_norm = shap / np.sum(np.abs(shap), axis=2, keepdims=True)

# Flatten per-policy SHAP tensor to (n_policies, n_states * n_objectives)
shap_vectors = shap_norm.reshape(n_policies, n_states * n_objectives)

# Actions are already (n_policies, n_states)
action_vectors = actions

# Compute pairwise Euclidean distances
print("📏 Computing pairwise distances...")

action_dists = squareform(pdist(action_vectors, metric='euclidean'))
shap_dists = squareform(pdist(shap_vectors, metric='euclidean'))

# Save results
os.makedirs("data", exist_ok=True)
np.save("data/action_dists.npy", action_dists)
np.save("data/shap_dists.npy", shap_dists)

print("Saved: data/action_dists.npy and data/shap_dists.npy")
