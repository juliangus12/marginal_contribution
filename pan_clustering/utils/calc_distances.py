# calc_distances.py - compute pairwise distances between policies in action and SHAP space
# author: julian gutierrez, Dartmouth College 25S
# part of the clustering module for interpretable MORDM
# 
# This script prepares the distance matrices required for Pareto-based clustering.
# SHAP values are normalized per lake state and flattened; actions are used directly.
# Outputs are stored in ../data/ and serve as inputs to cluster.py

import numpy as np
from scipy.spatial.distance import pdist, squareform
import os

# load policy action sequences and per-state SHAP values
actions = np.load("../data/actions_matrix.npy")         # shape: (n_policies, n_states)
shap = np.load("../data/shap_values.npy")               # shape: (n_states, n_policies, n_objectives)

# reorient shap to (n_policies, n_states, n_objectives)
shap = np.transpose(shap, (1, 0, 2))

n_policies, n_states = actions.shape
n_objectives = shap.shape[2]
print(f"Loaded {n_policies} policies with {n_states} lake states and {n_objectives} objectives")

# normalize SHAP per lake state so that sum over objectives = 1 for each policy and lake state
shap_norm = shap / np.sum(np.abs(shap), axis=2, keepdims=True)

# flatten SHAP vectors into shape (n_policies, n_states * n_objectives)
shap_vectors = shap_norm.reshape(n_policies, n_states * n_objectives)

# actions already in shape (n_policies, n_states)
action_vectors = actions

# compute pairwise Euclidean distances in action and SHAP space
action_dists = squareform(pdist(action_vectors, metric='euclidean'))
shap_dists = squareform(pdist(shap_vectors, metric='euclidean'))

# write to disk for use in clustering stage
os.makedirs("../data", exist_ok=True)
np.save("../data/action_dists.npy", action_dists)
np.save("../data/shap_dists.npy", shap_dists)

print("Saved: ../data/action_dists.npy and ../data/shap_dists.npy")
