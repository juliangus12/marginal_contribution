# cluster.py - perform bi-objective clustering using PAN
# author: julian gutierrez, Dartmouth College 25S
#
# PAN (Pareto Analysis via Neighborhoods) algorithm adapted from:
#   Zuzanna Osika, Richard Everitt, and Shimon Whiteson (2022).
#   "Navigating Trade-offs: Policy Summarization for Multi-Objective Reinforcement Learning."
#   Proceedings of the 39th International Conference on Machine Learning (ICML 2022).
#   https://proceedings.mlr.press/v162/osika22a.html
#
# Code adapted from: https://github.com/osikazuzanna/Bi-Objective-Clustering
#
# This script loads pairwise distances in action and SHAP space and clusters policies
# using a multi-objective evolutionary algorithm. Output includes cluster labels and
# hypervolume tradeoff information for analysis and visualization.

import numpy as np
import pickle
from pan_clustering.utils.pareto_analysis import PanClustering

# load previously computed distance matrices
action_dists = np.load("../data/action_dists.npy")
shap_dists = np.load("../data/shap_dists.npy")

# organize distances under meta-objective labels
distances = {
    "policies": action_dists,     # behavioral similarity
    "objectives": shap_dists      # explanatory similarity
}

# PAN hyperparameters
n_clusters = 6       # number of clusters
g = 5                # number of generations
pr = 0.9             # recombination prob.
pu = 0.6             # uniform mutation prob.
ps = 0.6             # swap mutation prob.
pm = 0.8             # point mutation prob.

# run multi-objective evolutionary clustering
cluster = PanClustering(
    n=n_clusters,
    g=g,
    distances_matrix=distances,
    pr=pr, pu=pu, ps=ps, pm=pm,
    local_opt=True
)
P, hype, hypervolumes = cluster.run()

# save clustering results and metrics
results = {
    "P": P,
    "hype": hype,
    "hypervolumes": hypervolumes
}
with open("../data/pan_cluster_output.pkl", "wb") as f:
    pickle.dump(results, f)

# convert clusters into flat label vector for downstream use
n_policies = action_dists.shape[0]
labels = np.full(n_policies, -1)
for i, cluster_group in enumerate(P):
    for idx in cluster_group:
        labels[idx] = i
np.save("../data/cluster_labels.npy", labels)

print("Saved cluster results to ../data/pan_cluster_output.pkl and ../data/cluster_labels.npy")
