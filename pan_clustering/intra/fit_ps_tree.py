# fit_ps_tree.py - fits symbolic regressors to clustered segments within a policy
# author: julian gutierrez, Dartmouth College 25S
#
# This script assumes intra-policy clustering has been performed.
# It uses the PS-Tree library to learn symbolic expressions for each segment
# mapping lake state (X_t) to action (a_t) within a policy.

import numpy as np
import matplotlib.pyplot as plt
from intra_cluster import load_policy_data, compute_joint_distance, cluster_policy_states
from pstree.pstree.cluster_gp_sklearn import PSTreeRegressor


# config
POLICY_IDX = 42
N_CLUSTERS = 4
ALPHA = 0.5
OBJECTIVES = ["utility", "reliability", "inertia", "max_P"]

# load action and SHAP vectors
actions, shap_vectors = load_policy_data(POLICY_IDX)
n_states = len(actions)

# mock state indicator for now (concentration proxy)
lake_states = np.linspace(0.0, 2.0, n_states).reshape(-1, 1)  # will become more complex later

# cluster the states
joint_dist = compute_joint_distance(actions, shap_vectors, alpha=ALPHA)
cluster_labels = cluster_policy_states(joint_dist, n_clusters=N_CLUSTERS)

# fit symbolic model per cluster (1D input for now: lake state)
fig, axs = plt.subplots(N_CLUSTERS, 1, figsize=(6, 3 * N_CLUSTERS))
for k in range(N_CLUSTERS):
    idxs = np.where(cluster_labels == k)[0]
    if len(idxs) < 2:
        axs[k].text(0.5, 0.5, f"Cluster {k}: Too few points", ha="center", va="center")
        continue

    X = lake_states[idxs]
    y = actions[idxs]

    model = PSTreeRegressor(
        tree_class=None,  # default
        height_limit=4,
        n_pop=30,
        n_gen=50,
        basic_primitive='optimal',
        size_objective=True
    )
    model.fit(X, y)
    X_plot = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
    y_plot = model.predict(X_plot)

    axs[k].plot(X, y, 'o', label="data")
    axs[k].plot(X_plot, y_plot, label="PS-Tree fit")
    axs[k].set_title(f"Cluster {k} PS-Tree")
    axs[k].set_xlabel("Lake Concentration")
    axs[k].set_ylabel("Action")
    axs[k].legend()
    axs[k].grid(True)

fig.suptitle(f"PS-Tree Fits for Intra-Policy Clusters (Policy {POLICY_IDX})")
plt.tight_layout()
plt.show()
