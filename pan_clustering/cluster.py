import numpy as np
import pickle
from pareto_analysis import PanClustering

# Load precomputed distances
action_dists = np.load("../data/action_dists.npy")
shap_dists = np.load("../data/shap_dists.npy")

# Build distance dictionary
distances = {
    "policies": action_dists,
    "objectives": shap_dists
}

# PAN configuration
n_clusters = 6
g = 5     # generations
pr = 0.9    # recombination
pu = 0.6    # uniform mutation
ps = 0.6    # swap mutation
pm = 0.8    # point mutation

# Run clustering
cluster = PanClustering(
    n=n_clusters,
    g=g,
    distances_matrix=distances,
    pr=pr, pu=pu, ps=ps, pm=pm,
    local_opt=True
)

P, hype, hypervolumes = cluster.run()

# Save full results
results = {
    "P": P,
    "hype": hype,
    "hypervolumes": hypervolumes
}
with open("../data/pan_cluster_output.pkl", "wb") as f:
    pickle.dump(results, f)

# Generate policy-to-cluster label vector
n_policies = action_dists.shape[0]
labels = np.full(n_policies, -1)
for i, cluster_group in enumerate(P):
    for idx in cluster_group:
        labels[idx] = i
np.save("../data/cluster_labels.npy", labels)

print("Saved cluster results to ../data/pan_cluster_output.pkl and ../data/cluster_labels.npy")
