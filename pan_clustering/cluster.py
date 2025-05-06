import numpy as np
from pareto_analysis import PanClustering

# Load precomputed distances
action_dists = np.load("../data/action_dists.npy")
shap_dists = np.load("../data/shap_dists.npy")

# Combine into a tuple or pass separately depending on PanClustering API
distances = {
    "behavior": action_dists,
    "objectives": shap_dists
}

# Example config
n_clusters = 6
g = 100     # generations
pr = 0.9    # recombination
pu = 0.6    # uniform mutation
ps = 0.6    # swap mutation
pm = 0.8    # point mutation

# Run PAN
cluster = PanClustering(n=n_clusters, g=g, distances=distances,
                        pr=pr, pu=pu, ps=ps, pm=pm, local_opt=True)

P, hype, hypervolumes = cluster.run()
