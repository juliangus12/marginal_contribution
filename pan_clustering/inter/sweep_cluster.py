import numpy as np
import pickle
from pan_clustering.utils.pareto_analysis import PanClustering
import os

# Load distances
action_dists = np.load("../data/action_dists.npy")
shap_dists = np.load("../data/shap_dists.npy")
distances = {
    "policies": action_dists,
    "objectives": shap_dists
}
n_policies = action_dists.shape[0]

# Sweep configuration
cluster_range = range(4, 11)         # Try k = 4 to 10
seeds = [0, 1, 2]                    # Try 3 random seeds per k
g = 200                              # Number of generations
pop_size = 30                        # Population size per generation

# Mutation / recombination settings
pr = 0.9
pu = 0.4
ps = 0.7
pm = 0.9

output_dir = "../data/pan_sweep_results"
os.makedirs(output_dir, exist_ok=True)

for k in cluster_range:
    for seed in seeds:
        print(f"Running PAN: k={k}, seed={seed}")
        
        cluster = PanClustering(
            n=pop_size,
            g=g,
            distances_matrix=distances,
            pr=pr, pu=pu, ps=ps, pm=pm,
            local_opt=True,
            seed=seed
        )
        
        P, hype, hypervolumes = cluster.run()

        # Save full clustering result
        result = {
            "k": k,
            "seed": seed,
            "P": P,
            "hypervolume": hype,
            "hypervolumes": hypervolumes
        }
        outfile = os.path.join(output_dir, f"pan_k{k}_seed{seed}.pkl")
        with open(outfile, "wb") as f:
            pickle.dump(result, f)
        print(f"✅ Saved: {outfile}")

        # Save cluster label vector
        labels = np.full(n_policies, -1)
        for i, group in enumerate(P):
            for idx in group:
                labels[idx] = i
        np.save(os.path.join(output_dir, f"labels_k{k}_seed{seed}.npy"), labels)
