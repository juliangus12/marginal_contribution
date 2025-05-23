import numpy as np
import pickle
from sklearn.metrics import silhouette_score

# load your precomputed distances
action_dists = np.load("../data/action_dists.npy")
shap_dists   = np.load("../data/shap_dists.npy")

# load the GA’s final population of partitions
with open("../data/pan_cluster_output.pkl","rb") as f:
    pan = pickle.load(f)
P   = pan["P"]            # list of partitions
n_p = action_dists.shape[0]

best_score = -np.inf
best_idx   = None

for i, part in enumerate(P):
    # build a flat label vector of size n_p
    labels = np.empty(n_p, dtype=int)
    for cid, members in enumerate(part):
        labels[members] = cid

    s_obj = silhouette_score(shap_dists,   labels, metric="precomputed")
    s_beh = silhouette_score(action_dists, labels, metric="precomputed")
    score = min(s_obj, s_beh)

    if score > best_score:
        best_score = score
        best_idx   = i

print("→ best partition index:", best_idx)
print("   silhouette_obj =", round(s_obj,3),
      " silhouette_beh =", round(s_beh,3))

# extract your winner
best_partition = P[best_idx]

# now rebuild labels for plotting
best_labels = np.empty(n_p, dtype=int)
for cid, members in enumerate(best_partition):
    best_labels[members] = cid

np.save("../data/best_cluster_labels.npy", best_labels)
