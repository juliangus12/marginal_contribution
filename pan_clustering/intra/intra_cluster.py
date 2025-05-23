# intra_cluster.py - clusters actions within a single policy based on behavioral and explanatory similarity
# author: julian gutierrez, Dartmouth College 25S
#
# This script performs intra-policy clustering using combined action and SHAP similarity.
# The goal is to segment a policy's decision rule into coherent behavior-rationale groups
# and prepare for symbolic modeling of each segment (e.g., with PS-Tree or local regressors).

import numpy as np
from sklearn.metrics import pairwise_distances
from sklearn.cluster import AgglomerativeClustering
import matplotlib.pyplot as plt

# config: paths and hyperparameters
SHAP_PATH = "../../data/shap_values.npy"
ACTION_PATH = "../../data/actions_matrix.npy"
OBJECTIVES = ["utility", "reliability", "inertia", "max_P"]
ALPHA = 0.5  # weighting between action and SHAP distance

def load_policy_data(policy_index):
    """extracts action and shap vectors for a single policy"""
    actions = np.load(ACTION_PATH)[policy_index]  # shape: (n_states,)
    shap = np.load(SHAP_PATH)                     # shape: (n_states, n_policies, n_objectives)
    shap = np.transpose(shap, (1, 0, 2))          # shape: (n_policies, n_states, n_objectives)
    shap_vectors = shap[policy_index]             # shape: (n_states, n_objectives)

    # normalize shap values for interpretability (L1 normalization)
    shap_vectors = shap_vectors / (np.sum(np.abs(shap_vectors), axis=1, keepdims=True) + 1e-16)

    return actions, shap_vectors

def compute_joint_distance(actions, shap_vectors, alpha=0.5):
    """computes pairwise distance matrix combining action and SHAP similarity"""
    action_dist = pairwise_distances(actions.reshape(-1, 1), metric='euclidean')
    shap_dist = pairwise_distances(shap_vectors, metric='euclidean')

    return alpha * action_dist + (1 - alpha) * shap_dist

def cluster_policy_states(dist_matrix, n_clusters=3):
    """clusters the lake states using Agglomerative Clustering"""
    clustering = AgglomerativeClustering(n_clusters=n_clusters, metric='precomputed', linkage='average')
    return clustering.fit_predict(dist_matrix)

def plot_clusters(actions, shap_vectors, labels, policy_index):
    """visualizes each cluster’s action vs lake concentration and SHAP profile (scatter plot style)"""
    n_clusters = np.max(labels) + 1
    objectives = np.array(OBJECTIVES)

    # reconstruct lake state grid
    lake_states = np.linspace(0.0, 2.0, len(actions))  # shape: (n_states,)

    fig, axs = plt.subplots(n_clusters, 2, figsize=(12, 3 * n_clusters))

    for k in range(n_clusters):
        idxs = np.where(labels == k)[0]

        # left: scatter of actions vs lake concentration
        axs[k, 0].scatter(lake_states[idxs], actions[idxs], label=f"Cluster {k}", s=20)
        axs[k, 0].set_ylabel("Action")
        axs[k, 0].set_xlabel("Lake Concentration (X)")
        axs[k, 0].legend()

        # right: SHAP contribution bars
        avg_shap = shap_vectors[idxs].mean(axis=0)
        axs[k, 1].bar(objectives, avg_shap)
        axs[k, 1].set_ylim(0, 1)
        axs[k, 1].set_ylabel("Mean SHAP")
        axs[k, 1].set_title(f"Cluster {k} SHAP Contribution")

    fig.suptitle(f"Intra-Policy Clustering (Policy {policy_index})")
    plt.tight_layout()
    plt.show()


def run_intra_clustering(policy_index, n_clusters=3, alpha=0.5):
    """runs full intra-policy clustering pipeline"""
    actions, shap_vectors = load_policy_data(policy_index)
    dist_matrix = compute_joint_distance(actions, shap_vectors, alpha)
    labels = cluster_policy_states(dist_matrix, n_clusters=n_clusters)
    plot_clusters(actions, shap_vectors, labels, policy_index)
# def plot_clusters(actions, shap_vectors, labels, policy_index):
#     """Visualizes action trajectory and SHAP bar plots with better layout and more distinct colors."""
#     import matplotlib.cm as cm

#     n_clusters = np.max(labels) + 1
#     objectives = np.array(OBJECTIVES)
#     lake_states = np.linspace(0.0, 2.0, len(actions))

#     # use better separated colors (e.g. Set2 or tab20)
#     colors = cm.get_cmap('Set2', n_clusters)

#     fig = plt.figure(figsize=(14, 2.5 + 2.5 * n_clusters))

#     # top trajectory plot
#     ax1 = plt.subplot2grid((n_clusters + 2, 2), (0, 0), colspan=2)
#     for i in range(len(actions) - 1):
#         x_pair = lake_states[i:i+2]
#         y_pair = actions[i:i+2]
#         cluster_color = colors(labels[i])
#         ax1.plot(x_pair, y_pair, color=cluster_color, linewidth=1.8)
#         ax1.scatter(lake_states[i], actions[i], color=cluster_color, edgecolor='k', s=30, zorder=3)

#     ax1.scatter(lake_states[-1], actions[-1], color=colors(labels[-1]), edgecolor='k', s=30, zorder=3)
#     ax1.set_xlabel("Lake Concentration (X)")
#     ax1.set_ylabel("Action")
#     ax1.set_title("Policy Action Trajectory (colored by intra-cluster label)")

#     # individual cluster SHAP bar plots
#     for k in range(n_clusters):
#         ax_bar = plt.subplot2grid((n_clusters + 2, 2), (k + 1, 0), colspan=2)
#         idxs = np.where(labels == k)[0]
#         avg_shap = shap_vectors[idxs].mean(axis=0)
#         ax_bar.bar(objectives, avg_shap, color=colors(k))
#         max_y = max(avg_shap.max(), 1e-2)
#         ax_bar.set_ylim(0, max_y * 1.2)
#         ax_bar.set_ylabel("Mean SHAP")
#         ax_bar.set_title(f"Cluster {k} Attribution")

#         for i, val in enumerate(avg_shap):
#             ax_bar.text(i, val + 0.01 * max_y, f"{val:.2f}", ha='center', va='bottom', fontsize=8)

#     plt.tight_layout(h_pad=3.5)
#     plt.suptitle(f"Intra-Policy Clustering: Actions + SHAP (Policy {policy_index})", fontsize=14, y=1.02)
#     plt.subplots_adjust(top=0.93)
#     plt.show()

# def plot_clusters(actions, shap_vectors, labels, policy_index):
#     """visualizes clusters with connected action points and separate SHAP bars per cluster"""
#     import matplotlib.cm as cm

#     n_clusters = np.max(labels) + 1
#     objectives = np.array(OBJECTIVES)
#     lake_states = np.linspace(0.0, 2.0, len(actions))  # x-axis values

#     colors = cm.get_cmap('tab10', n_clusters)

#     fig = plt.figure(figsize=(14, 4 + 2 * n_clusters))

#     # Top: connected scatter plot of actions over lake concentration
#     ax1 = plt.subplot2grid((n_clusters + 1, 2), (0, 0), colspan=2)
#     for k in range(n_clusters):
#         idxs = np.where(labels == k)[0]
#         sorted_idxs = idxs[np.argsort(lake_states[idxs])]
#         ax1.plot(lake_states[sorted_idxs], actions[sorted_idxs],
#                  color=colors(k), marker='o', label=f"Cluster {k}")
#     ax1.set_xlabel("Lake Concentration (X)")
#     ax1.set_ylabel("Action")
#     ax1.set_title("Action Trajectory by Cluster")
#     ax1.legend()

#     # Subplots for SHAP bars per cluster
#     for k in range(n_clusters):
#         ax_bar = plt.subplot2grid((n_clusters + 1, 2), (k + 1, 0))
#         idxs = np.where(labels == k)[0]
#         avg_shap = shap_vectors[idxs].mean(axis=0)
#         ax_bar.bar(objectives, avg_shap, color=colors(k))
#         ax_bar.set_ylim(0, 1)
#         ax_bar.set_ylabel("Mean SHAP")
#         ax_bar.set_title(f"Cluster {k} SHAP Contributions")

#     plt.tight_layout()
#     plt.suptitle(f"Intra-Policy Behavior & Attribution (Policy {policy_index})", fontsize=14, y=1.02)
#     plt.subplots_adjust(top=0.92)
#     plt.show()

# def plot_clusters(actions, shap_vectors, labels, policy_index):
#     """visualizes actions connected by lake concentration, colored by cluster, with separate SHAP bars and adaptive y-axis"""
#     import matplotlib.cm as cm

#     n_clusters = np.max(labels) + 1
#     objectives = np.array(OBJECTIVES)
#     lake_states = np.linspace(0.0, 2.0, len(actions))

#     colors = cm.get_cmap('tab10', n_clusters)

#     fig = plt.figure(figsize=(14, 4 + 2 * n_clusters))

#     # top: connected action trajectory
#     ax1 = plt.subplot2grid((n_clusters + 1, 2), (0, 0), colspan=2)
#     for i in range(len(actions) - 1):
#         x_pair = lake_states[i:i+2]
#         y_pair = actions[i:i+2]
#         cluster_color = colors(labels[i])
#         ax1.plot(x_pair, y_pair, color=cluster_color, linewidth=1.8)
#         ax1.scatter(lake_states[i], actions[i], color=cluster_color, edgecolor='k', zorder=3)
#     ax1.scatter(lake_states[-1], actions[-1], color=colors(labels[-1]), edgecolor='k', zorder=3)
#     ax1.set_xlabel("Lake Concentration (X)")
#     ax1.set_ylabel("Action")
#     ax1.set_title("Policy Action Trajectory (colored by intra-cluster label)")

#     # SHAP bar plots per cluster
#     for k in range(n_clusters):
#         ax_bar = plt.subplot2grid((n_clusters + 1, 2), (k + 1, 0))
#         idxs = np.where(labels == k)[0]
#         avg_shap = shap_vectors[idxs].mean(axis=0)
#         ax_bar.bar(objectives, avg_shap, color=colors(k))
#         max_y = max(avg_shap.max(), 1e-2)  # avoid flat axis
#         ax_bar.set_ylim(0, max_y * 1.2)
#         ax_bar.set_ylabel("Mean SHAP")
#         ax_bar.set_title(f"Cluster {k} Attribution")

#         # Add value labels
#         for i, val in enumerate(avg_shap):
#             ax_bar.text(i, val + 0.01 * max_y, f"{val:.2f}", ha='center', va='bottom', fontsize=8)

#     plt.tight_layout()
#     plt.suptitle(f"Intra-Policy Clustering: Actions + SHAP (Policy {policy_index})", fontsize=14, y=1.03)
#     plt.subplots_adjust(top=0.91)
#     plt.show()

def plot_clusters(actions, shap_vectors, labels, policy_index):
    """visualizes actions connected by lake concentration, colored by cluster, with centered SHAP bars"""
    import matplotlib.pyplot as plt

    n_clusters = np.max(labels) + 1
    objectives = np.array(OBJECTIVES)
    lake_states = np.linspace(0.0, 2.0, len(actions))

    color_list = ['#e41a1c', '#4daf4a', '#377eb8', '#ff7f00']
    colors = lambda k: color_list[k % len(color_list)]

    fig = plt.figure(figsize=(14, 4 + 2.5 * n_clusters))

    # Top plot: action trajectory
    ax1 = plt.subplot2grid((n_clusters + 1, 2), (0, 0), colspan=2)
    for i in range(len(actions) - 1):
        x_pair = lake_states[i:i+2]
        y_pair = actions[i:i+2]
        cluster_color = colors(labels[i])
        ax1.plot(x_pair, y_pair, color=cluster_color, linewidth=1.8)
        ax1.scatter(lake_states[i], actions[i], color=cluster_color, edgecolor='k', zorder=3)
    ax1.scatter(lake_states[-1], actions[-1], color=colors(labels[-1]), edgecolor='k', zorder=3)
    ax1.set_xlabel("Concentration (X)")
    ax1.set_ylabel("Action")
    ax1.set_title("Policy Action Trajectory (colored by intra-cluster label)")

    # SHAP bar plots
    for k in range(n_clusters):
        ax_bar = plt.subplot2grid((n_clusters + 1, 2), (k + 1, 0))
        idxs = np.where(labels == k)[0]
        avg_shap = shap_vectors[idxs].mean(axis=0)
        bars = ax_bar.bar(objectives, avg_shap, color=colors(k))

        # Add value labels above bars
        for i, bar in enumerate(bars):
            height = bar.get_height()
            if height >= 0:
                ax_bar.text(bar.get_x() + bar.get_width() / 2, height + 0.01, f"{height:.2f}",
                            ha='center', va='bottom', fontsize=8)
            else:
                ax_bar.text(bar.get_x() + bar.get_width() / 2, height - 0.01, f"{height:.2f}",
                            ha='center', va='top', fontsize=8)

        y_margin = max(abs(avg_shap.min()), abs(avg_shap.max())) * 1.2
        ax_bar.set_ylim(-y_margin, y_margin)
        ax_bar.axhline(0, color='gray', linewidth=0.8)
        ax_bar.set_ylabel("Mean SHAP")
        ax_bar.set_title(f"Cluster {k} Attribution")
        ax_bar.set_xticklabels(objectives, rotation=10)

    plt.tight_layout(h_pad=3.5)
    plt.suptitle(f"Intra-Policy Clustering: Actions + SHAP (Policy {policy_index})", fontsize=14, y=1.03)
    plt.subplots_adjust(top=0.92)
    plt.show()


# example usage
if __name__ == "__main__":
    run_intra_clustering(policy_index=42, n_clusters=4, alpha=0.5)
