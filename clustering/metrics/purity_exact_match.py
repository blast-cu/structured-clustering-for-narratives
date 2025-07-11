import argparse
import pickle

import numpy as np
from pyhocon import ConfigFactory
from sklearn.metrics import confusion_matrix


def load_data(config):
    with open(config["processed_chains_path"], "rb") as f:
        chain_data = pickle.load(f)

    # constraints = {}
    # with open(config["constraints_path"], "rb") as f:
    #     while True:
    #         try:
    #             batch = pickle.load(f)
    #             # Convert batch (list of tuples) to dict entries
    #             for k1, k2 in batch:
    #                 constraints[(k1, k2)] = 1
    #         except EOFError:
    #             break

    with open(config["clusters_path"] + "em_clusters_250_0.01_None_None_scikit_kmeans.pickle", "rb") as f:
        clustering_data = pickle.load(f)

    return chain_data, clustering_data


def generate_clusters(clustering_data, top_k=25):
    # create a dictionary to hold clusters, populate keys with number of clusters
    clusters = {i: [] for i in range(clustering_data["number_cluster"])}
    for idx, label in enumerate(clustering_data["labels"]):
        clusters[label].append(idx)

    clusters_top_k = {}
    for cluster_idx, cluster in clusters.items():
        # Get embeddings only for points in this cluster
        cluster_embs = clustering_data["embeddings"][cluster]
        # Get centroid for this specific cluster
        centroid_emb = clustering_data["cluster_centers"][cluster_idx]
        # Calculate distances from cluster points to their centroid
        distances = np.linalg.norm(cluster_embs - centroid_emb, axis=1)

        k = max(1, int(len(cluster) * top_k / 100))
        closest_indices_local = np.argsort(distances)[:k]
        closest_indices_global = [cluster[idx] for idx in closest_indices_local]
        clusters_top_k[cluster_idx] = closest_indices_global
    return clusters, clusters_top_k

def get_exact_match_purity(chain_data, clusters, clustering_data, top_k=False):
    chain_group_roles = chain_data["chain_group_roles"]
    if top_k:
        chain_group_roles_top_k = {}
        for cluster_idx, cluster in clusters.items():
            for idx in cluster:
                chain_group_roles_top_k[idx] = chain_group_roles[idx]
        chain_group_roles = chain_group_roles_top_k

    # Convert to sorted tuples for canonical representation
    def canonicalize_data_point(dp):
        # Handle nested dict structure: {group: {'role': role, 'stance': stance}}
        items = []
        for group, info in dp.items():
            if isinstance(info, dict):
                # Create a canonical representation of the nested dict
                canonical_info = tuple(sorted(info.items()))
                items.append((group, canonical_info))
            else:
                items.append((group, info))
        return tuple(sorted(items))

    # Find unique combinations and create class mapping
    unique_classes = list(set(canonicalize_data_point(dp) for dp in chain_group_roles.values()))
    class_mapping = {pattern: i for i, pattern in enumerate(unique_classes)}

    # Create true labels (ground truth classes)
    true_labels = []
    for chain_id in sorted(chain_group_roles.keys()):
        canonical = canonicalize_data_point(chain_group_roles[chain_id])
        class_label = class_mapping[canonical]
        true_labels.append(class_label)

    # Get predicted labels (cluster assignments) - ensure same ordering
    if top_k:
        # Generate predicted labels only for the selected top_k indices
        predicted_labels = []
        for chain_id in sorted(chain_group_roles.keys()):
            # Find which cluster this chain_id belongs to
            original_cluster_label = clustering_data['labels'][chain_id]
            predicted_labels.append(original_cluster_label)
    else:
        predicted_labels = clustering_data['labels']
    
    # Verify same number of data points
    if len(true_labels) != len(predicted_labels):
        raise ValueError(f"Mismatch: {len(true_labels)} true labels vs {len(predicted_labels)} predicted labels")

    # Create the confusion matrix (rows=true classes, cols=predicted clusters)
    cm = confusion_matrix(true_labels, predicted_labels)
    
    print(f"Confusion matrix shape: {cm.shape}")
    print(f"Number of unique true classes: {len(unique_classes)}")
    print(f"Number of clusters: {len(set(predicted_labels))}")

    # Calculate purity: for each cluster, take the most frequent true class
    cluster_purity = np.sum(np.max(cm, axis=0)) / np.sum(cm)
    
    print(f"Cluster purity: {cluster_purity * 100:.2f}")
    return cluster_purity


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='KMeans Clustering')
    parser.add_argument('-c', metavar='CONF', default='base', help='configuration (see config.conf)')

    args = parser.parse_args()
    config = ConfigFactory.parse_file('./config.conf')[args.c]

    chain_data, clustering_data = load_data(config)
    clusters, clusters_top_k = generate_clusters(clustering_data, top_k=100)
    get_exact_match_purity(chain_data, clusters_top_k, clustering_data, top_k=True)