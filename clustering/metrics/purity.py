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

def calculate_cluster_purity(chain_data, clusters, clustering_data, top_k=False):
    chain_group_roles = chain_data["chain_group_roles"]
    if top_k:
        chain_group_roles_top_k = {}
        for cluster_idx, cluster in clusters.items():
            for idx in cluster:
                chain_group_roles_top_k[idx] = chain_group_roles[idx]
        chain_group_roles = chain_group_roles_top_k

    # Extract all unique keys from the dataset
    all_keys = set()
    for dp in chain_group_roles.values():
        all_keys.update(dp.keys())
    all_keys = sorted(all_keys)
    
    print(f"Found keys: {all_keys}")

    # Group data points by cluster
    cluster_data = {}
    for chain_id, data_point in chain_group_roles.items():
        if top_k:
            cluster_label = clustering_data['labels'][chain_id]
        else:
            cluster_label = clustering_data['labels'][chain_id]
        
        if cluster_label not in cluster_data:
            cluster_data[cluster_label] = []
        cluster_data[cluster_label].append(data_point)

    # Calculate purity for each key
    key_purities = {}
    
    for key in all_keys:
        cluster_purities_for_key = []
        
        for cluster_label, cluster_points in cluster_data.items():
            # Get all values for this key in this cluster
            key_values = []
            for point in cluster_points:
                if key in point:
                    key_values.append(point[key])
            
            if key_values:
                # Find the most frequent value
                from collections import Counter
                value_counts = Counter(key_values)
                most_common_count = value_counts.most_common(1)[0][1]
                cluster_purity = most_common_count / len(key_values)
                cluster_purities_for_key.append(cluster_purity)
        
        # Average purity across all clusters for this key
        if cluster_purities_for_key:
            key_purity = np.mean(cluster_purities_for_key)
            key_purities[key] = key_purity
    
    # Calculate overall average purity
    overall_purity = np.mean(list(key_purities.values()))
    
    # Print results
    print(f"Purity per key:")
    for key, purity in key_purities.items():
        print(f"  {key}: {purity * 100:.2f}")
    print(f"Overall purity: {overall_purity * 100:.2f}")
    
    return key_purities, overall_purity


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='KMeans Clustering')
    parser.add_argument('-c', metavar='CONF', default='base', help='configuration (see config.conf)')

    args = parser.parse_args()
    config = ConfigFactory.parse_file('./config.conf')[args.c]

    chain_data, clustering_data = load_data(config)
    clusters, clusters_top_k = generate_clusters(clustering_data, top_k=100)
    calculate_cluster_purity(chain_data, clusters_top_k, clustering_data, top_k=True)