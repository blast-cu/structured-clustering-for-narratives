import argparse
import pickle

import numpy as np
from pyhocon import ConfigFactory
from sklearn.metrics import confusion_matrix

class Purity:
    def __init__(self, processed_chains_path, clustering_data):
        with open(processed_chains_path, "rb") as f:
            self.chain_data = pickle.load(f)

        self.clustering_data = clustering_data

        self.results = {
            "25": {
                "exact_match_purity": {"score": None},
                "role_purity": {
                    "key_purities": None,
                    "overall_purity": None
                }
            },
            "100": {
                "exact_match_purity": {"score": None},
                "role_purity": {
                    "key_purities": None,
                    "overall_purity": None
                }
            }
        }

    def generate_clusters(self, top_k=25):
        # create a dictionary to hold clusters, populate keys with number of clusters
        clusters = {i: [] for i in range(self.clustering_data["number_cluster"])}
        for idx, label in enumerate(self.clustering_data["labels"]):
            clusters[label].append(idx)

        clusters_top_k = {}
        for cluster_idx, cluster in clusters.items():
            # Get embeddings only for points in this cluster
            cluster_embs = self.clustering_data["embeddings"][cluster]
            # Get centroid for this specific cluster
            centroid_emb = self.clustering_data["cluster_centers"][cluster_idx]
            # Calculate distances from cluster points to their centroid
            distances = np.linalg.norm(cluster_embs - centroid_emb, axis=1)

            k = max(1, int(len(cluster) * top_k / 100))
            closest_indices_local = np.argsort(distances)[:k]
            closest_indices_global = [cluster[idx] for idx in closest_indices_local]
            clusters_top_k[cluster_idx] = closest_indices_global
        return clusters, clusters_top_k

    def get_exact_match_purity(self, clusters, top_k=False):
        chain_group_roles = self.chain_data["chain_group_roles"]
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
                original_cluster_label = self.clustering_data['labels'][chain_id]
                predicted_labels.append(original_cluster_label)
        else:
            predicted_labels = self.clustering_data['labels']

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

        print(f"Exact Match purity: {cluster_purity * 100:.2f}")

        return cluster_purity

    def calculate_cluster_purity(self, clusters, top_k=False):
        chain_group_roles = self. chain_data["chain_group_roles"]
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
                cluster_label = self.clustering_data['labels'][chain_id]
            else:
                cluster_label = self.clustering_data['labels'][chain_id]

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
        print(f"Overall role purity: {overall_purity * 100:.2f}")
        
        return key_purities, overall_purity

    def compute_purity(self):
        for k in [25, 100]:
            print(f"Computing purity for top_k={k}...")

            _, clusters_top_k = self.generate_clusters(top_k=k)
            exact_match_purity = self.get_exact_match_purity(clusters_top_k, top_k=True)
            key_purities, overall_purity = self.calculate_cluster_purity(clusters_top_k, top_k=True)

            self.results[str(k)]['exact_match_purity']['score'] = exact_match_purity
            self.results[str(k)]['role_purity']['key_purities'] = key_purities
            self.results[str(k)]['role_purity']['overall_purity'] = overall_purity

    def print_results(self):
        values = []
        
        # exact_match_purity for 25 and 100
        values.append(f"{self.results['25']['exact_match_purity']['score'] * 100:.2f}")
        values.append(f"{self.results['100']['exact_match_purity']['score'] * 100:.2f}")
        
        # overall_purity from role_purity for 25 and 100
        values.append(f"{self.results['25']['role_purity']['overall_purity'] * 100:.2f}")
        values.append(f"{self.results['100']['role_purity']['overall_purity'] * 100:.2f}")
        
        # key_purities - get all keys and sort them
        key_purities_25 = self.results['25']['role_purity']['key_purities']
        key_purities_100 = self.results['100']['role_purity']['key_purities']
        
        if key_purities_25 and key_purities_100:
            all_keys = sorted(set(key_purities_25.keys()) | set(key_purities_100.keys()))
            for key in all_keys:
                val_25 = key_purities_25.get(key)
                val_100 = key_purities_100.get(key)
                values.append(f"{val_25 * 100:.2f}" if val_25 is not None else 'N/A')
                values.append(f"{val_100 * 100:.2f}" if val_100 is not None else 'N/A')
        
        print('\t'.join(values))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='KMeans Clustering')
    parser.add_argument('-c', metavar='CONF', default='base', help='configuration (see config.conf)')

    args = parser.parse_args()
    config = ConfigFactory.parse_file('./config.conf')[args.c]

    with open(config["clusters_path"] + "clusters_250_0.5_.pickle", "rb") as f:
        clustering_data = pickle.load(f)

    purity = Purity(config["processed_chains_path"], clustering_data)

    purity.compute_purity()
    purity.print_results()