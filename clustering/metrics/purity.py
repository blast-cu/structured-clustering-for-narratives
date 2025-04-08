import json
import pickle

import numpy as np
from numpy.ma.extras import average
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm


def prepare_data(chain_annotations):
    role_labels, stance_labels = [], []
    for k,v in chain_annotations.items():
        if v['annotation'] is None:
            role_labels.append('None')
            stance_labels.append('None')
        else:
            if v['annotation']['role'] is not None:
                role_labels.append(v['annotation']['role'])
            else:
                role_labels.append('None')

            if v['annotation']['stance'] is not None:
                stance_labels.append(v['annotation']['stance'])
            else:
                stance_labels.append('None')
    return role_labels, stance_labels


def prepare_data_top_k(top_k, embeddings, centroids, cluster_labels, chain_annotations):
    clusters = {}
    for index, label in enumerate(cluster_labels):
        if label not in clusters:
            clusters[label] = [index]
        else:
            clusters[label].append(index)

    clusters_top_k = {}
    for cluster_idx, cluster in clusters.items():
        cluster_embs = embeddings[cluster]
        centroid_embs = centroids[cluster_idx]
        distances = np.linalg.norm(cluster_embs - centroid_embs, axis=1)

        k = max(1, int(len(cluster) * top_k / 100))
        closest_indices_local = np.argsort(distances)[:k]
        closest_indices_global = [cluster[idx] for idx in closest_indices_local]
        clusters_top_k[cluster_idx] = closest_indices_global

    cluster_labels_top_k = []
    role_labels_top_k = []
    stance_labels_top_k = []

    for cluster_idx, cluster in clusters_top_k.items():
        for idx in cluster:
            chain = chain_annotations[idx]
            if chain['annotation'] is None:
                role_labels_top_k.append('None')
                stance_labels_top_k.append('None')
            else:
                if chain['annotation']['role'] is not None:
                    role_labels_top_k.append(chain['annotation']['role'])
                else:
                    role_labels_top_k.append('None')

                if chain['annotation']['stance'] is not None:
                    stance_labels_top_k.append(chain['annotation']['stance'])
                else:
                    stance_labels_top_k.append('None')
            cluster_labels_top_k.append(cluster_idx)

    return role_labels_top_k, stance_labels_top_k, cluster_labels_top_k


def purity(true_labels, predicted_clusters):
    # Convert string labels to numeric if needed
    le = LabelEncoder()
    true_labels_encoded = le.fit_transform(true_labels)

    # Create the confusion matrix
    cm = confusion_matrix(true_labels_encoded, predicted_clusters)

    # Find the maximum number of data points of any class in each cluster
    cluster_purity = np.sum(np.max(cm, axis=0)) / np.sum(cm)

    return cluster_purity


if __name__ == '__main__':
    # Load data
    with open('../../data/immigration/embeddings.pickle', 'rb') as f:
        embeddings = pickle.load(f)

    with open('../../data/immigration/dcc_grid_search_results.json', 'r') as f:
        results = json.load(f)

    updated_results = []
    top_k = None

    for result in tqdm(results):
        num_clusters = result['num_clusters']
        constraint_weight = result['constraint_weight']

        with open(f'../../data/immigration/dcc_clusters_{num_clusters}'
                  f'_{constraint_weight}.pickle', 'rb') as f:
            clusters = pickle.load(f)

        if top_k is not None:
            role_labels, stance_labels, cluster_labels = prepare_data_top_k(top_k, embeddings['embs'],
                                                                            clusters['cluster_centers'],
                                                                            clusters['labels'],
                                                                            embeddings['chains'])
        else:
            role_labels, stance_labels = prepare_data(embeddings['chains'])
            cluster_labels = clusters['labels']

        assert len(role_labels) == len(cluster_labels)
        assert len(stance_labels) == len(cluster_labels)

        role_purity = round(purity(role_labels, cluster_labels), 3)
        stance_purity = round(purity(stance_labels, cluster_labels), 3)
        average_purity = round(average([role_purity, stance_purity]), 3)

        updated_results.append({
            'num_clusters': num_clusters,
            'constraint_weight': constraint_weight,
            'accuracy': result['accuracy'],
            'macro_f1_score': result['macro_f1_score'],
            'role_purity': role_purity,
            'stance_purity': stance_purity,
            'average_purity': average_purity
        })

    with open('../../data/immigration/dcc_grid_search_results_updated.json', 'w') as f:
        json.dump(updated_results, f, indent=4)