import argparse
import pickle
import random

import numpy as np
from pyhocon import ConfigFactory
from sklearn.cluster import KMeans


class KMeansClustering:
    def __init__(self, n_clusters, random_state):
        self.n_clusters = n_clusters
        self.random_state = random_state
        random.seed(self.random_state)
        np.random.seed(self.random_state)

    def kmeans(self, X):
        print(f"Clustering with {self.n_clusters} clusters...")
        clustering_model = KMeans(n_clusters=self.n_clusters, random_state=self.random_state, init='k-means++', n_init=10)
        clustering_model.fit(X)
        cluster_assignment = clustering_model.labels_
        cluster_centroids = clustering_model.cluster_centers_

        with open(config["clusters_path"] + f"clusters_{self.n_clusters}_0.0.pickle", 'wb') as f:
            pickle.dump({
                "number_cluster": self.n_clusters,
                "labels": cluster_assignment,
                "cluster_centers": cluster_centroids
            }, f, protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='KMeans Clustering')
    parser.add_argument('-c', metavar='CONF', default='base', help='configuration (see config.conf)')
    parser.add_argument('-k', metavar='N_CLUSTERS', default=5, type=int, help='number of clusters')
    args = parser.parse_args()
    config = ConfigFactory.parse_file('./config.conf')[args.c]

    print("N_CLUSTERS: " + str(args.k), flush=True)

    print("Loading data for clustering...", flush=True)

    # Load data
    with open(config["cluster_embs_path"], 'rb') as f:
        data = pickle.load(f)

    model = KMeansClustering(args.k, config['seed'])
    model.kmeans(data['embs'])