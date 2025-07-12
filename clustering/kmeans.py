import argparse
import os
import pickle
import random

import numpy as np
import torch
from pyhocon import ConfigFactory
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans

from clustering.metrics.purity import Purity
from models.regression import RegressionModel

# Disable tokenizer parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"


class KMeansClustering:
    def __init__(self, n_clusters, config):
        self.config = config
        self.n_clusters = n_clusters
        self.random_state = self.config["seed"]
        random.seed(self.random_state)
        np.random.seed(self.random_state)
        torch.manual_seed(self.config["seed"])
        torch.random.manual_seed(self.config["seed"])
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.sbert_model = SentenceTransformer(self.config["cluster_model"])
        self.sbert_model.to(self.device)

    def kmeans(self, X):
        print(f"Clustering with {self.n_clusters} clusters...", flush=True)
        clustering_model = KMeans(n_clusters=self.n_clusters, random_state=self.random_state, init='k-means++', n_init=10)
        clustering_model.fit(X)
        cluster_assignment = clustering_model.labels_
        cluster_centroids = clustering_model.cluster_centers_

        # Create clustering data for purity and regression
        clustering_data = {
            "number_cluster": self.n_clusters,
            "embeddings": X,
            "labels": cluster_assignment,
            "cluster_centers": cluster_centroids
        }

        # Compute and print purity results
        print("\n=== Purity Results ===", flush=True)
        purity_calculator = Purity(self.config["processed_chains_path"], clustering_data)
        purity_calculator.compute_purity()
        purity_calculator.print_results()
        print("======================\n", flush=True)

        # Run regression model after purity computation
        print("\n=== Regression Results ===", flush=True)
        regression_model = RegressionModel(self.config)
        data = regression_model.create_dataset(self.config, clustering_data)
        test_accuracy, f1_score = regression_model.regression(self.config, data)
        print("==========================\n", flush=True)

        # Save clustering results
        with open(self.config["clusters_path"] + f"clusters_{self.n_clusters}_0.0.pickle", 'wb') as f:
            pickle.dump(clustering_data, f, protocol=pickle.HIGHEST_PROTOCOL)

    def compute_embeddings(self, chain_sents):
        print("Computing embeddings...", flush=True)
        embeddings = self.sbert_model.encode(
            chain_sents, batch_size=32, show_progress_bar=True, normalize_embeddings=True
        )
        return embeddings

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='KMeans Clustering')
    parser.add_argument('-c', metavar='CONF', default='base', help='configuration (see config.conf)')
    parser.add_argument('-k', metavar='N_CLUSTERS', default=5, type=int, help='number of clusters')
    args = parser.parse_args()
    config = ConfigFactory.parse_file('./config.conf')[args.c]

    print("Running KMeans Clustering with configuration with the following parameters:" flush=True)
    print("N_CLUSTERS: " + str(args.k), flush=True)

    print("Loading data for clustering...", flush=True)

    # Load data
    with open(config["processed_chains_path"], 'rb') as f:
        data = pickle.load(f)
    chain_sents = data['chain_sents']

    model = KMeansClustering(args.k, config)
    embeddings = model.compute_embeddings(chain_sents)
    model.kmeans(embeddings)