import argparse
import os
import pickle
import random

import numpy as np
import torch
from pyhocon import ConfigFactory
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans

from clustering.metrics.cluster_metrics import ClusterMetrics

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def flatten_verbalizations(data):
    """Flatten causal_verbalizations.pickle into processed_chains and chain_sents.

    causal_verbalizations structure:
        {doc_id: {'text': str, 'event_pairs': [...], 'event_chains': {idx: {'event_chain': str, 'chain_text': str}}}}
    """
    chain_idx = 0
    processed_chains = {}
    chain_sents = []
    for doc_id, doc in data.items():
        if not doc.get("event_chains"):
            continue
        for chain_obj in doc["event_chains"].values():
            if chain_obj.get("chain_text") is None:
                continue
            processed_chains[chain_idx] = {
                "event_chain": chain_obj["event_chain"],
                "chain_text": chain_obj["chain_text"],
                "doc_id": doc_id,
            }
            chain_sents.append(chain_obj["chain_text"])
            chain_idx += 1
    print(f"Flattened {len(processed_chains)} event chains from {len(data)} documents.", flush=True)
    return processed_chains, chain_sents


class RedditKMeansClustering:
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

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.sbert_model = SentenceTransformer(self.config["cluster_model"])
        self.sbert_model.to(self.device)

    def compute_embeddings(self, chain_sents):
        print("Computing embeddings...", flush=True)
        embeddings = self.sbert_model.encode(
            chain_sents, batch_size=32, show_progress_bar=True, normalize_embeddings=True
        )
        return embeddings

    def kmeans(self, X, clusters_path):
        print(f"Clustering with {self.n_clusters} clusters...", flush=True)
        clustering_model = KMeans(
            n_clusters=self.n_clusters,
            random_state=self.random_state,
            init="k-means++",
            n_init=10,
        )
        clustering_model.fit(X)

        clustering_data = {
            "number_cluster": self.n_clusters,
            "embeddings": X,
            "labels": clustering_model.labels_,
            "cluster_centers": clustering_model.cluster_centers_,
        }

        print("\n=== Cluster Quality Metrics ===", flush=True)
        cluster_metrics = ClusterMetrics(clustering_data)
        cluster_metrics.print_results()
        print("==============================\n", flush=True)

        os.makedirs(clusters_path, exist_ok=True)
        out_path = os.path.join(clusters_path, f"clusters_{self.n_clusters}_0.0.pickle")
        with open(out_path, "wb") as f:
            pickle.dump(clustering_data, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"Saved clustering results to {out_path}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reddit KMeans Clustering")
    parser.add_argument("-c", metavar="CONF", default="reddit_parkinsons",
                        help="configuration (see config.conf)")
    parser.add_argument("-k", metavar="N_CLUSTERS", default=5, type=int,
                        help="number of clusters")
    args = parser.parse_args()
    config = ConfigFactory.parse_file("./config.conf")[args.c]

    print(f"Running Reddit KMeans with N_CLUSTERS={args.k}", flush=True)

    print("Loading causal verbalizations...", flush=True)
    with open(config["causal_verbalizations_path"], "rb") as f:
        data = pickle.load(f)

    processed_chains, chain_sents = flatten_verbalizations(data)

    clusters_path = os.path.join(config["output_path"], "clustering")

    model = RedditKMeansClustering(args.k, config)
    embeddings = model.compute_embeddings(chain_sents)
    model.kmeans(embeddings, clusters_path)
