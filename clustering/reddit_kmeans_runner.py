import argparse
import os
import pickle
import random

import numpy as np
import torch
from pyhocon import ConfigFactory
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans

import sys
import annotation.schemas as _schemas
sys.modules['schemas'] = _schemas

from clustering.metrics.cluster_metrics import ClusterMetrics
from clustering.reddit_kmeans import flatten_verbalizations

os.environ["TOKENIZERS_PARALLELISM"] = "false"

K_MIN = 25
K_MAX = 250
K_STEP = 25


def set_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.random.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def compute_embeddings(chain_sents, model_name, device):
    print("Computing embeddings...", flush=True)
    sbert_model = SentenceTransformer(model_name)
    sbert_model.to(device)
    embeddings = sbert_model.encode(
        chain_sents, batch_size=32, show_progress_bar=True, normalize_embeddings=True
    )
    return embeddings


def run_kmeans(X, k, seed):
    clustering_model = KMeans(n_clusters=k, random_state=seed, init="k-means++", n_init=10)
    clustering_model.fit(X)
    return {
        "number_cluster": k,
        "embeddings": X,
        "labels": clustering_model.labels_,
        "cluster_centers": clustering_model.cluster_centers_,
    }


def print_summary(all_metrics):
    print("\n" + "=" * 65, flush=True)
    print(f"{'k':>6}  {'Silhouette':>12}  {'Calinski-Harabasz':>18}  {'WCSS':>12}", flush=True)
    print("-" * 65, flush=True)
    for k, m in sorted(all_metrics.items()):
        print(f"{k:>6}  {m['silhouette']:>12.4f}  {m['calinski_harabasz']:>18.2f}  {m['cohesion']:>12.2f}", flush=True)
    print("=" * 65, flush=True)


def select_best_k(all_metrics):
    # Calinski-Harabasz is recommended for k-selection (peak value).
    # Break ties with silhouette score.
    best_k = max(all_metrics, key=lambda k: (all_metrics[k]["calinski_harabasz"], all_metrics[k]["silhouette"]))
    return best_k


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reddit KMeans Runner (k=25..250, step 25)")
    parser.add_argument("-c", metavar="CONF", default="reddit_parkinsons",
                        help="configuration (see config.conf)")
    args = parser.parse_args()
    config = ConfigFactory.parse_file("./config.conf")[args.c]

    seed = config["seed"]
    set_seeds(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    clusters_path = os.path.join(config["output_path"], "clustering")
    os.makedirs(clusters_path, exist_ok=True)

    print("Loading causal verbalizations...", flush=True)
    with open(config["causal_verbalizations_path"], "rb") as f:
        data = pickle.load(f)

    _, chain_sents = flatten_verbalizations(data)

    embeddings = compute_embeddings(chain_sents, config["cluster_model"], device)

    all_metrics = {}

    for k in range(K_MIN, K_MAX + 1, K_STEP):
        print(f"\n{'=' * 40}", flush=True)
        print(f"Clustering with k={k}...", flush=True)
        print(f"{'=' * 40}", flush=True)

        clustering_data = run_kmeans(embeddings, k, seed)

        metrics = ClusterMetrics(clustering_data).compute_all_metrics()
        all_metrics[k] = metrics
        print(f"  Silhouette:          {metrics['silhouette']:.4f}", flush=True)
        print(f"  Calinski-Harabasz:   {metrics['calinski_harabasz']:.2f}", flush=True)
        print(f"  WCSS:                {metrics['cohesion']:.2f}", flush=True)

        out_path = os.path.join(clusters_path, f"clusters_{k}_0.0.pickle")
        with open(out_path, "wb") as f:
            pickle.dump(clustering_data, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"  Saved -> {out_path}", flush=True)

    print_summary(all_metrics)

    best_k = select_best_k(all_metrics)
    print(f"\nBest k = {best_k}  (Calinski-Harabasz: {all_metrics[best_k]['calinski_harabasz']:.2f}, "
          f"Silhouette: {all_metrics[best_k]['silhouette']:.4f})", flush=True)

    summary = {"all_metrics": all_metrics, "best_k": best_k}
    summary_path = os.path.join(clusters_path, "kmeans_sweep_summary.pickle")
    with open(summary_path, "wb") as f:
        pickle.dump(summary, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Sweep summary saved -> {summary_path}", flush=True)
