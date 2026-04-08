import argparse
import os
import pickle
import sys

import numpy as np
from pyhocon import ConfigFactory

import annotation.schemas as _schemas
sys.modules['schemas'] = _schemas

from annotation.cluster_analysis import ClusterAnalyzer
from annotation.schemas import ClusterTheme
from clustering.reddit_kmeans import flatten_verbalizations


class RedditClusterAnalyzer(ClusterAnalyzer):
    def __init__(self, config, host, port, domain, chains_per_cluster):
        super().__init__(config, host, port, domain, chains_per_cluster)

        with open("./annotation/prompts/cluster_analysis/reddit_system_prompt.md", 'r', encoding='utf-8') as f:
            self.reasoning_system_prompt = f.read()

    def create_dataset(self, clustering_data, chain_sents, doc_ids):
        """Build cluster dict from Reddit verbalization data.

        Args:
            clustering_data: dict with 'embeddings', 'labels', 'cluster_centers', 'number_cluster'
            chain_sents:     flat list of verbalized chain strings (index-aligned with labels)
            doc_ids:         flat list of doc_id strings (index-aligned with labels)
        """
        chains_to_keep = self.filter_chains_by_centroid_proximity(clustering_data)
        clusters = {}
        for idx, label in enumerate(clustering_data['labels']):
            if idx not in chains_to_keep:
                continue
            if label not in clusters:
                clusters[label] = {"chains": [], "theme": ""}
            clusters[label]['chains'].append({
                "chain_idx": idx,
                "chain_text": chain_sents[idx],
                "doc_id":     doc_ids[idx],
            })
        return clusters

    def process_cluster(self, cluster):
        items_to_sample = min(self.chains_per_cluster, len(cluster['chains']))
        sampled_items = self.sample_from_different_documents(cluster['chains'], items_to_sample)

        sentences = ""
        for i, item in enumerate(sampled_items, 1):
            sentences += f"{i}. {item['chain_text']}\n\n"

        cluster['theme'] = self.annotate(self.domain, sentences)
        return cluster

    def annotate(self, domain, sentences):
        reasoning_user_prompt = (
            f"DOMAIN: \"{domain}\"\n"
            f"CLUSTER SENTENCES:\n{sentences}"
        )

        max_retries = 5
        retry_count = 0
        while retry_count < max_retries:
            try:
                reasoning_model_response = self.reasoning_model.chat(
                    self.reasoning_system_prompt,
                    reasoning_user_prompt,
                    think=True,
                    num_ctx=self.num_ctx)

                structured_response = self.output_model.chat(
                    self.structured_output_system_prompt,
                    reasoning_model_response,
                    think=False,
                    repeat_penalty=True,
                    format=ClusterTheme.model_json_schema(),
                    num_ctx=self.num_ctx)

                try:
                    response = ClusterTheme.model_validate_json(structured_response)
                    return response.model_dump()
                except Exception as e:
                    print(f"Exception: {e}", flush=True)
                    print("Invalid response. Retrying.", flush=True)
                    retry_count += 1
            except Exception as e:
                print(f"Exception: {e}", flush=True)
                print("Ollama Error. Retrying.", flush=True)
                retry_count += 1
        return None


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Reddit Cluster Analyzer')
    parser.add_argument('-c', metavar='CONF', default='reddit_parkinsons',
                        help='configuration (see config.conf)')
    parser.add_argument('--host', metavar='HOST', required=True)
    parser.add_argument('--port', default=9999, metavar='PORT')
    parser.add_argument('--domain', metavar='DOMAIN', required=True,
                        help='Health domain label, e.g. "Parkinson\'s Disease" or "Long Covid"')
    parser.add_argument('--clusters_file', metavar='CLUSTERS_FILE', required=True,
                        help='Path to a specific clusters_N_0.0.pickle to analyze')
    parser.add_argument('--chains_per_cluster', type=int, default=25, metavar='CHAINS_PER_CLUSTER')
    parser.add_argument('--workers', type=int, default=4, metavar='WORKERS')
    parser.add_argument('--save_interval', type=int, default=10, metavar='SAVE_INTERVAL')
    parser.add_argument('--sequential', action='store_true')
    args = parser.parse_args()

    config = ConfigFactory.parse_file('./config.conf')[args.c]

    # Load clustering results
    with open(args.clusters_file, 'rb') as f:
        clustering_data = pickle.load(f)

    # Load verbalizations
    with open(config['causal_verbalizations_path'], 'rb') as f:
        raw_data = pickle.load(f)
    processed_chains, chain_sents = flatten_verbalizations(raw_data)
    doc_ids = [processed_chains[i]['doc_id'] for i in range(len(processed_chains))]

    # Derive output path alongside the clusters file
    clusters_dir = os.path.dirname(args.clusters_file)
    k = clustering_data['number_cluster']
    save_path = os.path.join(clusters_dir, f'cluster_analysis_{k}.pickle')

    # Patch config with the derived save path
    config = dict(config)
    config['cluster_analysis_path'] = save_path

    analyzer = RedditClusterAnalyzer(config, args.host, args.port, args.domain, args.chains_per_cluster)
    dataset = analyzer.create_dataset(clustering_data, chain_sents, doc_ids)
    analyzer.process_clusters(dataset, args.workers, args.save_interval, args.sequential)
