import argparse
import concurrent
import os
import pickle
import random
import re
import shutil
import traceback

import numpy as np
from pyhocon import ConfigFactory
from sklearn.metrics import euclidean_distances
from tqdm import tqdm

from schemas import ClusterTheme
from utils.ollama_client import Ollama


class ClusterAnalyzer:
    def __init__(self, config, host, port, domain, chains_per_cluster):
        self.config = config

        self.reasoning_model = Ollama(host,
                                      port,
                                      config['reasoning_model'],
                                      config['seed'],
                                      config['temperature'])

        self.output_model = Ollama(host,
                                   port,
                                   config['output_model'],
                                   config['seed'],
                                   config['temperature'])

        self.num_ctx = config['num_ctx']
        self.chains_per_cluster = chains_per_cluster

        random.seed(config["seed"])

        with (open("./annotation/prompts/cluster_analysis/system_prompt.md", 'r', encoding='utf-8') as file):
            self.reasoning_system_prompt = file.read()

        with open("./annotation/prompts/cluster_analysis/structured_output.md", 'r', encoding='utf-8') as file:
            self.structured_output_system_prompt = file.read()

        self.domain = domain
        if self.domain == 'guncontrol':
            with open("./annotation/prompts/cluster_analysis/guncontrol_roles.md", 'r',
                      encoding='utf-8') as file:
                self.char_roles = file.read()
        elif self.domain == "immigration":
            with open("./annotation/prompts/cluster_analysis/immigration_roles.md", 'r',
                      encoding='utf-8') as file:
                self.char_roles = file.read()


    @staticmethod
    def filter_chains_by_centroid_proximity(clustering_data, top_k_percent=25.0):
        embeddings = clustering_data['embeddings']
        labels = clustering_data['labels']
        centroids = clustering_data['cluster_centers']

        chains_to_keep = set()

        # For each cluster, find chains closest to centroid
        for cluster_id in range(clustering_data['number_cluster']):
            # Get indices of chains in this cluster
            cluster_chain_indices = np.where(labels == cluster_id)[0]

            if len(cluster_chain_indices) == 0:
                continue

            # Get embeddings for chains in this cluster
            cluster_embeddings = embeddings[cluster_chain_indices]
            cluster_centroid = centroids[cluster_id].reshape(1, -1)

            # Calculate distances to centroid
            distances = euclidean_distances(cluster_embeddings, cluster_centroid).flatten()

            # Sort by distance and keep top k%
            sorted_indices = np.argsort(distances)
            num_to_keep = max(1, int(len(cluster_chain_indices) * top_k_percent / 100.0))
            closest_indices = sorted_indices[:num_to_keep]

            # Add the actual chain indices (not relative to cluster)
            chains_to_keep.update(cluster_chain_indices[closest_indices])

        print(
            f"Filtered from {len(labels)} to {len(chains_to_keep)} chains ({len(chains_to_keep) / len(labels) * 100:.1f}%)",
            flush=True)
        return chains_to_keep


    def create_dataset(self, clustering_data, processed_chains):
        chains_to_keep = self.filter_chains_by_centroid_proximity(clustering_data)
        clusters = {}
        for idx, label in enumerate(clustering_data['labels']):
            if idx not in chains_to_keep:
                continue
            if label not in clusters:
                clusters[label] = {
                    "chains": [],
                    "theme": ""
                }

            stance = ""
            char_roles = processed_chains['chain_group_roles'][idx]
            if 'Stance' in char_roles:
                stance = char_roles['Stance']
                char_roles.pop('Stance')
            chain_entry = {
                "chain_idx": idx,
                "chain_text": processed_chains['processed_chains'][idx]['chain_text'],
                "doc_id": processed_chains['processed_chains'][idx]['doc_id'],
                "char_role": char_roles,
                "stance": stance
            }
            clusters[label]['chains'].append(chain_entry)
        return clusters

    def process_clusters(self, clusters, num_workers, save_interval, sequential=False):
        processed_clusters = self.load_existing_progress()
        if len(processed_clusters) > 0:
            analyzed_clusters = processed_clusters
        else:
            analyzed_clusters = {}

        total_clusters = len(clusters)
        existing_clusters = len(analyzed_clusters)
        clusters_to_process = total_clusters - existing_clusters

        print(f"Total documents: {total_clusters}", flush=True)
        print(f"Existing processed documents: {existing_clusters}", flush=True)
        print(f"Documents to process: {clusters_to_process}", flush=True)

        if sequential:
            # Sequential processing - only process unprocessed documents
            with tqdm(total=clusters_to_process, desc="Processing clusters") as pbar:
                processed_count = 0
                for cluster_idx, cluster in clusters.items():
                    if cluster_idx in analyzed_clusters:
                        # Skip already processed documents - don't update progress bar
                        continue
                    try:
                        processed_cluster = self.process_cluster(cluster)
                        analyzed_clusters[cluster_idx] = processed_cluster
                        processed_count += 1

                        # Update the progress bar
                        pbar.update(1)

                        # Save annotated_docs at regular intervals
                        if processed_count % save_interval == 0:
                            self.save_progress(analyzed_clusters)
                            print(f"Progress saved after processing {processed_count} documents.", flush=True)

                    except Exception as e:
                        print(f"Error processing document {cluster_idx}: {e}", flush=True)
                        traceback.print_exc()
                        pbar.update(1)  # Still update progress bar even if there's an error
        else:
            # Parallel processing - only submit futures for unprocessed documents
            unprocessed_items = {doc_idx: doc for doc_idx, doc in clusters.items()
                                 if doc_idx not in analyzed_clusters}

            if not unprocessed_items:
                print("All documents already processed!", flush=True)
                return

            with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
                futures = {executor.submit(self.process_cluster, cluster): cluster_idx for cluster_idx, cluster in
                           unprocessed_items.items()}
                # Initialize tqdm progress bar to track document processing
                with tqdm(total=len(unprocessed_items), desc="Processing documents") as pbar:
                    processed_count = 0

                    # Process documents and save at regular intervals
                    for future in concurrent.futures.as_completed(futures):
                        cluster_idx = futures[future]
                        try:
                            cluster = future.result()
                            analyzed_clusters[cluster_idx] = cluster
                            processed_count += 1

                            # Update the progress bar
                            pbar.update(1)

                            # Save annotated_docs at regular intervals
                            if processed_count % save_interval == 0:
                                self.save_progress(analyzed_clusters)
                                print(f"Progress saved after processing {processed_count} documents.", flush=True)

                        except Exception as e:
                            print(f"Error processing document {cluster_idx}: {e}", flush=True)
                            traceback.print_exc()
                            pbar.update(1)  # Still update progress bar even if there's an error

        # Final save at the end, in case it wasn't saved during the last interval
        self.save_progress(analyzed_clusters)
        print(f"Processing complete. Final save completed.", flush=True)


    def process_cluster(self, cluster):
        if self.domain == 'guncontrol':
            domain = "Gun Control"
        elif self.domain == 'immigration':
            domain = "Immigration"
        
        items_to_sample = min(self.chains_per_cluster, len(cluster['chains']))
        sampled_items = random.sample(cluster['chains'], items_to_sample)

        sentences = ""
        idx = 1
        for item in sampled_items:
            item_str = (f"{idx}. Sentence: {item['chain_text']}\n Character Roles: {item['char_role']}\n Stance:"
                        f" {item['stance']}")
            sentences = sentences + item_str + "\n\n"
            idx += 1

        cluster['theme'] = self.annotate(domain, sentences)
        return cluster

    def annotate(self, domain, sentences):
        reasoning_user_prompt = (f"DOMAIN: \"{domain}\" \n CHARACTER ROLES: \n \"{self.char_roles}\" \n SENTENCES: \n"
                       f"{sentences} \n ")

        max_retries = 5
        retry_count = 0
        while retry_count < max_retries:
            try:
                reasoning_model_response = self.reasoning_model.chat(self.reasoning_system_prompt,
                                                                     reasoning_user_prompt,
                                                                     think=True,
                                                                     num_ctx=self.num_ctx)
                structured_response = self.output_model.chat(self.structured_output_system_prompt,
                                                             reasoning_model_response,
                                                             think=False,
                                                             repeat_penalty=True,
                                                             format=ClusterTheme.model_json_schema(),
                                                             num_ctx=self.num_ctx)
                try:
                    response = ClusterTheme.model_validate_json(structured_response)
                    response = response.model_dump()
                    return response['theme']
                except Exception as e:
                    print("Exception: " + str(e), flush=True)
                    print("Invalid response. Please try again.", flush=True)
                    retry_count += 1
            except Exception as e:
                print("Exception: " + str(e), flush=True)
                print("Ollama Error. Please try again.", flush=True)
                retry_count += 1
        return None

    def save_progress(self, analyzed_clusters):
        with open(self.config["cluster_analysis_path"], 'wb') as f:
            pickle.dump(analyzed_clusters, f)

    def load_existing_progress(self):
        """Load existing processed documents if save file exists, creating a backup first"""
        save_path = self.config["cluster_analysis_path"]

        if os.path.exists(save_path):
            print(f"Found existing save file: {save_path}")
            try:
                existing_data = self.create_backup_and_load(save_path)
                print(f"Loaded {len(existing_data)} existing documents")
                return existing_data
            except Exception as e:
                print(f"Error loading existing save file: {e}")
                print("Starting fresh...")
                return {}
        else:
            print("No existing save file found. Starting fresh...")
            return {}

    @staticmethod
    def get_next_backup_number(base_path):
        """Find the next highest backup number for the given file path"""
        directory = os.path.dirname(base_path)
        base_name = os.path.splitext(os.path.basename(base_path))[0]
        extension = os.path.splitext(base_path)[1]

        # Find all existing backup files and extract their numbers
        backup_numbers = []

        if os.path.exists(directory):
            for filename in os.listdir(directory):
                # Check if the file matches the backup pattern
                backup_pattern = f"{base_name}_backup_(\d+){re.escape(extension)}"
                match = re.match(backup_pattern, filename)
                if match:
                    backup_numbers.append(int(match.group(1)))

        # Return the next highest number
        if backup_numbers:
            return max(backup_numbers) + 1
        else:
            return 1

    @staticmethod
    def create_backup_and_load(save_path):
        """Create a backup of existing save file and load the data from the original file"""
        backup_number = ClusterAnalyzer.get_next_backup_number(save_path)

        directory = os.path.dirname(save_path)
        base_name = os.path.splitext(os.path.basename(save_path))[0]
        extension = os.path.splitext(save_path)[1]

        backup_name = f"{base_name}_backup_{backup_number}{extension}"
        backup_path = os.path.join(directory, backup_name)

        # Create backup by copying the existing file
        shutil.copy2(save_path, backup_path)
        print(f"Created backup: {backup_path}")

        # Load data from the original file
        with open(save_path, 'rb') as f:
            existing_data = pickle.load(f)
        print(f"Successfully loaded data from original file: {save_path}")
        return existing_data


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process event chains')
    parser.add_argument('-c', metavar='CONF', default='base', help='configuration (see config.conf)')
    parser.add_argument('--host', metavar='HOST')
    parser.add_argument('--port', default=9999, metavar='PORT')
    parser.add_argument('--domain', metavar='DOMAIN')
    parser.add_argument('--chains_per_cluster', type=int, default=25, metavar='CHAINS_PER_CLUSTER')
    parser.add_argument('--workers', type=int, default=4, metavar='WORKERS', help='Number of worker threads')
    parser.add_argument('--save_interval', type=int, default=10, metavar='SAVE_INTERVAL',
                        help='Number of documents to process before saving progress')
    parser.add_argument('--sequential', action='store_true', help='Run in sequential mode instead of parallel')

    args = parser.parse_args()
    config = ConfigFactory.parse_file('./config.conf')[args.c]

    with open("./data/mfc/guncontrol/clustering/clusters_300_0.5_.pickle", 'rb') as f:
        clustering_data = pickle.load(f)

    with open(config["processed_chains_path"], "rb") as f:
        processed_chains = pickle.load(f)

    analyzer = ClusterAnalyzer(config, args.host, args.port, args.domain, args.chains_per_cluster)
    dataset_for_analysis = analyzer.create_dataset(clustering_data, processed_chains)
    analyzer.process_clusters(dataset_for_analysis, args.workers, args.save_interval, args.sequential)