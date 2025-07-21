import argparse
import csv
import pickle
import random

import numpy as np
from pyhocon import ConfigFactory


def rank_clusters_by_density(clustering_data):
    """Rank clusters by density using average distance to centroid.
    
    Returns:
        list: Cluster IDs ordered by density (highest to lowest)
    """
    cluster_densities = []
    
    for cluster_id in range(clustering_data["number_cluster"]):
        cluster_mask = clustering_data["labels"] == cluster_id
        cluster_points = clustering_data["embeddings"][cluster_mask]
        center = clustering_data["cluster_centers"][cluster_id]
        
        if len(cluster_points) > 0:
            distances = np.linalg.norm(cluster_points - center, axis=1)
            avg_distance = np.mean(distances)
            cluster_densities.append((cluster_id, avg_distance, len(cluster_points)))
        else:
            cluster_densities.append((cluster_id, float('inf'), 0))
    
    # Sort by density (lower average distance = higher density)
    cluster_densities.sort(key=lambda x: x[1])
    
    # Return cluster IDs in rank order
    return [cluster_id for cluster_id, _, _ in cluster_densities]


def gen_data(cluster_analysis, ranked_clusters, output_file="theme_qual_data.tsv"):
    """Generate TSV data for top 50 ranked clusters.
    
    Args:
        cluster_analysis: Dictionary with cluster data
        ranked_clusters: List of cluster IDs ranked by density
        output_file: Output TSV filename
    """
    # Take top 50 ranked clusters
    top_clusters = ranked_clusters[:50]
    
    with open(output_file, 'w', newline='', encoding='utf-8') as tsvfile:
        writer = csv.writer(tsvfile, delimiter='\t')
        
        # Write header
        writer.writerow(['All_Sentences_and_Roles', 'Theme'])
        
        for cluster_id in top_clusters:
            if cluster_id in cluster_analysis:
                cluster_data = cluster_analysis[cluster_id]
                chains = cluster_data['chains']
                theme = cluster_data['theme']
                
                # Randomly sample 25 items (or all if fewer than 25)
                sample_size = min(25, len(chains))
                sampled_chains = random.sample(chains, sample_size)
                
                # Combine all sentences and roles into one cell
                all_sentences = []
                for i, chain in enumerate(sampled_chains, 1):
                    sentence_and_roles = f"{i}. Sentence: {chain['chain_text']} | Character Roles: {chain['char_role']}"
                    all_sentences.append(sentence_and_roles)
                
                # Join all sentences with newlines
                combined_sentences = '\n'.join(all_sentences)
                writer.writerow([combined_sentences, theme])
    
    print(f"Generated TSV data for {len(top_clusters)} clusters in {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Generate intrusion detection task data")
    parser.add_argument('-c', '--config', default='base', help='Configuration name')

    args = parser.parse_args()
    config = ConfigFactory.parse_file('./config.conf')[args.config]

    with open(config['cluster_analysis_path'], "rb") as f:
        cluster_analysis = pickle.load(f)

    with open(config['cluster_eval_path'], "rb") as f:
        clustering_data = pickle.load(f)

    ranked_clusters = rank_clusters_by_density(clustering_data)
    gen_data(cluster_analysis, ranked_clusters)

if __name__ == "__main__":
    main()