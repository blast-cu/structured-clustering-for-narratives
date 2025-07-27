import argparse
import csv
import json
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


def generate_cluster_themes_json(cluster_analysis, output_file="cluster_themes.json"):
    """Generate JSON file with cluster index as key and theme as value for all clusters.
    
    Args:
        cluster_analysis: Dictionary with cluster data
        output_file: Output JSON filename
    """
    cluster_themes = {}
    
    for cluster_id, cluster_data in cluster_analysis.items():
        cluster_themes[str(cluster_id)] = cluster_data['theme']
    
    # Sort by numeric value of cluster ID for proper ordering
    sorted_cluster_themes = dict(sorted(cluster_themes.items(), key=lambda x: int(x[0])))
    
    with open(output_file, 'w', encoding='utf-8') as jsonfile:
        json.dump(sorted_cluster_themes, jsonfile, indent=2, ensure_ascii=False)
    
    print(f"Generated JSON with {len(cluster_themes)} cluster themes in {output_file}")


def gen_data(cluster_analysis, ranked_clusters, output_file="theme_qual_data.tsv"):
    """Generate TSV data from stratified sampling of clusters by density.
    
    Sample 15 from top 25%, 20 from middle 25-75%, 15 from bottom 25%.
    
    Args:
        cluster_analysis: Dictionary with cluster data
        ranked_clusters: List of cluster IDs ranked by density (highest to lowest)
        output_file: Output TSV filename
    """
    total_clusters = len(ranked_clusters)
    
    # Calculate quartile boundaries
    top_25_end = total_clusters // 4
    bottom_25_start = 3 * total_clusters // 4
    
    # Stratify clusters by density regions
    top_25_clusters = ranked_clusters[:top_25_end]
    middle_50_clusters = ranked_clusters[top_25_end:bottom_25_start]
    bottom_25_clusters = ranked_clusters[bottom_25_start:]
    
    # Sample from each region
    sampled_clusters = []
    
    # Sample 15 from top 25%
    top_sample_size = min(15, len(top_25_clusters))
    top_sample = random.sample(top_25_clusters, top_sample_size) if top_25_clusters else []
    sampled_clusters.extend([(cluster_id, "Top 25%") for cluster_id in top_sample])
    
    # Sample 20 from middle 25-75%
    middle_sample_size = min(20, len(middle_50_clusters))
    middle_sample = random.sample(middle_50_clusters, middle_sample_size) if middle_50_clusters else []
    sampled_clusters.extend([(cluster_id, "Middle 25-75%") for cluster_id in middle_sample])
    
    # Sample 15 from bottom 25%
    bottom_sample_size = min(15, len(bottom_25_clusters))
    bottom_sample = random.sample(bottom_25_clusters, bottom_sample_size) if bottom_25_clusters else []
    sampled_clusters.extend([(cluster_id, "Bottom 25%") for cluster_id in bottom_sample])
    
    print(f"Sampled {len(top_sample)} from top 25% ({len(top_25_clusters)} available)")
    print(f"Sampled {len(middle_sample)} from middle 25-75% ({len(middle_50_clusters)} available)")
    print(f"Sampled {len(bottom_sample)} from bottom 25% ({len(bottom_25_clusters)} available)")
    print(f"Total clusters sampled: {len(sampled_clusters)}")
    
    with open(output_file, 'w', newline='', encoding='utf-8') as tsvfile:
        writer = csv.writer(tsvfile, delimiter='\t')
        
        # Write header with third column for sampling region
        writer.writerow(['All_Sentences_and_Roles', 'Theme', 'Density_Region'])
        
        for cluster_id, region in sampled_clusters:
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
                
                # Pretty print the theme JSON string
                if isinstance(theme, str):
                    try:
                        theme_dict = json.loads(theme)
                        pretty_theme = json.dumps(theme_dict, indent=2, ensure_ascii=False)
                    except json.JSONDecodeError:
                        pretty_theme = theme  # Use original if not valid JSON
                else:
                    pretty_theme = json.dumps(theme, indent=2, ensure_ascii=False)
                
                writer.writerow([combined_sentences, pretty_theme, region])
    
    print(f"Generated TSV data for {len(sampled_clusters)} clusters in {output_file}")


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
    
    # Generate JSON file with all cluster themes
    generate_cluster_themes_json(cluster_analysis)
    
    # Generate stratified TSV data
    gen_data(cluster_analysis, ranked_clusters)

if __name__ == "__main__":
    main()