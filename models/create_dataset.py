import argparse
import os
import pickle
import random
import sys

import numpy as np
import pandas as pd
from pyhocon import ConfigFactory
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, f_classif
from collections import Counter
from scipy import stats

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import the schemas module and create an alias for pickle compatibility
import annotation.schemas as schemas
sys.modules['schemas'] = schemas


def create_structural_cluster_features(clusters, chain_strengths, total_clusters):
    """Create structural narrative features from cluster assignments.
    
    Args:
        clusters: List of cluster IDs for chains in document
        chain_strengths: List of chain strengths (number of roles/entities)
        total_clusters: Total number of possible clusters
        
    Returns:
        List of structural features
    """
    if not clusters:
        return [0] * 8  # Return zeros if no clusters
    
    cluster_counts = Counter(clusters)
    unique_clusters = len(cluster_counts)
    total_chains = len(clusters)
    
    # 1. Dominant cluster ID (normalized to [0,1] to avoid large values)
    dominant_cluster = cluster_counts.most_common(1)[0][0] / max(total_clusters, 1)
    
    # 2. Cluster diversity (Shannon entropy) - safe calculation
    if total_chains <= 1 or unique_clusters <= 1:
        cluster_diversity = 0.0
    else:
        probs = np.array(list(cluster_counts.values())) / total_chains
        # Safe entropy calculation
        cluster_diversity = -np.sum(probs * np.log2(np.maximum(probs, 1e-10)))
        # Normalize by max possible entropy
        max_entropy = np.log2(unique_clusters)
        cluster_diversity = cluster_diversity / max(max_entropy, 1e-10)
    
    # 3. Narrative strength (average chain strength) - safe calculation
    narrative_strength = np.mean(chain_strengths) if chain_strengths else 0.0
    narrative_strength = float(narrative_strength)  # Ensure it's a float
    
    # 4. Narrative coverage (ratio of unique clusters to total chains)
    narrative_coverage = unique_clusters / max(total_chains, 1)
    
    # 5. Cluster concentration (max cluster frequency / total chains)
    cluster_concentration = max(cluster_counts.values()) / max(total_chains, 1)
    
    # 6. Max cluster strength (strength of strongest narrative)
    max_cluster_strength = float(max(chain_strengths)) if chain_strengths else 0.0
    
    # 7. Ratio of unique clusters to possible clusters
    cluster_coverage_ratio = unique_clusters / max(total_clusters, 1)
    
    # 8. Coefficient of variation of chain strengths
    if len(chain_strengths) <= 1 or narrative_strength == 0:
        strength_variation = 0.0
    else:
        strength_std = float(np.std(chain_strengths))
        strength_variation = strength_std / max(narrative_strength, 1e-10)
    
    # Ensure all values are finite
    features = [
        dominant_cluster,
        cluster_diversity, 
        narrative_strength,
        narrative_coverage,
        cluster_concentration,
        max_cluster_strength,
        cluster_coverage_ratio,
        strength_variation
    ]
    
    # Safety check: replace any inf/nan with 0
    features = [0.0 if not np.isfinite(f) else float(f) for f in features]
    
    return features


def create_dataset(config, clustering_data, processed_chains, corpus):
    """Create dataset for BERT training from clustering data and processed chains.
    
    Args:
        config: Configuration dictionary
        clustering_data: Clustering results
        processed_chains: Processed event chains
        corpus: Original corpus data
        
    Returns:
        Dictionary containing train/dev/test splits and metadata
    """
    print("Creating dataset...", flush=True)
    
    print("Creating structural narrative features instead of frequency counts...", flush=True)

    immig_roles = {
        "Immigrants:Hero": 0,
        "Immigrants:Threat": 0,
        "Immigrants:Victim": 0,
        "Immigration Advocates:Hero": 0,
        "Immigration Advocates:Threat": 0,
        "Immigration Advocates:Victim": 0,
        "Government:Hero": 0,
        "Government:Threat": 0,
        "Government:Victim": 0,
        "Judiciary:Hero": 0,
        "Judiciary:Threat": 0,
        "Judiciary:Victim": 0,
        "Law Enforcement:Hero": 0,
        "Law Enforcement:Threat": 0,
        "Law Enforcement:Victim": 0,
        "Politicians:Hero": 0,
        "Politicians:Threat": 0,
        "Politicians:Victim": 0
    }

    stance = {
        "Stance:Pro": 0,
        "Stance:Anti": 0
    }

    doc_to_clusters = {}

    for chain_idx, chain in processed_chains['processed_chains'].items():
        doc_id = chain['doc_id']
        if doc_id not in doc_to_clusters:
            doc_to_clusters[doc_id] = {
                'chains': [],
                'clusters': [],  # Changed from set to list to preserve order
                'cluster_freq': [0] * config['num_clusters'],
                'role_freq': immig_roles.copy(),
                'stance_freq': stance.copy(),
                'chain_to_cluster': {},
                'text': corpus[doc_id]['text'],
                'frame_label': corpus[doc_id]['primary_frame'],
                'chain_strengths': []  # To track narrative strength
            }
        doc_to_clusters[doc_id]['chains'].append([chain_idx])
        cluster_id = clustering_data['labels'][chain_idx]
        doc_to_clusters[doc_id]['clusters'].append(cluster_id)
        doc_to_clusters[doc_id]['cluster_freq'][cluster_id] += 1
        doc_to_clusters[doc_id]['chain_to_cluster'][chain_idx] = cluster_id
        
        # Calculate chain strength (number of roles/entities in this chain)
        chain_strength = len(processed_chains['chain_group_roles'][chain_idx])
        doc_to_clusters[doc_id]['chain_strengths'].append(chain_strength)

        chain_group_roles = processed_chains['chain_group_roles'][chain_idx]
        for char, role in chain_group_roles.items():
            key = "{}:{}".format(char, role)
            if key in doc_to_clusters[doc_id]['role_freq']:
                doc_to_clusters[doc_id]['role_freq'][key] += 1
            if key in doc_to_clusters[doc_id]['stance_freq']:
                doc_to_clusters[doc_id]['stance_freq'][key] += 1

    # Generate structural narrative features for each document
    for doc_id in doc_to_clusters:
        doc = doc_to_clusters[doc_id]
        clusters = doc['clusters']
        strengths = doc['chain_strengths']
        
        # Create structural features
        doc['structural_features'] = create_structural_cluster_features(clusters, strengths, config['num_clusters'])
        
        # generate a list using values of the role_stance_freq dictionary  
        doc['role_freq_list'] = list(doc['role_freq'].values())
        doc['stance_freq_list'] = list(doc['stance_freq'].values())

    data = []
    for doc_id, doc in doc_to_clusters.items():
        data.append({
            'doc_id': doc_id,
            'text': doc['text'],
            'cluster_feats': doc['structural_features'],  # Use structural features by default
            'cluster_feats_frequency': doc['cluster_freq'],  # Keep original frequency features
            'role_feats': doc['role_freq_list'],
            'stance_feats': doc['stance_freq_list'],
            'frame_label': doc['frame_label']
        })
    
    df = pd.DataFrame(data)
    
    # Use structural features (no dimensionality reduction needed - already compact)
    cluster_feats_array = np.array(df['cluster_feats'].tolist())
    print(f"Structural cluster features shape: {cluster_feats_array.shape}")
    
    # Debug: Check for problematic values
    print(f"Cluster features - Min: {np.min(cluster_feats_array)}, Max: {np.max(cluster_feats_array)}")
    print(f"Any inf values: {np.any(np.isinf(cluster_feats_array))}")
    print(f"Any nan values: {np.any(np.isnan(cluster_feats_array))}")
    
    cluster_feats_reduced = cluster_feats_array  # No reduction needed
    
    # Normalize cluster_feats and role_stance_feats using StandardScaler
    cluster_scaler = StandardScaler()
    role_stance_scaler = StandardScaler()
    
    cluster_feats_normalized = cluster_scaler.fit_transform(cluster_feats_reduced)
    role_feats_normalized = role_stance_scaler.fit_transform(df['role_feats'].tolist())
    stance_feats_normalized = role_stance_scaler.fit_transform(df['stance_feats'].tolist())
    
    df['cluster_feats'] = cluster_feats_normalized.tolist()
    df['role_feats'] = role_feats_normalized.tolist()
    df['stance_feats'] = stance_feats_normalized.tolist()
    
    label_encoder = LabelEncoder()
    df['frame_label_encoded'] = label_encoder.fit_transform(df['frame_label'])
    
    # Split into train, dev, test (70%, 15%, 15%)
    train_df, temp_df = train_test_split(
        df, test_size=0.3, random_state=config["seed"], 
        stratify=df['frame_label_encoded']
    )
    dev_df, test_df = train_test_split(
        temp_df, test_size=0.5, random_state=config["seed"],
        stratify=temp_df['frame_label_encoded']
    )
    
    # Reset indices to maintain order for doc_id recovery
    train_df = train_df.reset_index(drop=True)
    dev_df = dev_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    dataset = {
        'train_df': train_df,
        'dev_df': dev_df,
        'test_df': test_df,
        'label_encoder': label_encoder,
        'cluster_scaler': cluster_scaler,
        'role_stance_scaler': role_stance_scaler,
        'feature_names': ['dominant_cluster', 'cluster_diversity', 'narrative_strength', 
                         'narrative_coverage', 'cluster_concentration', 'max_cluster_strength',
                         'cluster_coverage_ratio', 'strength_variation'],
        'num_structural_features': cluster_feats_array.shape[1]
    }
    
    return dataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Create dataset for BERT training')
    parser.add_argument('-c', metavar='CONF', default='base', help='configuration (see config.conf)')

    args = parser.parse_args()
    config = ConfigFactory.parse_file('./config.conf')[args.c]

    print("Loading data from disk...", flush=True)

    with open(config["cluster_eval_path"], "rb") as f:
        clustering_data = pickle.load(f)

    with open(config["processed_chains_path"], "rb") as f:
        processed_chains = pickle.load(f)

    with open(config["char_event_chains_path"], "rb") as f:
        corpus = pickle.load(f)

    # Set seed for reproducibility
    seed = config["seed"]
    random.seed(seed)
    np.random.seed(seed)

    # Create dataset
    dataset = create_dataset(config, clustering_data, processed_chains, corpus)
    
    # Save dataset to disk
    print("Saving dataset to disk...", flush=True)
    with open(config["frame_prediction_data_path"] + "frame_prediction_data.pickle", "wb") as f:
        pickle.dump(dataset, f)
    
    print("Dataset creation complete!", flush=True)
    print(f"Train samples: {len(dataset['train_df'])}")
    print(f"Dev samples: {len(dataset['dev_df'])}")
    print(f"Test samples: {len(dataset['test_df'])}")