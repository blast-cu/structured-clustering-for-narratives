import argparse
import pickle
import random

import numpy as np
import pandas as pd
from pyhocon import ConfigFactory
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split


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
                'clusters': set(),
                'cluster_freq': [0] * config['num_clusters'],
                'role_freq': immig_roles.copy(),
                'stance_freq': stance.copy(),
                'chain_to_cluster': {},
                'text': corpus[doc_id]['text'],
                'frame_label': corpus[doc_id]['primary_frame']
            }
        doc_to_clusters[doc_id]['chains'].append([chain_idx])
        doc_to_clusters[doc_id]['clusters'].add(clustering_data['labels'][chain_idx])
        doc_to_clusters[doc_id]['cluster_freq'][clustering_data['labels'][chain_idx]] += 1
        doc_to_clusters[doc_id]['chain_to_cluster'][chain_idx] = clustering_data['labels'][chain_idx]

        chain_group_roles = processed_chains['chain_group_roles'][chain_idx]
        for char, role in chain_group_roles.items():
            key = "{}:{}".format(char, role)
            if key in doc_to_clusters[doc_id]['role_freq']:
                doc_to_clusters[doc_id]['role_freq'][key] += 1
            if key in doc_to_clusters[doc_id]['stance_freq']:
                doc_to_clusters[doc_id]['stance_freq'][key] += 1

        # generate a list using values of the role_stance_freq dictionary
        doc_to_clusters[doc_id]['role_freq_list'] = list(doc_to_clusters[doc_id]['role_freq'].values())
        doc_to_clusters[doc_id]['stance_freq_list'] = list(doc_to_clusters[doc_id]['stance_freq'].values())

    data = []
    for doc_id, doc in doc_to_clusters.items():
        data.append({
            'doc_id': doc_id,
            'text': doc['text'],
            'cluster_feats': doc['cluster_freq'],
            'role_feats': doc['role_freq_list'],
            'stance_feats': doc['stance_freq_list'],
            'frame_label': doc['frame_label']
        })
    
    df = pd.DataFrame(data)
    
    # Normalize cluster_feats and role_stance_feats to [0,1] using StandardScaler
    cluster_scaler = StandardScaler()
    role_stance_scaler = StandardScaler()
    
    cluster_feats_normalized = cluster_scaler.fit_transform(df['cluster_feats'].tolist())
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
        'role_stance_scaler': role_stance_scaler
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