import argparse
import json
import os
import pickle
import time

import pandas as pd
from filelock import FileLock
from numpy import mean, std
from pyhocon import ConfigFactory
from sklearn import preprocessing, metrics
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler


def create_dataset(num_clusters, constraint_weight):
    print("Creating dataset...")

    with open("./data/immigration/annotated_event_chains.pickle", 'rb') as f:
        annotated_event_chains = pickle.load(f)

    with open(f"./data/immigration/clusters_{num_clusters}_{constraint_weight}.pickle", 'rb') as f:
        clusters = pickle.load(f)

    with open("./data/immigration/embeddings.pickle", 'rb') as f:
        embeddings = pickle.load(f)

    corpus = {}
    for chain_idx in embeddings['chain_to_doc']:
        doc_key = embeddings['chain_to_doc'][chain_idx]
        if doc_key not in corpus:
            doc = annotated_event_chains[doc_key]
            corpus[doc_key] = {
                "primary_frame": doc['primary_frame'],
                "event_chain_clusters": {}
            }
        cluster = clusters['labels'][chain_idx]
        if cluster not in corpus[doc_key]["event_chain_clusters"]:
            corpus[doc_key]["event_chain_clusters"][cluster] = 1
        else:
            corpus[doc_key]["event_chain_clusters"][cluster] += 1

    data_dict = []
    for doc_key in corpus:
        doc = corpus[doc_key]
        article = {}
        for i in range(num_clusters):
            key = 'Cluster ' + str(i)
            article[key] = 0
        for cluster in doc['event_chain_clusters']:
            article['Cluster ' + str(cluster)] = doc['event_chain_clusters'][cluster]
        article['frame'] = doc['primary_frame']
        data_dict.append(article)
    data = pd.DataFrame.from_dict(data_dict)


    return data

def regression(data, config, num_clusters, constraint_weight,):
    print("Running regression...")
    label_encoder = preprocessing.LabelEncoder()
    data['frame'] = label_encoder.fit_transform(data['frame'])
    label_mapping = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
    for k, v in label_mapping.items():
        print(v, ' : ', k)

    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=config['seed'])

    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # feature_selection(config, X_train_scaled, y_train, label_mapping)

    model = LogisticRegression(solver='lbfgs', penalty='l2', C=0.5)

    # # define the model evaluation procedure
    # cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=42)
    # # evaluate the model and collect the scores
    # n_scores = cross_val_score(model, X_train_scaled, y_train, scoring='accuracy', cv=cv, n_jobs=-1)
    # # report the model performance
    # print('Mean Train Accuracy: %.3f (%.3f)' % (mean(n_scores), std(n_scores)))

    model.fit(X_train_scaled, y_train)
    Y_pred = model.predict(X_test_scaled)
    test_accuracy = round(metrics.accuracy_score(y_test.to_numpy(), Y_pred), 2)

    print(f"Test Accuracy for num_clusters={num_clusters}, constraint_weight={constraint_weight}: {test_accuracy}")
    print(metrics.classification_report(y_test.to_numpy(), Y_pred))

    print(metrics.classification_report(y_test.to_numpy(), Y_pred))

    # Log the results
    log_results(num_clusters, constraint_weight, test_accuracy, config)

def log_results(num_clusters, constraint_weight, accuracy, config):
    """
    Log the results to a JSON file in a thread-safe manner using filelock.

    Args:
        num_clusters: Number of clusters used
        constraint_weight: Constraint weight for clustering
        accuracy: Test accuracy
        results_file: Path to the results file
    """
    # Create a new result entry
    new_result = {
        "num_clusters": num_clusters,
        "constraint_weight": constraint_weight,
        "accuracy": accuracy,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }

    results_file = config["grid_search_path"]

    # Define lock file
    lock_file = f"{results_file}.lock"

    # Use filelock for thread-safe file access
    with FileLock(lock_file, timeout=10):
        # Read existing results if file exists
        results = []
        if os.path.exists(results_file) and os.path.getsize(results_file) > 0:
            try:
                with open(results_file, 'r') as f:
                    results = json.load(f)
            except json.JSONDecodeError:
                # File exists but is corrupted
                print(f"Warning: Results file corrupted. Creating new file.", flush=True)
                results = []

        # Add new result
        results.append(new_result)

        # Write updated results to file
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)

    print(f"Results logged successfully to {results_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Regression")
    parser.add_argument('-c', metavar='CONF', default='base', help='configuration (see config.conf)')
    parser.add_argument('-k', type=int, default=250, help='number of clusters to use')
    parser.add_argument('-w', type=float, default=1.0, help='constraint weight for clustering')
    args = parser.parse_args()

    config = ConfigFactory.parse_file('./config.conf')[args.c]

    print(f"Running regression with:")
    print(f"  - Number of clusters: {args.k}")
    print(f"  - Constraint weight: {args.w}")

    data = create_dataset(args.k, args.w)
    regression(data, config, args.k, args.w)