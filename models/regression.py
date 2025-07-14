import argparse
import pickle
import random

import numpy as np
import pandas as pd
from numpy import mean, std
from pyhocon import ConfigFactory
from sklearn import preprocessing, metrics
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics.pairwise import euclidean_distances


class RegressionModel:
    def __init__(self, config):
        self.config = config
        random.seed(self.config["seed"])
        np.random.seed(self.config["seed"])

    def filter_chains_by_centroid_proximity(self, clustering_data, processed_chains, top_k_percent=50.0):
        """
        Filter event chains to keep only those in the top k% closest to their cluster centroids.
        
        Args:
            clustering_data: Dictionary containing embeddings, labels, and cluster_centers
            processed_chains: Dictionary containing processed chain data
            top_k_percent: Percentage of closest chains to keep per cluster (default 50%)
            
        Returns:
            Set of chain indices to keep
        """
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
            
        print(f"Filtered from {len(labels)} to {len(chains_to_keep)} chains ({len(chains_to_keep)/len(labels)*100:.1f}%)", flush=True)
        return chains_to_keep

    def create_dataset(self, config, clustering_data, use_centroid_filtering=False, centroid_top_k_percent=50.0):
        print("Creating dataset...", flush=True)

        with open(self.config["relations_path"], 'rb') as f:
            corpus = pickle.load(f)

        with open(config["processed_chains_path"], 'rb') as f:
            processed_chains = pickle.load(f)

        # Filter chains if centroid filtering is enabled
        if use_centroid_filtering:
            print(f"Filtering chains to top {centroid_top_k_percent}% closest to centroids...", flush=True)
            chains_to_keep = self.filter_chains_by_centroid_proximity(
                clustering_data, processed_chains, centroid_top_k_percent
            )
        else:
            chains_to_keep = set(processed_chains['processed_chains'].keys())

        dataset = {}
        for chain_idx in processed_chains['processed_chains']:
            # Skip chains that are not in the filtered set
            if chain_idx not in chains_to_keep:
                continue
            doc_key = processed_chains['processed_chains'][chain_idx]['doc_id']
            if doc_key not in dataset:
                doc = corpus[doc_key]
                dataset[doc_key] = {
                    "primary_frame": doc['primary_frame'],
                    "event_chain_clusters": {}
                }
            cluster = clustering_data['labels'][chain_idx]
            if cluster not in dataset[doc_key]["event_chain_clusters"]:
                dataset[doc_key]["event_chain_clusters"][cluster] = 1
            else:
                dataset[doc_key]["event_chain_clusters"][cluster] += 1

        data_dict = []
        for doc_key in dataset:
            doc = dataset[doc_key]
            article = {}
            for i in range(clustering_data['number_cluster']):
                key = 'Cluster ' + str(i)
                article[key] = 0
            for cluster in doc['event_chain_clusters']:
                article['Cluster ' + str(cluster)] = doc['event_chain_clusters'][cluster]
            article['frame'] = doc['primary_frame']
            data_dict.append(article)
        data = pd.DataFrame.from_dict(data_dict)

        return data

    def regression(self, config, data, use_centroid_filtering=False, centroid_top_k_percent=50.0):
        print("Running regression...", flush=True)
        # Print filtering info in results
        if use_centroid_filtering:
            print(f"# Centroid filtering: top {centroid_top_k_percent}% of chains used", flush=True)
        else:
            print(f"# No centroid filtering: all chains used", flush=True)
        label_encoder = preprocessing.LabelEncoder()
        data['frame'] = label_encoder.fit_transform(data['frame'])
        label_mapping = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
        for k, v in label_mapping.items():
            print(v, ' : ', k, flush=True)

        X = data.iloc[:, :-1]
        y = data.iloc[:, -1]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=config['seed'])
        
        print(f"Train set size: {len(X_train)}", flush=True)
        print(f"Test set size: {len(X_test)}", flush=True)

        # Create pipeline to avoid data leakage in cross-validation
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', LogisticRegression(solver='lbfgs', penalty='l2', C=0.5, max_iter=1000))
        ])

        # define the model evaluation procedure
        cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=config['seed'])
        # evaluate the model and collect the scores (pipeline handles scaling within each fold)
        accuracy_scores = cross_val_score(pipeline, X_train, y_train, scoring='accuracy', cv=cv, n_jobs=-1)
        f1_scores = cross_val_score(pipeline, X_train, y_train, scoring='f1_macro', cv=cv, n_jobs=-1)
        # report the model performance
        print('Mean Train Accuracy: %.2f (%.2f)' % (mean(accuracy_scores) * 100, std(accuracy_scores) * 100), flush=True)
        print('Mean Train F1: %.2f (%.2f)' % (mean(f1_scores) * 100, std(f1_scores) * 100), flush=True)

        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        test_accuracy = round(metrics.accuracy_score(y_test.to_numpy(), y_pred) * 100, 2)
        f1_score = round(metrics.f1_score(y_test.to_numpy(), y_pred, average='macro') * 100, 2)

        print(f"Test Accuracy: {test_accuracy}", flush=True)
        print(f"F1 Score: {f1_score}", flush=True)
        print(metrics.classification_report(y_test.to_numpy(), y_pred), flush=True)

        # Print tab-separated row with all metrics
        train_accuracy = round(mean(accuracy_scores) * 100, 2)
        train_f1 = round(mean(f1_scores) * 100, 2)
        print(f"{train_accuracy}\t{test_accuracy}\t{train_f1}\t{f1_score}", flush=True)
            
        return test_accuracy, f1_score, train_accuracy, train_f1

    def run_regression(self, config, clustering_data):
        """
        Run regression twice: once on top 25% closest to centroids, then on all chains.
        Prints results in tab-separated format: Train Acc, Test Acc, Train F1, Test F1.
        """
        # Run 1: Top 25% closest to centroids
        data_filtered = self.create_dataset(config, clustering_data, 
                                          use_centroid_filtering=True, 
                                          centroid_top_k_percent=25.0)
        test_acc_filtered, f1_filtered, train_acc_filtered, train_f1_filtered = self.regression(config, data_filtered, 
                                                        use_centroid_filtering=True, 
                                                        centroid_top_k_percent=25.0)
        
        # Run 2: All chains
        data_all = self.create_dataset(config, clustering_data, 
                                     use_centroid_filtering=False)
        test_acc_all, f1_all, train_acc_all, train_f1_all = self.regression(config, data_all, 
                                             use_centroid_filtering=False)
        
        # Format results: Top 25% block then All chains block (Train Acc, Test Acc, Train F1, Test F1)
        values = [f"{train_acc_filtered:.2f}", f"{test_acc_filtered:.2f}", f"{train_f1_filtered:.2f}", f"{f1_filtered:.2f}",
                  f"{train_acc_all:.2f}", f"{test_acc_all:.2f}", f"{train_f1_all:.2f}", f"{f1_all:.2f}"]
        
        print('\t'.join(values))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Regression")
    parser.add_argument('-c', metavar='CONF', default='base', help='configuration (see config.conf)')
    parser.add_argument('--use-centroid-filtering', action='store_true', 
                        help='Filter chains to only use those closest to cluster centroids')
    parser.add_argument('--centroid-top-k-percent', type=float, default=50.0,
                        help='Percentage of chains closest to centroids to keep (default: 50.0)')
    args = parser.parse_args()

    config = ConfigFactory.parse_file('./config.conf')[args.c]

    print(f"Running regression with:", flush=True)

    model = RegressionModel(config)

    with open("./data/mfc/immigration/clustering/clusters_250_0.5_.pickle", 'rb') as f:
        clustering_data = pickle.load(f)

    data = model.create_dataset(config, clustering_data, 
                                args.use_centroid_filtering, args.centroid_top_k_percent)
    model.regression(config, data, 
                     args.use_centroid_filtering, args.centroid_top_k_percent)