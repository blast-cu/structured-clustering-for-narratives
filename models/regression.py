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


class RegressionModel:
    def __init__(self, config):
        self.config = config
        random.seed(self.config["seed"])
        np.random.seed(self.config["seed"])

    def create_dataset(self, config, clustering_data):
        print("Creating dataset...", flush=True)

        with open(self.config["relations_path"], 'rb') as f:
            corpus = pickle.load(f)

        with open(config["processed_chains_path"], 'rb') as f:
            processed_chains = pickle.load(f)

        dataset = {}
        for chain_idx in processed_chains['processed_chains']:
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

    def regression(self, config, data):
        print("Running regression...", flush=True)
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
            ('classifier', LogisticRegression(solver='lbfgs', penalty='l2', C=0.5))
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

        return test_accuracy, f1_score


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Regression")
    parser.add_argument('-c', metavar='CONF', default='base', help='configuration (see config.conf)')
    args = parser.parse_args()

    config = ConfigFactory.parse_file('./config.conf')[args.c]

    print(f"Running regression with:", flush=True)

    model = RegressionModel(config)

    with open("./data/mfc/immigration/clustering/clusters_250_0.5_.pickle", 'rb') as f:
        clustering_data = pickle.load(f)

    data = model.create_dataset(config, clustering_data)
    model.regression(config, data)