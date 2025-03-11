import json
# import yaml
import os
import argparse
from dotenv import load_dotenv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import numpy as np

from character.cluster.run import KmeansClusterer


# load the environment variables.
load_dotenv()
DATA_PATH = os.environ.get("DATA_DIR")
SCRIPT_PATH = os.path.dirname(os.path.realpath(__file__))

"""
This script...
"""

class KmeansSupClusterer(KmeansClusterer):

    def __init__(self, data: list[str], initial_centers: list[str]):
        super().__init__(data)
        self.k = len(initial_centers)
        self.initial_centers = self.vectorizer.transform(initial_centers)
        self.initial_centers = self.initial_centers.toarray()

    def fit(self):
        kmeans = KMeans(n_clusters=self.k, init=self.initial_centers, random_state=14)
        kmeans.fit(self.vectors)
        labels = kmeans.labels_
        return labels

    def run(self):
        labels = self.fit()
        self.cluster_dict = super().clean_labels(labels)

        

if __name__ == "__main__":
    # args
    parser = argparse.ArgumentParser(
        description="Analyze a dataset using Kmeans clustering."
    )
    parser.add_argument(
        "--filename", type=str,
        help="Name of the file of LLM output in DATA_DIR to analyze.",
    )

    args = parser.parse_args()

    # read data
    with open(os.path.join(DATA_PATH, "results", args.filename), "r") as f:
        data = json.load(f)['data']
    
    character_strings = [v['annotation']['characters'] for v in data.values()]
    characters = [item for sublist in character_strings for item in sublist]  # flatten the list of lists

    # declare initial centers
    if 'imm' in args.filename:
        initial_centers = [
            "undocumented immigrants",
            "documented immigrants",
            "law enforcement",
            "families",
            "politicians",
            "voters",
            "activists",
            "researchers",
            "journalists",
            "business owners"
            
        ]
    elif 'gun' in args.filename:
        initial_centers = [
            "gun owners",
            "gun control supporters",
            "law enforcement",
            "politicians",
            "voters",
            "activists",
            "researchers",
            "journalists",
            "business owners"
        ]
    else:
        raise ValueError(f"{args.filename} not recognized.")

    # call class
    clusterer = KmeansSupClusterer(characters, initial_centers)
    clusterer.run()
    # clusterer.print_clusters()
    out_filename = args.filename.split(".")[0] + f"_sup_clusters.json"
    out_filepath = os.path.join(DATA_PATH, "results", "clustering", out_filename)
    clusterer.save_clusters(out_filepath)