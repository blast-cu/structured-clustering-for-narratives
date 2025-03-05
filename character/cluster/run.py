import json
# import yaml
import os
import argparse
from dotenv import load_dotenv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans


# load the environment variables.
load_dotenv()
DATA_PATH = os.environ.get("DATA_DIR")
SCRIPT_PATH = os.path.dirname(os.path.realpath(__file__))

"""
This script...
"""

class KmeansClusterer():

    def __init__(self, data: list[str]):

        data = [d.lower().strip() for d in data]
        data = list(set(data))  # remove duplicates

        self.vectorizer = TfidfVectorizer()
        self.data = data
        self.vectors = self.vectorizer.fit_transform(data)

        # self.frequency = {}
        # for d in data:
        #     if d not in self.frequency:
        #         self.frequency[d] = 1
        #     else:
        #         self.frequency[d] += 1


    def run(self, k):

        kmeans = KMeans(n_clusters=k, n_init='auto', random_state=14) # Specify the number of clusters
        kmeans.fit(self.vectors)
        labels = kmeans.labels_

        self.cluster_dict = {}
        for i, label in enumerate(labels):
            current_data = self.data[i]
            label = str(label)
            if label not in self.cluster_dict:
                self.cluster_dict[label] = []
            self.cluster_dict[label].append(current_data)
        
        for cluster, item in self.cluster_dict.items():  # sort alphabetically
            self.cluster_dict[cluster] = sorted(item)

        # sort keys
        self.cluster_dict = dict(sorted(self.cluster_dict.items(), key=lambda item: int(item[0])))

    def print_clusters(self):
        for cluster, item in self.cluster_dict.items():
            print(f">> Cluster {cluster}")
            for d in item:
                print(d)
            print("\n")

    def save_clusters(self, file_path):
        with open(file_path, "w") as f:
            json.dump(self.cluster_dict, f, indent=4)


if __name__ == "__main__":
    # args
    parser = argparse.ArgumentParser(
        description="Analyze a dataset using Kmeans clustering."
    )
    parser.add_argument(
        "--filename", type=str,
        help="Name of the file of LLM output in DATA_DIR to analyze.",
    )
    parser.add_argument(
        "--k", type=int,
        help="Number of clusters.",
    )
    args = parser.parse_args()

    # read data
    with open(os.path.join(DATA_PATH, "results", args.filename), "r") as f:
        data = json.load(f)['data']
    
    character_strings = [v['annotation']['characters'] for v in data.values()]
    characters = [item for sublist in character_strings for item in sublist]  # flatten the list of lists

    # call class

    clusterer = KmeansClusterer(characters)
    clusterer.run(k=args.k)
    # clusterer.print_clusters()
    out_filename = args.filename.split(".")[0] + f"_{args.k}_clusters.json"
    out_filepath = os.path.join(DATA_PATH, "results", "clustering", out_filename)
    clusterer.save_clusters(out_filepath)