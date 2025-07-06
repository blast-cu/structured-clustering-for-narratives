import argparse
import pickle

from pyhocon import ConfigFactory


def load_data(config):
    with open(config["processed_chains_path"], "rb") as f:
        chain_data = pickle.load(f)

    constraints = {}
    with open(config["constraints_path"], "rb") as f:
        while True:
            try:
                batch = pickle.load(f)
                # Convert batch (list of tuples) to dict entries
                for k1, k2 in batch:
                    constraints[(k1, k2)] = 1
            except EOFError:
                break

    with open(config["clusters_path"] + "clusters_1000_0.0.pickle", "rb") as f:
        clusters = pickle.load(f)

    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='KMeans Clustering')
    parser.add_argument('-c', metavar='CONF', default='base', help='configuration (see config.conf)')

    args = parser.parse_args()
    config = ConfigFactory.parse_file('./config.conf')[args.c]

    load_data(config)