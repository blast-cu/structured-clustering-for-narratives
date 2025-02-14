import argparse
import numpy as np
import pickle

from pyhocon import ConfigFactory
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

event_chains = {}
event_chain_to_doc = {}
constraints = []
emb_idx_to_chain_idx = {}


def process_chains(annotated_docs):
    for doc_idx, doc in annotated_docs.items():
        for chain_idx, chain in doc['event_chains'].items():
            if 'CAUSAL' in chain['event_chain']:
                event_chains[chain_idx] = chain
                event_chain_to_doc[chain_idx] = doc_idx

def generate_constraints():
    for chain_idx_1, chain_1 in tqdm(event_chains.items()):
        for chain_idx_2, chain_2 in event_chains.items():
            if chain_idx_1 != chain_idx_2 and chain_1['annotation'] is not None and chain_2['annotation'] is not None:
                if (chain_1['annotation']['role'] == chain_2['annotation']['role'] or
                            chain_1['annotation']['stance'] == chain_2['annotation']['stance']):
                        if [chain_idx_1, chain_idx_2] not in constraints and [chain_idx_2, chain_idx_1] not in constraints:
                            constraints.append([chain_idx_1, chain_idx_2])

def embed_chains(config):
    embs_input = []
    for chain_idx, chain in event_chains.items():
        embs_input.append(chain['chain_text'])
        emb_idx_to_chain_idx[len(embs_input) - 1] = chain_idx

    embedding_model = SentenceTransformer(config["cluster_model"])
    embs = embedding_model.encode(embs_input, normalize_embeddings=True)
    return embs

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Post Processing')
    parser.add_argument('-c', metavar='CONF', default='base', help='configuration (see config.conf)')

    args = parser.parse_args()
    config = ConfigFactory.parse_file('./config.conf')[args.c]

    with open(config["annotated_event_chains_path"], "rb") as f:
        annotated_docs = pickle.load(f)

    process_chains(annotated_docs)
    generate_constraints()
    embs = embed_chains(config)

    out = {
        "chains": event_chains,
        "chain_to_doc": event_chain_to_doc,
        "constraints": np.array(constraints),
        "emb_idx_to_chain_idx": emb_idx_to_chain_idx,
        "embs": embs
    }

    with open(config["cluster_embs_path"], 'wb') as f:
        pickle.dump(out, f, protocol=pickle.HIGHEST_PROTOCOL)
