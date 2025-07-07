import argparse
import csv
import json
import os
import pickle
import random
import sys

from pyhocon import ConfigFactory

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import the schemas module and create an alias for pickle compatibility
import annotation.schemas as schemas
sys.modules['schemas'] = schemas

def generate_data(annotated_chains, config, sample_size=50):
    # Filter keys where event_chains dict has non-zero length
    valid_keys = [key for key, item in annotated_chains.items() 
                  if 'event_chains' in item and len(item['event_chains']) > 0]
    
    sampled_keys = random.sample(valid_keys, min(sample_size, len(valid_keys)))
    sampled_chains = {key: annotated_chains[key] for key in sampled_keys}
    
    # From each sampled chain, randomly sample one item from event_chains
    sampled_event_chains = {}
    for key, item in sampled_chains.items():
        event_chain_keys = list(item['event_chains'].keys())
        # Filter event chains that have non-empty annotation
        valid_event_keys = [ek for ek in event_chain_keys 
                           if 'annotation' in item['event_chains'][ek] 
                           and item['event_chains'][ek]['annotation']]
        
        if valid_event_keys:
            random_event_key = random.choice(valid_event_keys)
            sampled_event_chains[key] = {
                'original_item': item,
                'sampled_event_chain': item['event_chains'][random_event_key]
            }

    out_dict = []
    for key, item in sampled_event_chains.items():
        row = {
            'doc_text': item['original_item']['text'],
            'event_chain': item['sampled_event_chain']['event_chain'],
            'chain_text': item['sampled_event_chain']['chain_text'],
            'events_present': "",
            'entities_present': "",
            'correct_verbalization': "",
            'no_hallucinations': "",
            'verbalization_score': "",
            'char_annotation': json.dumps(item['sampled_event_chain']['annotation'], indent=2),
            'all_entities': "",
            'correct_char_groups': "",
            'correct_roles': "",
            'correct_stance': "",
            'no_hallucinations': "",
            'char_score': ""
        }
        out_dict.append(row)

    with open(config['chain_char_annotations_path'], 'w', newline='') as csvfile:
        fieldnames = ['doc_text', 'event_chain', 'chain_text', 'events_present', 
                     'entities_present', 'correct_verbalization', 'no_hallucinations', 'verbalization_score',
                     'char_annotation', 'all_entities', 'correct_char_groups', 
                     'correct_roles', 'correct_stance', 'no_hallucinations', 'char_score']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames, delimiter='\t')
        writer.writeheader()
        writer.writerows(out_dict)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='KMeans Clustering')
    parser.add_argument('-c', metavar='CONF', default='base', help='configuration (see config.conf)')

    args = parser.parse_args()
    config = ConfigFactory.parse_file('./config.conf')[args.c]
    
    with open(config["char_event_chains_path"], 'rb') as f:
        annotated_chains = pickle.load(f)

    random.seed(config["seed"])
        
    generate_data(annotated_chains, config, sample_size=50)