import argparse
import itertools
import pickle
import random
import sys
import os
import shelve

from collections import Counter
from typing import List, Tuple

from pyhocon import ConfigFactory
from tqdm import tqdm

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import the schemas module and create an alias for pickle compatibility
import annotation.schemas as schemas
sys.modules['schemas'] = schemas


def process_event_chains(data):
    chain_idx = 0
    processed_chains = {}
    chain_sents = []
    for doc_id in data:
        if not data[doc_id]["event_chains"]:
            continue
        for event_chain in data[doc_id]["event_chains"]:
            chain_obj = data[doc_id]["event_chains"][event_chain]
            chain_obj['doc_id'] = doc_id
            processed_chains[chain_idx] = chain_obj
            chain_idx += 1

            chain_sents.append(chain_obj['chain_text'])
    print(f"Processed {len(processed_chains)} event chains.")
    return processed_chains, chain_sents


def compute_constraints_generator(processed_chains, chain_group_roles, batch_size=10000000):
    """Generator that yields constraint batches for memory efficiency"""
    batch:List[Tuple[int, int]] = []
    
    # Calculate total combinations
    n = len(processed_chains)
    total_combinations = n * (n - 1) // 2
    print(f"Total combinations to process: {total_combinations}")
    
    for k1, k2 in tqdm(itertools.combinations(processed_chains.keys(), 2), 
                       total=total_combinations, 
                       desc="Processing constraints"):
        # Check if both have same character groups with same roles
        if chain_group_roles[k1] != chain_group_roles[k2]:
            batch.append((k1, k2))
            if len(batch) >= batch_size:
                yield batch
                batch = []
    
    # Yield remaining constraints
    if batch:
        yield batch


def compute_constraints(processed_chains, normalize_groups=False, constraints_file=None):
    target_roles = {'Hero', 'Victim', 'Threat'}
    
    # Define normalize_group function once
    def normalize_group(group):
        immigration_groups = {'Immigrants', 'Refugees', 'Asylum Seekers', 'Workers'}
        return 'Immigration_People' if group in immigration_groups else group
    
    # Pre-compute candidates, resolutions, and normalizations for all chains
    print("Pre-computing chain group roles...")
    chain_group_roles = {}
    for chain_id, chain_data in tqdm(processed_chains.items(), desc="Computing group roles"):
        # Extract candidates
        candidates = [char for char in chain_data['annotation']['characters']
                     if char['role'] in target_roles and char['character_group'] != 'Other']
        
        # Get stance (only one per chain)
        stance = chain_data['annotation']['stance'] if 'stance' in chain_data['annotation'] else None
        
        # Resolve character group conflicts
        candidates = resolve_character_groups(candidates)
        
        # Add stance to the resolved groups
        for group in candidates:
            candidates[group] = {'role': candidates[group], 'stance': stance}
        
        # Normalize groups (optional)
        if normalize_groups:
            candidates = {normalize_group(group): role for group, role in candidates.items()}

        chain_group_roles[chain_id] = candidates
    
    # Use shelve for memory-efficient constraint storage
    if constraints_file:
        print(f"Writing constraints incrementally to {constraints_file}")
        processed_combinations = 0
        
        # Use shelve to store constraints as a persistent dictionary
        with shelve.open(constraints_file, 'c') as constraints_db:
            for batch in compute_constraints_generator(processed_chains, chain_group_roles):
                # Create a batch dictionary to update shelve efficiently
                batch_dict = {}
                for k1, k2 in batch:
                    # Use repr to create a string key that preserves the tuple structure
                    key = repr((k1, k2))
                    batch_dict[key] = 1
                
                # Batch update the shelve database
                constraints_db.update(batch_dict)
                processed_combinations += len(batch)
                if processed_combinations % 1000000 == 0:  # Print every 1M constraints
                    print(f"Processed {processed_combinations} combinations so far...")
        
        print(f"Total constraints written: {processed_combinations}")
        return None, chain_group_roles  # Return None for constraints since they're on disk
    else:
        # Fall back to in-memory approach for small datasets
        constraints = []
        for batch in compute_constraints_generator(processed_chains, chain_group_roles):
            constraints.extend(batch)
        return constraints, chain_group_roles

def resolve_character_groups(candidates):
    """Resolve conflicts within character groups using majority voting"""
    group_roles = {}

    # Group characters by character_group
    for char in candidates:
        group = char['character_group']
        if group not in group_roles:
            group_roles[group] = []
        group_roles[group].append(char['role'])

    # Resolve conflicts using majority voting
    resolved_groups = {}
    for group, roles in group_roles.items():
        if len(set(roles)) > 1:  # Multiple different roles for same group
            # Use majority voting
            role_counts = Counter(roles)
            max_count = max(role_counts.values())
            tied_roles = [role for role, count in role_counts.items() if count == max_count]

            if len(tied_roles) == 1:
                majority_role = tied_roles[0]
            else:
                # Tie-breaking: choose randomly
                majority_role = random.choice(tied_roles)

            resolved_groups[group] = majority_role
        else:
            # No conflict, use the single role
            resolved_groups[group] = roles[0]

    return resolved_groups


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process event chains')
    parser.add_argument('-c', metavar='CONF', default='base', help='configuration (see config.conf)')

    args = parser.parse_args()
    config = ConfigFactory.parse_file('./config.conf')[args.c]

    random.seed(config["seed"])

    with open(config["char_event_chains_path"], 'rb') as f:
        annotated_chains = pickle.load(f)

    processed_chains, chain_sents = process_event_chains(annotated_chains)
    
    # Use memory-efficient approach with disk writing for large datasets
    constraints_file = config["processed_chains_path"].replace('.pkl', '_constraints.pkl')
    _, chain_group_roles = compute_constraints(
        processed_chains, 
        normalize_groups=True, 
        constraints_file=config["constraints_path"]
    )

    output = {
        "processed_chains": processed_chains,
        "chain_sents": chain_sents,
        "chain_group_roles": chain_group_roles
    }
    with open(config["processed_chains_path"], 'wb') as f:
        pickle.dump(output, f)