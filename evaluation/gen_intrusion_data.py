import argparse
import json
import pickle
import random
import csv
from typing import Dict, List, Tuple, Any
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
from pyhocon import ConfigFactory


class IntrusionDataGenerator:
    def __init__(self, processed_chains_path: str, random_seed: int = 42):
        """
        Initialize the intrusion data generator.
        
        Args:
            processed_chains_path: Path to processed event chains pickle file
            random_seed: Random seed for reproducibility
        """
        self.random_seed = random_seed
        random.seed(random_seed)
        np.random.seed(random_seed)
        
        # Load processed chains
        with open(processed_chains_path, 'rb') as f:
            data = pickle.load(f)
            self.processed_chains = data['processed_chains']
            self.chain_sents = data['chain_sents']
        
        print(f"Loaded {len(self.processed_chains)} processed event chains")
        
        # Verify index mapping
        self._verify_index_mapping()
    
    def _verify_index_mapping(self):
        """Verify that indices in processed_chains match chain_sents indices."""
        max_idx = max(self.processed_chains.keys())
        if max_idx >= len(self.chain_sents):
            raise ValueError(f"Index mismatch: max processed_chains index {max_idx} >= chain_sents length {len(self.chain_sents)}")
        
        # Spot check a few mappings
        for idx in random.sample(list(self.processed_chains.keys()), min(5, len(self.processed_chains))):
            chain_text_from_dict = self.processed_chains[idx]['chain_text']
            chain_text_from_list = self.chain_sents[idx]
            if chain_text_from_dict != chain_text_from_list:
                raise ValueError(f"Text mismatch at index {idx}")
        
        print(" Index mapping verified successfully")
    
    def _compute_cluster_statistics(self, clustering_data: Dict) -> Dict:
        """Compute statistics for each cluster to determine difficulty levels."""
        embeddings = clustering_data['embeddings']
        labels = clustering_data['labels']
        centroids = clustering_data['cluster_centers']
        
        cluster_stats = {}
        
        for cluster_id in range(clustering_data['number_cluster']):
            cluster_indices = np.where(labels == cluster_id)[0]
            if len(cluster_indices) < 2:  # Need at least 2 points for positive examples
                continue
                
            cluster_embeddings = embeddings[cluster_indices]
            centroid = centroids[cluster_id]
            
            # Calculate distances from points to centroid
            distances_to_centroid = euclidean_distances(cluster_embeddings, centroid.reshape(1, -1)).flatten()
            
            # Calculate inter-cluster distances
            other_centroids = np.delete(centroids, cluster_id, axis=0)
            min_centroid_distance = np.min(euclidean_distances(centroid.reshape(1, -1), other_centroids))
            
            cluster_stats[cluster_id] = {
                'indices': cluster_indices,
                'size': len(cluster_indices),
                'cohesion': np.std(distances_to_centroid),  # Lower is more cohesive
                'min_centroid_distance': min_centroid_distance,  # Higher means more separated
                'distances_to_centroid': distances_to_centroid
            }
        
        return cluster_stats
    
    def _categorize_difficulty(self, cluster_stats: Dict) -> Dict[str, List[int]]:
        """Categorize clusters by difficulty level."""
        # Extract metrics for percentile calculation
        cohesions = [stats['cohesion'] for stats in cluster_stats.values()]
        centroid_distances = [stats['min_centroid_distance'] for stats in cluster_stats.values()]
        
        cohesion_33 = np.percentile(cohesions, 33)
        cohesion_67 = np.percentile(cohesions, 67)
        centroid_33 = np.percentile(centroid_distances, 33)
        centroid_67 = np.percentile(centroid_distances, 67)
        
        difficulty_clusters = {'easy': [], 'medium': [], 'hard': []}
        
        for cluster_id, stats in cluster_stats.items():
            cohesion = stats['cohesion']
            centroid_dist = stats['min_centroid_distance']
            
            # Easy: high separation, low cohesion (tight clusters far apart)
            if centroid_dist >= centroid_67 and cohesion <= cohesion_33:
                difficulty_clusters['easy'].append(cluster_id)
            # Hard: low separation, high cohesion (loose clusters close together)
            elif centroid_dist <= centroid_33 and cohesion >= cohesion_67:
                difficulty_clusters['hard'].append(cluster_id)
            else:
                difficulty_clusters['medium'].append(cluster_id)
        
        return difficulty_clusters
    
    def _sample_positive_examples(self, cluster_stats: Dict, cluster_id: int) -> Tuple[int, int]:
        """Sample two positive examples from a cluster."""
        cluster_indices = cluster_stats[cluster_id]['indices']
        distances = cluster_stats[cluster_id]['distances_to_centroid']
        
        # Sort by distance to centroid and take top 25%
        sorted_indices = cluster_indices[np.argsort(distances)]
        top_25_percent = max(1, int(len(sorted_indices) * 0.25))
        candidate_indices = sorted_indices[:top_25_percent]
        
        # Sample two points ensuring they don't have the same doc_id
        max_attempts = 50
        for _ in range(max_attempts):
            if len(candidate_indices) < 2:
                # Fall back to closest two if not enough candidates
                positive_1_idx = sorted_indices[0]
                positive_2_idx = sorted_indices[min(1, len(sorted_indices) - 1)]
                break
                
            # Sample two different indices
            sampled_indices = random.sample(candidate_indices.tolist(), 2)
            positive_1_idx, positive_2_idx = sampled_indices
            
            # Check if they have different doc_ids
            doc_id_1 = self.processed_chains[positive_1_idx]['doc_id']
            doc_id_2 = self.processed_chains[positive_2_idx]['doc_id']
            
            if doc_id_1 != doc_id_2:
                break
        else:
            # If we couldn't find different doc_ids in top 25%, expand search
            for i in range(len(sorted_indices) - 1):
                for j in range(i + 1, len(sorted_indices)):
                    idx1, idx2 = sorted_indices[i], sorted_indices[j]
                    doc_id_1 = self.processed_chains[idx1]['doc_id']
                    doc_id_2 = self.processed_chains[idx2]['doc_id']
                    if doc_id_1 != doc_id_2:
                        positive_1_idx, positive_2_idx = idx1, idx2
                        break
                else:
                    continue
                break
            else:
                # Last resort: take the two closest regardless of doc_id
                positive_1_idx = sorted_indices[0]
                positive_2_idx = sorted_indices[min(1, len(sorted_indices) - 1)]
        
        return positive_1_idx, positive_2_idx
    
    def _sample_intruder(self, clustering_data: Dict, source_cluster_id: int, 
                        difficulty: str, cluster_stats: Dict) -> int:
        """Sample an intruder from a different cluster."""
        centroids = clustering_data['cluster_centers']
        source_centroid = centroids[source_cluster_id]
        
        # Find candidate clusters (excluding source)
        candidate_clusters = [cid for cid in cluster_stats.keys() if cid != source_cluster_id]
        
        if difficulty == 'easy':
            # Choose from distant clusters
            distances = euclidean_distances(source_centroid.reshape(1, -1), centroids[candidate_clusters]).flatten()
            # Take clusters from top 50% of distances
            distant_threshold = np.percentile(distances, 50)
            distant_clusters = [candidate_clusters[i] for i, d in enumerate(distances) if d >= distant_threshold]
            target_cluster = random.choice(distant_clusters) if distant_clusters else random.choice(candidate_clusters)
        
        elif difficulty == 'hard':
            # Choose from nearby clusters
            distances = euclidean_distances(source_centroid.reshape(1, -1), centroids[candidate_clusters]).flatten()
            # Take clusters from bottom 50% of distances
            close_threshold = np.percentile(distances, 50)
            close_clusters = [candidate_clusters[i] for i, d in enumerate(distances) if d <= close_threshold]
            target_cluster = random.choice(close_clusters) if close_clusters else random.choice(candidate_clusters)
        
        else:  # medium
            target_cluster = random.choice(candidate_clusters)
        
        # Sample a representative point from target cluster
        target_indices = cluster_stats[target_cluster]['indices']
        target_distances = cluster_stats[target_cluster]['distances_to_centroid']
        
        # Sample from points reasonably close to target centroid (top 70%)
        close_indices = target_indices[target_distances <= np.percentile(target_distances, 70)]
        intruder_idx = random.choice(close_indices)
        
        return intruder_idx
    
    def generate_intrusion_examples(self, clustering_data_1: Dict, clustering_data_2: Dict,
                                  total_triplets: int = 50,
                                  difficulty_distribution: Tuple[float, float, float] = (0.3, 0.4, 0.3)) -> Tuple[Dict[str, List[Dict]], Dict[str, Dict]]:
        """
        Generate intrusion detection examples for two clustering methods.
        
        Args:
            clustering_data_1: First clustering method results
            clustering_data_2: Second clustering method results  
            total_triplets: Total number of triplets to generate per method
            difficulty_distribution: Tuple of (easy_ratio, medium_ratio, hard_ratio)
            
        Returns:
            Tuple of (examples_dict_by_method, solutions_dict_by_method)
        """
        # Ensure difficulty distribution sums to 1
        easy_ratio, medium_ratio, hard_ratio = difficulty_distribution
        if abs(sum(difficulty_distribution) - 1.0) > 1e-6:
            raise ValueError(f"Difficulty distribution must sum to 1.0, got {sum(difficulty_distribution)}")
        
        # Calculate triplets per difficulty level per method
        easy_per_method = int(total_triplets * easy_ratio)
        medium_per_method = int(total_triplets * medium_ratio)
        hard_per_method = total_triplets - easy_per_method - medium_per_method  # Remainder goes to hard
        
        print(f"Generating {total_triplets} triplets per method:")
        print(f"  Per difficulty: Easy={easy_per_method}, Medium={medium_per_method}, Hard={hard_per_method}")
        
        examples_by_method = {}
        solutions_by_method = {}
        
        # Pre-compute cluster statistics for both methods to find common difficulty constraints
        method_cluster_stats = {}
        method_difficulty_clusters = {}
        
        for method_name, clustering_data in [("method_1", clustering_data_1), ("method_2", clustering_data_2)]:
            cluster_stats = self._compute_cluster_statistics(clustering_data)
            difficulty_clusters = self._categorize_difficulty(cluster_stats)
            method_cluster_stats[method_name] = cluster_stats
            method_difficulty_clusters[method_name] = difficulty_clusters
            
            print(f"{method_name} difficulty distribution: Easy={len(difficulty_clusters['easy'])}, "
                  f"Medium={len(difficulty_clusters['medium'])}, Hard={len(difficulty_clusters['hard'])}")
        
        # Find the minimum available clusters across both methods for each difficulty
        min_easy = min(len(method_difficulty_clusters['method_1']['easy']), 
                      len(method_difficulty_clusters['method_2']['easy']))
        min_medium = min(len(method_difficulty_clusters['method_1']['medium']), 
                        len(method_difficulty_clusters['method_2']['medium']))
        min_hard = min(len(method_difficulty_clusters['method_1']['hard']), 
                      len(method_difficulty_clusters['method_2']['hard']))
        
        # Adjust counts to ensure both methods can generate the same number
        actual_easy = min(easy_per_method, min_easy * 10)  # Allow up to 10 examples per cluster
        actual_medium = min(medium_per_method, min_medium * 10)
        actual_hard = min(hard_per_method, min_hard * 10)
        
        print(f"Adjusted counts to ensure consistency:")
        print(f"  Easy: {actual_easy} (was {easy_per_method})")
        print(f"  Medium: {actual_medium} (was {medium_per_method})")
        print(f"  Hard: {actual_hard} (was {hard_per_method})")
        
        # Process both clustering methods
        for method_idx, (method_name, clustering_data) in enumerate([
            ("method_1", clustering_data_1), 
            ("method_2", clustering_data_2)
        ]):
            print(f"Processing {method_name}...")
            
            # Initialize lists for this method
            examples_by_method[method_name] = []
            solutions_by_method[method_name] = {}
            example_id = 0
            
            # Use pre-computed statistics
            cluster_stats = method_cluster_stats[method_name]
            difficulty_clusters = method_difficulty_clusters[method_name]
            
            # Generate examples for each difficulty level using adjusted counts
            difficulty_counts = {
                'easy': actual_easy,
                'medium': actual_medium,
                'hard': actual_hard
            }
            
            for difficulty, count in difficulty_counts.items():
                available_clusters = difficulty_clusters[difficulty]
                if len(available_clusters) == 0:
                    print(f"  Warning: No {difficulty} clusters available for {method_name}")
                    continue
                
                for _ in range(count):
                    # Sample source cluster
                    source_cluster = random.choice(available_clusters)
                    
                    # Sample positive examples
                    pos1_idx, pos2_idx = self._sample_positive_examples(cluster_stats, source_cluster)
                    
                    # Sample intruder
                    intruder_idx = self._sample_intruder(clustering_data, source_cluster, difficulty, cluster_stats)
                    
                    # Get sentence texts
                    pos1_text = self.chain_sents[pos1_idx]
                    pos2_text = self.chain_sents[pos2_idx]
                    intruder_text = self.chain_sents[intruder_idx]
                    
                    # Create triplet with random order
                    sentences = [pos1_text, pos2_text, intruder_text]
                    sentence_roles = ['positive', 'positive', 'intruder']
                    
                    # Shuffle while tracking intruder position
                    combined = list(zip(sentences, sentence_roles))
                    random.shuffle(combined)
                    shuffled_sentences, shuffled_roles = zip(*combined)
                    
                    # Find intruder position after shuffle
                    intruder_position = shuffled_roles.index('intruder')
                    
                    # Create example dict
                    example = {
                        str(i): sent for i, sent in enumerate(shuffled_sentences)
                    }
                    examples_by_method[method_name].append(example)
                    
                    # Store solution
                    solutions_by_method[method_name][example_id] = {
                        'answer': intruder_position,
                        'difficulty': difficulty,
                        'method': method_name,
                        'source_cluster': int(source_cluster),
                        'positive_indices': [int(pos1_idx), int(pos2_idx)],
                        'intruder_index': int(intruder_idx)
                    }
                    
                    example_id += 1
            
            # Shuffle the examples to mix difficulty levels
            combined_data = list(zip(examples_by_method[method_name], 
                                   solutions_by_method[method_name].values()))
            random.shuffle(combined_data)
            
            # Reassign shuffled data with new sequential IDs
            examples_by_method[method_name] = [ex for ex, _ in combined_data]
            solutions_by_method[method_name] = {
                i: {**sol, 'original_id': i} for i, (_, sol) in enumerate(combined_data)
            }
            
            print(f"  Generated and shuffled {len(examples_by_method[method_name])} examples for {method_name}")
        
        total_examples = sum(len(examples) for examples in examples_by_method.values())
        print(f"Generated {total_examples} total examples across both methods")
        return examples_by_method, solutions_by_method
    
    def save_data_separate(self, examples_by_method: Dict[str, List[Dict]], 
                          solutions_by_method: Dict[str, Dict], 
                          output_prefix: str):
        """Save examples and solutions as separate files for each method."""
        for method_name in examples_by_method.keys():
            # Create file names with method suffix
            csv_path = f"{output_prefix}_{method_name}.tsv"
            solutions_path = f"{output_prefix}_{method_name}_solutions.pkl"
            
            # Save examples as tab-separated CSV with pretty-printed JSON
            with open(csv_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f, delimiter='\t')
                writer.writerow(['example'])  # Header
                for example in examples_by_method[method_name]:
                    # Pretty print JSON with each sentence on a new line
                    json_str = json.dumps(example, ensure_ascii=False, indent=2)
                    writer.writerow([json_str])
            
            # Save solutions as pickle
            with open(solutions_path, 'wb') as f:
                pickle.dump(solutions_by_method[method_name], f, protocol=pickle.HIGHEST_PROTOCOL)
            
            print(f"Saved {len(examples_by_method[method_name])} examples to {csv_path}")
            print(f"Saved solutions to {solutions_path}")


def evaluate_predictions(csv_path: str, solutions_path: str, predictions_path: str):
    """
    Evaluate predictions against ground truth.
    
    Args:
        csv_path: Path to examples CSV
        solutions_path: Path to solutions pickle file
        predictions_path: Path to predictions file (one answer per line)
    """
    # Load solutions
    with open(solutions_path, 'rb') as f:
        solutions = pickle.load(f)
    
    # Load predictions
    with open(predictions_path, 'r') as f:
        predictions = [int(line.strip()) for line in f]
    
    if len(predictions) != len(solutions):
        raise ValueError(f"Mismatch: {len(predictions)} predictions vs {len(solutions)} examples")
    
    # Calculate overall accuracy
    correct = sum(1 for i, pred in enumerate(predictions) if pred == solutions[i]['answer'])
    total_accuracy = correct / len(predictions) * 100
    
    print(f"Overall Accuracy: {correct}/{len(predictions)} = {total_accuracy:.2f}%")
    
    # Breakdown by difficulty
    difficulty_stats = {}
    for difficulty in ['easy', 'medium', 'hard']:
        difficulty_indices = [i for i, sol in solutions.items() if sol['difficulty'] == difficulty]
        if difficulty_indices:
            difficulty_correct = sum(1 for i in difficulty_indices if predictions[i] == solutions[i]['answer'])
            difficulty_total = len(difficulty_indices)
            difficulty_accuracy = difficulty_correct / difficulty_total * 100
            difficulty_stats[difficulty] = {
                'correct': difficulty_correct,
                'total': difficulty_total,
                'accuracy': difficulty_accuracy
            }
            print(f"{difficulty.capitalize()} Accuracy: {difficulty_correct}/{difficulty_total} = {difficulty_accuracy:.2f}%")
    
    # Breakdown by method
    method_stats = {}
    for method in ['method_1', 'method_2']:
        method_indices = [i for i, sol in solutions.items() if sol['method'] == method]
        if method_indices:
            method_correct = sum(1 for i in method_indices if predictions[i] == solutions[i]['answer'])
            method_total = len(method_indices)
            method_accuracy = method_correct / method_total * 100
            method_stats[method] = {
                'correct': method_correct,
                'total': method_total,
                'accuracy': method_accuracy
            }
            print(f"{method.replace('_', ' ').title()} Accuracy: {method_correct}/{method_total} = {method_accuracy:.2f}%")
    
    return {
        'overall': {'correct': correct, 'total': len(predictions), 'accuracy': total_accuracy},
        'by_difficulty': difficulty_stats,
        'by_method': method_stats
    }


def main():
    parser = argparse.ArgumentParser(description="Generate intrusion detection task data")
    parser.add_argument('-c', '--config', default='base', help='Configuration name')
    parser.add_argument('--output-csv', default='intrusion_examples.tsv', help='Output CSV path')
    parser.add_argument('--output-solutions', default='intrusion_solutions.pkl', help='Output solutions path')
    parser.add_argument('--total-triplets', type=int, default=50,
                       help='Total number of triplets to generate across both methods (default: 300)')
    parser.add_argument('--difficulty-split', nargs=3, type=float, default=[0.1, 0.5, 0.4],
                       help='Difficulty distribution: easy medium hard (must sum to 1.0, default: 0.3 0.4 0.3)')
    parser.add_argument('--evaluate', help='Path to predictions file for evaluation')
    
    args = parser.parse_args()
    
    if args.evaluate:
        # Evaluation mode
        evaluate_predictions(args.output_csv, args.output_solutions, args.evaluate)
    else:
        # Generation mode
        config = ConfigFactory.parse_file('./config.conf')[args.config]
        
        # Load clustering results
        with open("./data/mfc/immigration/clustering/clusters_100_0.0.pickle", 'rb') as f:
            clustering_data_1 = pickle.load(f)
        
        with open("./data/mfc/immigration/clustering/clusters_100_0.5.pickle", 'rb') as f:
            clustering_data_2 = pickle.load(f)
        
        # Generate intrusion data
        generator = IntrusionDataGenerator(config["processed_chains_path"], config["seed"])
        examples, solutions = generator.generate_intrusion_examples(
            clustering_data_1, clustering_data_2, 
            total_triplets=args.total_triplets,
            difficulty_distribution=tuple(args.difficulty_split)
        )
        
        # Save data separately for each method
        output_prefix = args.output_csv.replace('.tsv', '').replace('.csv', '')
        generator.save_data_separate(examples, solutions, output_prefix)


if __name__ == "__main__":
    main()