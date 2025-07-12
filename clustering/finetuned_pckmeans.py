import argparse
import os
import pickle
import random
from typing import Optional, Dict, List, Tuple

import numpy as np
import torch
from datasets import Dataset
from numpy import ndarray
from pyhocon import ConfigFactory
from sklearn.cluster import KMeans
from sentence_transformers import SentenceTransformer, SentenceTransformerTrainingArguments, SentenceTransformerTrainer
from sentence_transformers.losses import CosineSimilarityLoss, TripletLoss, MultipleNegativesRankingLoss
from tqdm import tqdm

from clustering.initializer.cl_kmeans_plus_plus import KMeansPlusPlusInit, InitializationStrategy
from clustering.weighted_pckmeans import ConstrainedKMeans
from clustering.metrics.purity import Purity
from utils.constraint_flat_db import ConstraintFlatDB
from utils.constraints_graph_db import ConstraintGraphDB


class SBERTConstrainedClusteringTrainer:
    """
    EM-style training framework that alternates between SBERT fine-tuning and constrained KMeans
    to learn constraint-aware embeddings
    """

    def __init__(self,
                 base_model_name: str = 'all-MiniLM-L6-v2',
                 max_iterations: int = 10,
                 positive_threshold: float = 0.75,
                 negative_threshold: float = 0.5,
                 batch_size: int = 16,
                 epochs_per_iteration: int = 10,
                 warmup_steps: int = 100,
                 learning_rate: float = 2e-5,
                 save_dir: str = 'results',
                 # Example memory parameters
                 max_example_memory: int = 10000,
                 memory_decay_factor: float = 0.0,
                 # Dataset generation parameters
                 max_anchors: int = 10000,
                 use_all_anchors: bool = False,
                 sample_near_centroids: bool = False,
                 centroid_distance_threshold: float = 0.5,
                 centroid_similarity_threshold: float = -1.0,
                 max_comparisons_per_anchor: int = 1000,
                 target_dataset_size: Optional[int] = None,
                 positive_negative_ratio: float = 1.0,
                 # Initialization strategy parameters
                 progressive_init: bool = True,
                 centroid_adjustment_percentage: float = 0.1,
                 # Loss function for SBERT finetuning
                 loss_function: str = 'cosine_similarity',
                 # PCKMeans parameters
                 n_clusters: int = 100,
                 w_cl: float = 0.5,
                 seed: int = 42,
                 # Best model selection criteria
                 best_model_criteria: str = 'constraint_violations'):  # 'constraint_violations' or 'purity'
        """
        Initialize the EM-style training framework

        Args:
            base_model_name: Name of the base SBERT model to use
            max_iterations: Maximum number of EM iterations
            positive_threshold: Similarity threshold for positive examples
            negative_threshold: Similarity threshold for negative examples
            batch_size: Batch size for SBERT fine-tuning
            epochs_per_iteration: Number of epochs per EM iteration
            warmup_steps: Warmup steps for SBERT fine-tuning
            learning_rate: Learning rate for SBERT fine-tuning
            save_dir: Directory to save results
            max_example_memory: Maximum number of examples to keep in memory
            memory_decay_factor: Factor to control decay of old examples (0.0 = no decay, 1.0 = full decay)
            max_anchors: Maximum number of anchor points for similarity dataset generation
            max_comparisons_per_anchor: Maximum number of comparisons per anchor point
            target_dataset_size: Target size for similarity dataset (if None, use natural size)
            positive_negative_ratio: Target ratio of positive to negative examples
            progressive_init: Whether to use progressive initialization strategy
            loss_function: Loss function for SBERT fine-tuning ('cosine_similarity', 'triplet', or 'multiple_negatives')
            centroid_adjustment_percentage: Percentage of closest points to use for centroid adjustment
            n_clusters: Number of clusters for PCKMeans
            w_cl: Weight for cannot-link constraints
        """
        self.base_model_name = base_model_name
        self.max_iterations = max_iterations
        self.positive_threshold = positive_threshold
        self.negative_threshold = negative_threshold
        self.batch_size = batch_size
        self.epochs_per_iteration = epochs_per_iteration
        self.warmup_steps = warmup_steps
        self.learning_rate = learning_rate
        self.save_dir = save_dir

        # Example memory parameters
        self.max_example_memory = max_example_memory
        self.memory_decay_factor = memory_decay_factor

        # Dataset generation parameters
        self.max_anchors = max_anchors
        self.use_all_anchors = use_all_anchors
        self.max_comparisons_per_anchor = max_comparisons_per_anchor
        self.sample_near_centroids = sample_near_centroids
        self.centroid_distance_threshold = centroid_distance_threshold
        self.centroid_similarity_threshold = centroid_similarity_threshold
        self.target_dataset_size = target_dataset_size
        self.positive_negative_ratio = positive_negative_ratio

        # Initialization strategy parameters
        self.progressive_init = progressive_init
        self.centroid_adjustment_percentage = centroid_adjustment_percentage

        # Loss function
        self.loss_function = loss_function

        # PCKMeans parameters
        self.n_clusters = n_clusters
        self.w_cl = w_cl

        # Best model selection criteria
        self.best_model_criteria = best_model_criteria

        # Random seed
        self.seed = seed

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.random.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Initialize SBERT model
        self.sbert_model = SentenceTransformer(base_model_name)
        self.sbert_model.to(self.device)

        # Initialize constraints
        self.sorted_constraints = None
        self.constraint_graph = None

        # Create save directory if it doesn't exist
        os.makedirs(save_dir, exist_ok=True)

        # Initialize example memory
        self.example_memory = {
            'positive_pairs': set(),
            'negative_pairs': set()
        }

        # Initialize metrics tracking
        self.metrics = {
            'iterations': [],
            'constraint_violations': [],
            'inertia': [],
            'positive_examples': [],
            'negative_examples': [],
            'purity_scores': []
        }

    @staticmethod
    def _compute_similarity(embeddings: np.ndarray,
                            idx_i: int,
                            idx_j: int) -> float:
        """
        Compute cosine similarity between two embeddings

        Args:
            embeddings: Embedding matrix
            idx_i: Index of first embedding
            idx_j: Index of second embedding

        Returns:
            Cosine similarity between embeddings
        """
        emb_i = embeddings[idx_i]
        emb_j = embeddings[idx_j]

        # Ensure embeddings are normalized
        norm_i = np.linalg.norm(emb_i)
        norm_j = np.linalg.norm(emb_j)

        if norm_i == 0 or norm_j == 0:
            return 0.0

        return np.dot(emb_i, emb_j) / (norm_i * norm_j)

    def generate_training_examples(self, 
                                    data: Dict, 
                                    model: 'ConstrainedKMeans',
                                    embeddings: Optional[np.ndarray] = None) -> Dict:
            """
            Generate training examples for SBERT fine-tuning based on clustering results
            
            For each anchor example, we generate:
            1. Positive examples: 
            - one from the same cluster, similar but no constraint violations
            - one from a different cluster, similar but no constraint violations
            2. Negative examples:
            - one from the same cluster that violates a constraint
            - one from a different cluster (dissimilar)
            
            Args:
                data: Dictionary containing original embeddings and constraints
                model: Fitted ConstrainedKMeans model
                embeddings: Updated embeddings (if None, use data['embs'])
                
            Returns:
                Dictionary containing positive and negative pairs
            """
            # Use updated embeddings if provided
            if embeddings is None:
                embeddings = data['embs']
            
            labels = model.labels_
            
            # Use constraint structures from the class (set in train method)
            # constraint_points = ConstraintGraphDB(self.constraint_graph_path)
            
            # Store examples
            positive_same_cluster = []
            positive_diff_cluster = []
            negative_same_cluster = []
            negative_diff_cluster = []
            
            # Anchor selection strategy
            if self.sample_near_centroids:
                print("Sampling anchor points near cluster centroids...", flush=True)
                anchor_indices = self._sample_points_near_centroids(
                    embeddings, 
                    model.cluster_centers_, 
                    labels,
                    self.centroid_distance_threshold,
                    self.centroid_similarity_threshold,
                    # No maximum points per cluster - let the threshold determine the count
                    max_points_per_cluster=None
                )
                print(f"Sampled {len(anchor_indices)} anchor points near centroids", flush=True)
            elif self.use_all_anchors:
                print(f"Using all {len(embeddings)} data points as anchors...", flush=True)
                anchor_indices = list(range(len(embeddings)))
            else:
                # Sample a subset of points to use as anchors
                sample_size = min(self.max_anchors, len(embeddings))
                anchor_indices = random.sample(range(len(embeddings)), sample_size)
                print(f"Sampled {len(anchor_indices)} random anchor points...", flush=True)
            
            print("Generating training examples for each anchor point...", flush=True)
            for anchor_idx in tqdm(anchor_indices):
                anchor_cluster = labels[anchor_idx]
                
                # Get all points in the same cluster
                same_cluster_indices = np.where(labels == anchor_cluster)[0]
                same_cluster_indices = [idx for idx in same_cluster_indices if idx != anchor_idx]
                
                # Skip if no points in same cluster
                if len(same_cluster_indices) == 0:
                    continue
                
                # Get all points in different clusters
                diff_cluster_indices = np.where(labels != anchor_cluster)[0]
                
                # Skip if no points in different clusters
                if len(diff_cluster_indices) == 0:
                    continue
                    
                # Simultaneously process #1 and #3: Find positive examples (no constraints) and negative examples (with constraints) 
                # from the same cluster in a single pass
                
                # Get the constraints for the anchor point
                anchor_constraints = self.constraint_graph.get(anchor_idx, set())
                # anchor_constraints = constraint_points.get_set(anchor_idx)
                
                # Divide same-cluster points into those with and without constraints
                constrained_same_cluster = []
                unconstrained_same_cluster = []
                
                for idx in same_cluster_indices:
                    if idx in anchor_constraints:
                        constrained_same_cluster.append(idx)
                    else:
                        unconstrained_same_cluster.append(idx)
                
                # Calculate similarities for all points in same cluster at once
                if unconstrained_same_cluster:
                    # Process positive examples (unconstrained points)
                    sims_unconstrained = np.array([
                        self._compute_similarity(embeddings, anchor_idx, j)
                        for j in unconstrained_same_cluster
                    ])
                    
                    if len(sims_unconstrained) > 0:
                        # Find most similar unconstrained point
                        most_similar_idx = np.argmax(sims_unconstrained)
                        pos_same_idx = unconstrained_same_cluster[most_similar_idx]
                        sim_pos_same = sims_unconstrained[most_similar_idx]
                        
                        # Only add if similarity is above threshold
                        if sim_pos_same > self.positive_threshold:
                            positive_same_cluster.append((int(anchor_idx), int(pos_same_idx)))
                
                if constrained_same_cluster:
                    # Process negative examples (constrained points)
                    sims_constrained = np.array([
                        self._compute_similarity(embeddings, anchor_idx, j)
                        for j in constrained_same_cluster
                    ])
                    
                    if len(sims_constrained) > 0:
                        # Find most similar constrained point
                        most_similar_constrained_idx = np.argmax(sims_constrained)
                        neg_same_idx = constrained_same_cluster[most_similar_constrained_idx]
                        sim_neg_same = sims_constrained[most_similar_constrained_idx]
                        
                        # Only add if similarity is above threshold
                        if sim_neg_same > self.negative_threshold:
                            negative_same_cluster.append((int(anchor_idx), int(neg_same_idx)))
                
                # 2. Find positive example from a different cluster (similar but no constraints)
                # Sample points from different clusters for efficiency
                sample_size_diff = min(self.max_comparisons_per_anchor, len(diff_cluster_indices))
                sampled_diff = random.sample(list(diff_cluster_indices), sample_size_diff)
                
                # Calculate similarities
                sims_diff_cluster = np.array([
                    self._compute_similarity(embeddings, anchor_idx, j)
                    for j in sampled_diff
                ])
                
                # Filter out points with constraint violations
                valid_diff_indices = []
                for i, idx in enumerate(sampled_diff):
                    # if idx in constraint_points.get_set(anchor_idx):
                    if idx in self.constraint_graph.get(anchor_idx, set()):
                        continue  # Skip if there's a constraint
                    valid_diff_indices.append(i)
                
                if valid_diff_indices:
                    # Find most similar valid point from different cluster
                    valid_diff_sims = sims_diff_cluster[valid_diff_indices]
                    most_similar_diff_idx = valid_diff_indices[np.argmax(valid_diff_sims)]
                    pos_diff_idx = sampled_diff[most_similar_diff_idx]
                    sim_pos_diff = sims_diff_cluster[most_similar_diff_idx]
                    
                    # Only add if similarity is above threshold
                    if sim_pos_diff > self.positive_threshold * 0.9:  # Slightly lower threshold
                        positive_diff_cluster.append((int(anchor_idx), int(pos_diff_idx)))
                
                # 4. Find negative example from a different cluster (can be any)
                # Use already sampled points from different clusters
                if sampled_diff:
                    # Find most dissimilar point from different cluster
                    least_similar_diff_idx = np.argmin(sims_diff_cluster)
                    neg_diff_idx = sampled_diff[least_similar_diff_idx]
                    
                    # Add as negative example
                    negative_diff_cluster.append((int(anchor_idx), int(neg_diff_idx)))
            
            # Combine all positive and negative pairs
            positive_pairs = positive_same_cluster + positive_diff_cluster
            negative_pairs = negative_same_cluster + negative_diff_cluster
            
            # Create a mapping from anchor to its positive and negative pairs
            anchor_to_pairs = {}
            for i, j in positive_pairs:
                if i not in anchor_to_pairs:
                    anchor_to_pairs[i] = {'positive': [], 'negative': []}
                anchor_to_pairs[i]['positive'].append((i, j))
            
            for i, j in negative_pairs:
                if i not in anchor_to_pairs:
                    anchor_to_pairs[i] = {'positive': [], 'negative': []}
                anchor_to_pairs[i]['negative'].append((i, j))
            
            # Apply ratio-based balancing either using target size or maintaining ratio
            if self.target_dataset_size is not None:
                # Calculate target sizes for positive and negative based on ratio
                total_samples = min(self.target_dataset_size, len(positive_pairs) + len(negative_pairs))
                target_positive = int(total_samples * self.positive_negative_ratio / (1.0 + self.positive_negative_ratio))
                target_negative = total_samples - target_positive
                
                # Sample positive and negative pairs
                if len(positive_pairs) > target_positive:
                    positive_pairs = random.sample(positive_pairs, target_positive)
                if len(negative_pairs) > target_negative:
                    negative_pairs = random.sample(negative_pairs, target_negative)
            else:
                # No target size, but still maintain the ratio
                actual_ratio = len(positive_pairs) / max(1, len(negative_pairs))
                desired_ratio = self.positive_negative_ratio
                
                if actual_ratio < desired_ratio:
                    # Too few positives compared to negatives - subsample negatives
                    target_negative = int(len(positive_pairs) / desired_ratio)
                    
                    # Try to maintain pairs from same anchors when possible
                    if len(anchor_to_pairs) > 0:
                        # Prioritize anchors that have both positive and negative examples
                        balanced_anchors = [anchor for anchor, pairs in anchor_to_pairs.items() 
                                        if pairs['positive'] and pairs['negative']]
                        
                        # Start with all positive pairs
                        sampled_positive = positive_pairs.copy()
                        sampled_negative = []
                        
                        # First add one negative for each balanced anchor
                        for anchor in balanced_anchors:
                            if len(sampled_negative) < target_negative:
                                neg_pair = random.choice(anchor_to_pairs[anchor]['negative'])
                                sampled_negative.append(neg_pair)
                        
                        # Then randomly sample the rest
                        remaining_negative = target_negative - len(sampled_negative)
                        if remaining_negative > 0 and len(negative_pairs) > len(sampled_negative):
                            remaining_candidates = [p for p in negative_pairs if p not in sampled_negative]
                            sampled_negative.extend(random.sample(remaining_candidates, 
                                                min(remaining_negative, len(remaining_candidates))))
                        
                        negative_pairs = sampled_negative
                    else:
                        # Fallback to random sampling if anchor mapping is empty
                        negative_pairs = random.sample(negative_pairs, min(target_negative, len(negative_pairs)))
                elif actual_ratio > desired_ratio:
                    # Too many positives compared to negatives - subsample positives
                    target_positive = int(len(negative_pairs) * desired_ratio)
                    
                    # Try to maintain pairs from same anchors when possible
                    if len(anchor_to_pairs) > 0:
                        # Prioritize anchors that have both positive and negative examples
                        balanced_anchors = [anchor for anchor, pairs in anchor_to_pairs.items() 
                                        if pairs['positive'] and pairs['negative']]
                        
                        # Start with all negative pairs
                        sampled_negative = negative_pairs.copy()
                        sampled_positive = []
                        
                        # First add one positive for each balanced anchor
                        for anchor in balanced_anchors:
                            if len(sampled_positive) < target_positive:
                                pos_pair = random.choice(anchor_to_pairs[anchor]['positive'])
                                sampled_positive.append(pos_pair)
                        
                        # Then randomly sample the rest
                        remaining_positive = target_positive - len(sampled_positive)
                        if remaining_positive > 0 and len(positive_pairs) > len(sampled_positive):
                            remaining_candidates = [p for p in positive_pairs if p not in sampled_positive]
                            sampled_positive.extend(random.sample(remaining_candidates, 
                                                min(remaining_positive, len(remaining_candidates))))
                        
                        positive_pairs = sampled_positive
                    else:
                        # Fallback to random sampling if anchor mapping is empty
                        positive_pairs = random.sample(positive_pairs, min(target_positive, len(positive_pairs)))

            # constraint_points.close()
            
            print(f"Generated {len(positive_same_cluster)} positive same-cluster pairs")
            print(f"Generated {len(positive_diff_cluster)} positive different-cluster pairs")
            print(f"Generated {len(negative_same_cluster)} negative same-cluster pairs (constraint violations)")
            print(f"Generated {len(negative_diff_cluster)} negative different-cluster pairs")
            print(f"Final dataset: {len(positive_pairs)} positive, {len(negative_pairs)} negative")
            print(f"Actual positive-to-negative ratio: {len(positive_pairs) / max(1, len(negative_pairs)):.2f}")
            
            return {
                'positive_pairs': positive_pairs,
                'negative_pairs': negative_pairs
            }

    def _sample_points_near_centroids(self, 
                            embeddings: np.ndarray, 
                            centroids: np.ndarray, 
                            labels: np.ndarray,
                            distance_threshold: float,
                            similarity_threshold: float = 0.5,  # New parameter
                            max_points_per_cluster: int = None) -> List[int]:
        """
        Sample points that are close to their cluster centroids and have high similarity to them
        
        Args:
            embeddings: Array of embeddings
            centroids: Array of cluster centroids
            labels: Array of cluster assignments
            distance_threshold: Distance threshold (percentile of distances)
            similarity_threshold: Minimum cosine similarity to centroid
            max_points_per_cluster: Maximum number of points per cluster (if None, no limit)
            
        Returns:
            List of indices of points close to centroids with high similarity
        """
        selected_indices = []
        
        # Process each cluster
        for cluster_id in range(len(centroids)):
            # Get indices of points in this cluster
            cluster_indices = np.where(labels == cluster_id)[0]
            
            if len(cluster_indices) == 0:
                continue
                
            # Calculate distances to centroid
            centroid = centroids[cluster_id]
            distances = np.array([
                np.sum((embeddings[idx] - centroid) ** 2) 
                for idx in cluster_indices
            ])
            
            # Determine threshold distance (as percentile of all distances in this cluster)
            if distance_threshold < 1.0:
                # Interpret as percentile if < 1.0
                threshold_distance = np.percentile(distances, distance_threshold * 100)
            else:
                # Use as absolute value if >= 1.0
                threshold_distance = distance_threshold
            
            # Calculate cosine similarities to centroid
            similarities = []
            for idx in cluster_indices:
                point_embedding = embeddings[idx]
                # Compute cosine similarity
                norm_point = np.linalg.norm(point_embedding)
                norm_centroid = np.linalg.norm(centroid)
                
                if norm_point == 0 or norm_centroid == 0:
                    similarity = 0.0
                else:
                    similarity = np.dot(point_embedding, centroid) / (norm_point * norm_centroid)
                    
                similarities.append(similarity)
            similarities = np.array(similarities)
            
            # Select points within distance threshold AND above similarity threshold
            close_indices = [
                cluster_indices[i] for i in range(len(distances)) 
                if distances[i] <= threshold_distance and similarities[i] >= similarity_threshold
            ]
            
            # If max_points_per_cluster is specified, limit the number of points from this cluster
            if max_points_per_cluster is not None and len(close_indices) > max_points_per_cluster:
                close_indices = random.sample(close_indices, max_points_per_cluster)
            
            # Add to selected indices
            selected_indices.extend(close_indices)
        
        print(f"Selected {len(selected_indices)} points near centroids across {len(centroids)} clusters", flush=True)
        print(f"Points satisfy both distance threshold ({distance_threshold}) and similarity threshold ({similarity_threshold})", flush=True)
        
        return selected_indices

    def update_example_memory(self, new_examples: Dict[str, List[Tuple[int, int]]]) -> Dict[str, List[Tuple[int, int]]]:
        """
        Update example memory with new examples

        Args:
            new_examples: Dictionary containing new positive and negative pairs

        Returns:
            Updated example memory
        """
        # Apply memory decay if configured
        if self.memory_decay_factor > 0:
            # Reduce the weight of old examples
            current_positive = list(self.example_memory['positive_pairs'])
            current_negative = list(self.example_memory['negative_pairs'])

            # Keep only a fraction of old examples based on decay factor
            keep_count_positive = int(len(current_positive) * (1.0 - self.memory_decay_factor))
            keep_count_negative = int(len(current_negative) * (1.0 - self.memory_decay_factor))

            # Keep the most recent examples
            old_positive = set(current_positive[-keep_count_positive:]) if keep_count_positive > 0 else set()
            old_negative = set(current_negative[-keep_count_negative:]) if keep_count_negative > 0 else set()
        else:
            # Keep all old examples (up to max_memory/2)
            old_positive = set(list(self.example_memory['positive_pairs'])[-self.max_example_memory // 2:])
            old_negative = set(list(self.example_memory['negative_pairs'])[-self.max_example_memory // 2:])

        # Update with new examples
        self.example_memory['positive_pairs'] = old_positive | set(new_examples['positive_pairs'])
        self.example_memory['negative_pairs'] = old_negative | set(new_examples['negative_pairs'])

        # Ensure we don't exceed max_memory by keeping most recent examples
        if len(self.example_memory['positive_pairs']) > self.max_example_memory // 2:
            self.example_memory['positive_pairs'] = set(
                list(self.example_memory['positive_pairs'])[-self.max_example_memory // 2:]
            )

        if len(self.example_memory['negative_pairs']) > self.max_example_memory // 2:
            self.example_memory['negative_pairs'] = set(
                list(self.example_memory['negative_pairs'])[-self.max_example_memory // 2:]
            )

        # Convert to dictionary of lists
        return {
            'positive_pairs': list(self.example_memory['positive_pairs']),
            'negative_pairs': list(self.example_memory['negative_pairs'])
        }

    @staticmethod
    def adjust_centroids(previous_centroids: np.ndarray,
                         old_embeddings: np.ndarray,
                         new_embeddings: np.ndarray,
                         percentage: float = 0.1) -> np.ndarray:
        """
        Adjust centroids based on change in embedding space

        Args:
            previous_centroids: Previous cluster centroids
            old_embeddings: Previous embeddings
            new_embeddings: Updated embeddings
            percentage: Percentage of closest points to use (default: 10%)

        Returns:
            Adjusted centroids
        """
        adjusted_centroids = []

        # Calculate number of points to use (at least 3, at most 100)
        num_points = max(3, min(100, int(len(old_embeddings) * percentage)))
        print(f"Using {num_points} closest points ({percentage * 100:.1f}% of data) to adjust centroids", flush=True)

        for centroid in previous_centroids:
            # Find closest points to this centroid in old space
            distances = np.sum((old_embeddings - centroid) ** 2, axis=1)
            closest_indices = np.argsort(distances)[:num_points]  # Top n% closest points

            # Find where these points are in the new space and compute new centroid
            new_position = np.mean(new_embeddings[closest_indices], axis=0)
            adjusted_centroids.append(new_position)

        return np.array(adjusted_centroids)

    def hybrid_initialization(self,
                              model: ConstrainedKMeans,
                              new_embeddings: np.ndarray,
                              sorted_constraints: List[Tuple[int, int]] = None,
                              reinit_fraction: float = 0.3) -> np.ndarray:
        """
        Reinitialize worst-performing centroids while keeping others

        Args:
            model: Previous ConstrainedKMeans model
            new_embeddings: Updated embeddings
            constraints: Cannot-link constraints
            reinit_fraction: Fraction of centroids to reinitialize

        Returns:
            Updated centroids
        """
        # Keep track of which clusters have the most constraint violations
        cluster_violations = {}
        for i, j in model.persistent_violations:
            c1, c2 = model.labels_[i], model.labels_[j]
            cluster_violations[c1] = cluster_violations.get(c1, 0) + 1
            cluster_violations[c2] = cluster_violations.get(c2, 0) + 1

        # Sort clusters by violation count
        sorted_clusters = sorted(cluster_violations.items(),
                                 key=lambda x: x[1], reverse=True)

        # Determine which centroids to reinitialize
        n_reinit = max(1, int(reinit_fraction * model.n_clusters))
        clusters_to_reinit = [c for c, _ in sorted_clusters[:n_reinit]]

        # If no violations, reinitialize random centroids
        if not clusters_to_reinit:
            clusters_to_reinit = random.sample(range(model.n_clusters), n_reinit)

        # Create new centroids list
        new_centroids = model.cluster_centers_.copy()

        # Reinitialize selected centroids
        initializer = KMeansPlusPlusInit(
            strategy=InitializationStrategy.CONSTRAINT_AWARE,
            w_cl=model.w_cl
        )
        reinit_centroids = initializer.initialize(
            new_embeddings, n_reinit, sorted_constraints, random_state=42
        )

        for i, cluster_id in enumerate(clusters_to_reinit):
            new_centroids[cluster_id] = reinit_centroids[i]

        return new_centroids

    def finetune_sbert(self,
                       examples: Dict[str, List[Tuple[int, int]]],
                       sentences: List[str]) -> None:
        """
        Fine-tune SBERT using modern SentenceTransformerTrainer approach

        Args:
            examples: Dictionary containing positive and negative pairs
            sentences: Original sentences
        """

        # Prepare the datasets in the format expected by the Trainer
        positive_pairs = examples['positive_pairs']
        negative_pairs = examples['negative_pairs']

        print(f"Creating {len(positive_pairs)} positive and {len(negative_pairs)} negative training examples", flush=True)

        # Create a dataset in HuggingFace datasets format
        train_data = []

        # Add positive pairs
        for i, j in positive_pairs:
            train_data.append({
                'text_1': sentences[i],
                'text_2': sentences[j],
                'label': 1.0
            })

        # Add negative pairs
        for i, j in negative_pairs:
            train_data.append({
                'text_1': sentences[i],
                'text_2': sentences[j],
                'label': 0.0
            })

        # Create the Dataset object
        train_dataset = Dataset.from_list(train_data)

        # For evaluation, hold out 10% of data
        # train_eval = train_dataset.train_test_split(test_size=0.1)
        # train_dataset = train_eval['train']
        # eval_dataset = train_eval['test']

        print(f"Train dataset: {len(train_dataset)} examples", flush=True)
        # print(f"Eval dataset: {len(eval_dataset)} examples", flush=True)

        # Define loss function based on configuration
        if self.loss_function == 'cosine_similarity':
            loss = CosineSimilarityLoss
        elif self.loss_function == 'triplet':
            loss = TripletLoss
        elif self.loss_function == 'multiple_negatives':
            loss = MultipleNegativesRankingLoss
        else:
            # Default to cosine similarity loss
            print(f"Warning: Unknown loss function '{self.loss_function}'. Using CosineSimilarityLoss instead.", flush=True)
            loss = CosineSimilarityLoss

        # Define training arguments
        output_dir = os.path.join(self.save_dir, "sbert_checkpoints")
        os.makedirs(output_dir, exist_ok=True)

        # Configure training arguments
        training_args = SentenceTransformerTrainingArguments(
            output_dir=output_dir,
            num_train_epochs=self.epochs_per_iteration,
            per_device_train_batch_size=self.batch_size,
            per_device_eval_batch_size=self.batch_size,
            warmup_steps=self.warmup_steps,
            weight_decay=0.01,
            logging_dir=os.path.join(output_dir, "logs"),
            logging_steps=100,
            greater_is_better=False,
            learning_rate=self.learning_rate,
            report_to=[],
            save_strategy="no",
            save_total_limit=0,           
            load_best_model_at_end=False
        )

        # Create the trainer
        trainer = SentenceTransformerTrainer(
            model=self.sbert_model,
            args=training_args,
            train_dataset=train_dataset,
            loss=loss,
        )

        # Train the model
        print(f"Fine-tuning SBERT for {self.epochs_per_iteration} epochs...", flush=True)
        trainer.train()

        print("Fine-tuning complete", flush=True)

    def train(self,
              config,
              sentences: List,
              initialization_strategy: str = "",
              kmeans_params: Dict = None,
              existing_metrics: List[Dict] = None,
              centroid_percentile:int=None,
              pairwise_percentile:int=None) -> Dict:
        """
        Run the EM-style training framework

        Args:
            data: Dictionary containing 'embs', 'constraints', and 'sentences'
            n_clusters: Number of clusters
            w_cl: Weight for cannot-link constraints
            kmeans_params: Additional parameters for KMeans
            existing_metrics: Existing metrics to continue tracking

        Returns:
            Dictionary containing final model, SBERT model, and metrics
        """

        print("Centroid Distance Threshold:", self.centroid_distance_threshold, flush=True)
        print("Max Anchors:", self.max_anchors, flush=True)

        # Set default KMeans parameters
        if kmeans_params is None:
            kmeans_params = {
                'max_iter': 100,
                'tol': 1e-4,
                'early_stopping_tol': 5,
                'random_state': 42
            }

        # Initialize embeddings and constraints
        embeddings = self.sbert_model.encode(
            sentences, batch_size=32, show_progress_bar=True, normalize_embeddings=True
        )

        # Initialize metrics
        iteration_metrics = [] if existing_metrics is None else existing_metrics.copy()

        # Initialize models
        previous_pckmeans_model = None
        previous_embeddings = None

        print("Loading constraint graph and sorted constraints...", flush=True)

        self.constraint_graph_path = config["constraints_graph_path"]
        self.sorted_constraints_path = config["constraints_flat_path"]

        self.sorted_constraints = ConstraintFlatDB(self.sorted_constraints_path).get_all_tuples_as_list()
        self.constraint_graph = ConstraintGraphDB(self.constraint_graph_path).get_all_sets_as_dict()

        best_model = {
            'model': ConstrainedKMeans,
            'embs': ndarray,
            'inertia': float('inf'),
            'constraint_violations': float('inf'),
            'purity': 0.0
        }

        # Main EM loop
        for iteration in range(self.max_iterations):
            print(f"\n=== EM Iteration {iteration + 1}/{self.max_iterations} ===\n", flush=True)

            # Determine initialization strategy
            if initialization_strategy == "":
                if self.progressive_init:
                    if iteration < 2:
                        # Complete reinitialization
                        initialization_strategy = "from_scratch"
                    elif iteration < self.max_iterations - 2:
                        # Hybrid approach
                        initialization_strategy = "hybrid"
                    else:
                        # Warm start with adjustments
                        initialization_strategy = "warm_start"
                else:
                    # Always use from_scratch strategy
                    initialization_strategy = "from_scratch"

            print(f"Using initialization strategy: {initialization_strategy}",flush=True)


            # Apply the appropriate initialization strategy
            init_centroids = None

            if initialization_strategy == "scikit_kmeans":
                # Use scikit-learn KMeans for initialization
                print("Using scikit-learn KMeans for initialization...", flush=True)
                sk_kmeans = KMeans(n_clusters=self.n_clusters, random_state=config["seed"], init='k-means++', n_init=5)
                sk_kmeans.fit(embeddings)
                init_centroids = sk_kmeans.cluster_centers_
            elif initialization_strategy == "from_scratch" or iteration == 0:
                # Standard initialization from scratch
                print("Using standard initialization from scratch...", flush=True)
                init_centroids = None  # Let the model handle initialization
            elif initialization_strategy == "hybrid" and previous_pckmeans_model is not None:
                # Hybrid reinitialization
                print("Using hybrid initialization with previous model...", flush=True)
                init_centroids = self.hybrid_initialization(
                    previous_pckmeans_model, embeddings, self.sorted_constraints_path, reinit_fraction=0.3
                )
            elif previous_pckmeans_model is not None and previous_embeddings is not None:
                # Warm start with adjustments
                print("Using warm start with adjusted centroids...", flush=True)
                init_centroids = self.adjust_centroids(
                    previous_pckmeans_model.cluster_centers_, previous_embeddings, embeddings,
                    percentage=self.centroid_adjustment_percentage
                )

            # Create initializer
            initializer = KMeansPlusPlusInit(
                strategy=InitializationStrategy.CONSTRAINT_AWARE,
                w_cl=self.w_cl
            )

            # Create KMeans model
            pckmeans_model = ConstrainedKMeans(
                n_clusters=self.n_clusters,
                initializer=initializer,
                w_cl=self.w_cl,
                centroid_percentile=centroid_percentile,
                pairwise_percentile=pairwise_percentile,
                # constraint_graph=self.constraint_graph,
                # sorted_constraints=self.sorted_constraints,
                **kmeans_params
            )

            # Run clustering
            print("Running constrained KMeans clustering...", flush=True)

            # Set initial centroids if we have them
            if init_centroids is not None:
                print("Using custom initialization for centroids...", flush=True)
                pckmeans_model.cluster_centers_ = init_centroids
                # Fit with custom initialization
                pckmeans_model.fit(embeddings,
                                    sorted_constraints=self.sorted_constraints,
                                    constraint_graph=self.constraint_graph,
                                    skip_init=True)
            else:
                # Regular fit
                pckmeans_model.fit(embeddings,
                                    sorted_constraints=self.sorted_constraints,
                                    constraint_graph=self.constraint_graph,
                                    skip_init=False)

            # Get clustering metrics
            final_metrics = pckmeans_model.history_[-1]
            print(f"Inertia: {final_metrics.inertia:.2f}, "
                  f"Constraint violations: {final_metrics.constraint_violations}", flush=True)
            
            # Calculate purity if using purity-based best model selection
            current_purity = 0.0
            if self.best_model_criteria == 'purity':
                print("Calculating exact match purity score...", flush=True)
                clustering_data = {
                    "number_cluster": self.n_clusters,
                    "labels": pckmeans_model.labels_,
                    "embeddings": embeddings,
                    "cluster_centers": pckmeans_model.cluster_centers_
                }
                purity_calculator = Purity(config["processed_chains_path"], clustering_data)
                purity_calculator.compute_purity()
                
                # Get exact match purity (average of 25% and 100%)
                purity_25 = purity_calculator.results['25']['exact_match_purity']['score']
                purity_100 = purity_calculator.results['100']['exact_match_purity']['score']
                current_purity = (purity_25 + purity_100) / 2.0 * 100  # Convert to percentage
                
                print(f"Current exact match purity: {current_purity:.2f}%", flush=True)
            

            # Check if best model based on configured criteria
            is_best = False
            if self.best_model_criteria == 'purity':
                # Best model based on highest purity
                if current_purity > best_model['purity']:
                    is_best = True
            else:
                # Best model based on lowest constraint violations (original criteria)
                if final_metrics.constraint_violations < best_model['constraint_violations']:
                    is_best = True
            
            if is_best:
                best_model['model'] = pckmeans_model
                best_model['embs'] = embeddings
                best_model['inertia'] = final_metrics.inertia
                best_model['constraint_violations'] = final_metrics.constraint_violations
                best_model['purity'] = current_purity
                print(f"Found new best model! ({self.best_model_criteria}: {current_purity:.2f}% purity, {final_metrics.constraint_violations} violations)", flush=True)

            # Check if last iteration
            if iteration == self.max_iterations - 1:
                print("Reached maximum iterations. Stopping training.", flush=True)
                break

            # self.constraint_graph = pckmeans_model.constraint_graph
            # self.sorted_constraints = pckmeans_model.sorted_constraints

            # Generate training examples
            print("Generating training examples...", flush=True)
            new_examples = self.generate_training_examples(
                data, pckmeans_model, embeddings
            )

            print(f"Generated {len(new_examples['positive_pairs'])} positive pairs and "
                  f"{len(new_examples['negative_pairs'])} negative pairs", flush=True)

            # # Update example memory
            # examples = self.update_example_memory(new_examples)
            # print(f"Example memory has {len(examples['positive_pairs'])} positive pairs and "
            #       f"{len(examples['negative_pairs'])} negative pairs", flush=True)

            examples = new_examples

            # Check if we have enough examples
            if len(examples['positive_pairs']) < 10 or len(examples['negative_pairs']) < 10:
                print("Warning: Not enough training examples generated. Trying with looser thresholds.", flush=True)
                # Try with looser thresholds
                temp_pos_threshold = self.positive_threshold * 0.9
                temp_neg_threshold = self.negative_threshold * 0.9

                # Update thresholds temporarily
                self.positive_threshold = temp_pos_threshold
                self.negative_threshold = temp_neg_threshold

                # Try again
                new_examples = self.generate_training_examples(
                    data, pckmeans_model, embeddings
                )
                examples = self.update_example_memory(new_examples)

                # Restore original thresholds
                self.positive_threshold = self.positive_threshold / 0.9
                self.negative_threshold = self.negative_threshold / 0.9

                print(f"With looser thresholds, generated {len(examples['positive_pairs'])} positive pairs and "
                      f"{len(examples['negative_pairs'])} negative pairs", flush=True)

                # If still not enough, consider stopping
                if len(examples['positive_pairs']) < 10 or len(examples['negative_pairs']) < 10:
                    print("Warning: Still not enough examples. Continuing with what we have.",flush=True)
                    if iteration > 0:
                        break



            print("Fine-tuning SBERT...",flush=True)
            self.finetune_sbert(examples, sentences)

            # Update embeddings
            print("Updating embeddings...", flush=True)
            previous_embeddings = embeddings.copy()
            embeddings = self.sbert_model.encode(
                sentences, batch_size=32, show_progress_bar=True, normalize_embeddings=True
            )

            # Save iteration metrics
            iteration_metrics.append({
                'iteration': iteration + 1,
                'inertia': final_metrics.inertia,
                'constraint_violations': final_metrics.constraint_violations,
                'positive_examples': len(examples['positive_pairs']),
                'negative_examples': len(examples['negative_pairs']),
                'purity': current_purity
            })

            # Update previous model
            previous_pckmeans_model = pckmeans_model

        # self.sorted_constraints.close()
        # self.constraint_graph.close()

        # Compute and print final purity for the best model
        print("\n=== Final Purity Results for Best Model ===", flush=True)
        final_clustering_data = {
            "number_cluster": self.n_clusters,
            "labels": best_model['model'].labels_,
            "embeddings": best_model['embs'],
            "cluster_centers": best_model['model'].cluster_centers_
        }
        final_purity_calculator = Purity(config["processed_chains_path"], final_clustering_data)
        final_purity_calculator.compute_purity()
        final_purity_calculator.print_results()
        print("==========================================\n", flush=True)

        return best_model

    def save(self, config, pckmeans_model, embs, init_strategy):
        """Save the model to a file."""

        print("Saving clustering results...", flush=True)

        output = {
            "number_cluster": self.n_clusters,
            "w_cl": self.w_cl,
            "centroid_percentile":pckmeans_model.centroid_percentile,
            "pairwise_percentile":pckmeans_model.pairwise_percentile,
            "cluster_centers": pckmeans_model.cluster_centers_,
            "labels": pckmeans_model.labels_,
            "embeddings": embs,
            "violations": pckmeans_model.get_violation_statistics()
        }

        with open(config["clusters_path"] + f"em_clusters_{self.n_clusters}_{self.w_cl}_{init_strategy}_"
                                            f"{self.centroid_distance_threshold}_{self.max_anchors}.pickle",
                  'wb') as f:
            pickle.dump(output, f, protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PCKmeans Clustering')
    parser.add_argument('-c', metavar='CONF', default='base', help='configuration (see config.conf)')
    parser.add_argument('-k', metavar='N_CLUSTERS', default=5, type=int, help='number of clusters')
    parser.add_argument('-i', metavar='MAX_ITER', default=5, type=int, help='maximum number of iterations')
    parser.add_argument('-w', metavar='W_CL', default=1.0, type=float, help='weight for cannot-link constraints')
    parser.add_argument('--centroid_percentile', metavar='CENTROID_PERCENTILE', default=None, type=float, help='percentile threshold for distance to cluster centers (e.g., 25 for top 25%)')
    parser.add_argument('--pairwise_percentile', metavar='PAIRWISE_PERCENTILE', default=None, type=float, help='percentile threshold for pairwise distances (e.g., 10 for bottom 10%)')
    parser.add_argument('--init_strategy', metavar='INIT_STRATEGY', default='from_scratch', type=str,
                        help='initialization strategy (e.g., "scikit_kmeans", "from_scratch", "hybrid", "warm_start")')
    parser.add_argument('--best_model_criteria', metavar='BEST_MODEL_CRITERIA', default='purity', type=str,
                        help='criteria for selecting best model: "constraint_violations" or "purity"')


    args = parser.parse_args()
    config = ConfigFactory.parse_file('./config.conf')[args.c]

    print("N_CLUSTERS: " + str(args.k), flush=True)
    print("W_CL: " + str(args.w), flush=True)
    print("CENTROID_PERCENTILE: " + str(args.centroid_percentile), flush=True)
    print("PAIRWISE_PERCENTILE: " + str(args.pairwise_percentile), flush=True)

    print("Loading data for clustering...", flush=True)

    with open(config["processed_chains_path"], 'rb') as f:
        data = pickle.load(f)

    framework = SBERTConstrainedClusteringTrainer(n_clusters=args.k,
                                                  w_cl=args.w,
                                                  use_all_anchors=True,
                                                  centroid_distance_threshold=1.0,
                                                  sample_near_centroids=True,
                                                  progressive_init=False,
                                                  memory_decay_factor=1.0,
                                                  save_dir=config["clusters_path"]+"/finetuned_pckmeans",
                                                  best_model_criteria=args.best_model_criteria)

    best_model = framework.train(config,
                                 data["chain_sents"], 
                                 initialization_strategy=args.init_strategy)

    framework.save(config, best_model['model'], best_model['embs'], args.init_strategy)