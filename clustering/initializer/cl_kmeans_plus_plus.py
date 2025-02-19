from enum import Enum
from typing import List, Tuple, Dict, Optional, Set

import numpy as np
import numpy.typing as npt

from clustering.initializer.base_initializer import BaseInitializer


class InitializationStrategy(Enum):
    """Enumeration of available initialization strategies"""
    STANDARD = "standard"  # Standard K-means++ without constraint awareness
    CONSTRAINT_AWARE = "constraint_aware"  # K-means++ with constraint awareness

class KMeansPlusPlusInit(BaseInitializer):
    """Configurable KMeans++ initialization with optional constraint awareness"""

    def __init__(self,
                 strategy: InitializationStrategy = InitializationStrategy.STANDARD,
                 w_cl: float = 1.0):
        """
        Initialize the KMeans++ initializer

        Args:
            strategy: Initialization strategy to use
            w_cl: Weight for constraint satisfaction (only used if strategy is CONSTRAINT_AWARE)
        """
        self.strategy = strategy
        self.w_cl = w_cl

    def _build_constraint_graph(self,
                                n_samples: int,
                                cl_constraints: List[Tuple[int, int]]) -> Dict[int, Set[int]]:
        """Build adjacency graph of constraints for efficient lookup"""
        constraint_graph = {i: set() for i in range(n_samples)}
        for i, j in cl_constraints:
            constraint_graph[i].add(j)
            constraint_graph[j].add(i)
        return constraint_graph

    def _compute_scores(self,
                        X: npt.NDArray[np.float64],
                        centers: npt.NDArray[np.float64],
                        center_indices: List[int],
                        constraint_graph: Optional[Dict[int, Set[int]]] = None) -> npt.NDArray[np.float64]:
        """
        Compute scores for each point based on the chosen strategy

        Args:
            X: Input data
            centers: Currently selected centers
            center_indices: Indices of selected centers
            constraint_graph: Constraint adjacency graph (only used for CONSTRAINT_AWARE)

        Returns:
            Array of scores for each point
        """
        n_samples = X.shape[0]

        # Calculate minimum distances to existing centers
        if len(centers) > 0:
            min_distances = np.min([
                np.sum((X - center) ** 2, axis=1)
                for center in centers
            ], axis=0)
        else:
            min_distances = np.ones(n_samples)

        if self.strategy == InitializationStrategy.STANDARD:
            return min_distances

        elif self.strategy == InitializationStrategy.CONSTRAINT_AWARE:
            # Add constraint satisfaction bonus
            constraint_bonus = np.zeros(n_samples)
            if constraint_graph is not None:
                for i in range(n_samples):
                    if i not in center_indices:
                        satisfied_constraints = sum(
                            1 for idx in center_indices
                            if i in constraint_graph[idx]
                        )
                        constraint_bonus[i] = self.w_cl * satisfied_constraints
            return min_distances + constraint_bonus

    def initialize(self,
                   X: npt.NDArray[np.float64],
                   n_clusters: int,
                   cl_constraints: List[Tuple[int, int]],
                   random_state: Optional[int] = None) -> npt.NDArray[np.float64]:
        """
        Initialize cluster centers using configurable KMeans++

        Args:
            X: Input data of shape (n_samples, n_features)
            n_clusters: Number of clusters to initialize
            cl_constraints: List of cannot-link constraints
            random_state: Random seed for reproducibility

        Returns:
            Initial cluster centers of shape (n_clusters, n_features)
        """
        if random_state is not None:
            np.random.seed(random_state)

        n_samples, n_features = X.shape
        centers = np.zeros((n_clusters, n_features))
        center_indices = []

        # Build constraint graph if needed
        constraint_graph = None
        if self.strategy == InitializationStrategy.CONSTRAINT_AWARE:
            constraint_graph = self._build_constraint_graph(n_samples, cl_constraints)

        # Select first center randomly
        first_idx = np.random.randint(n_samples)
        centers[0] = X[first_idx]
        center_indices.append(first_idx)

        # Select remaining centers
        for k in range(1, n_clusters):
            # Compute scores based on strategy
            scores = self._compute_scores(
                X, centers[:k], center_indices, constraint_graph
            )

            # Zero out already selected points
            scores[center_indices] = 0

            # Select next center
            if np.sum(scores) == 0:
                # If all scores are 0, choose randomly from remaining points
                remaining = list(set(range(n_samples)) - set(center_indices))
                next_idx = np.random.choice(remaining)
            else:
                # Choose proportional to scores
                probs = scores / np.sum(scores)
                next_idx = np.random.choice(n_samples, p=probs)

            centers[k] = X[next_idx]
            center_indices.append(next_idx)

        return centers