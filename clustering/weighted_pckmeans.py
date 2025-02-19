import argparse
import pickle
from dataclasses import dataclass
from typing import Optional, List, Tuple, Dict, Set

import numpy as np
import numpy.typing as npt
from pyhocon import ConfigFactory

from clustering.initializer.base_initializer import BaseInitializer
from clustering.initializer.cl_kmeans_plus_plus import KMeansPlusPlusInit, InitializationStrategy


@dataclass
class ClusteringMetrics:
    """Metrics for tracking clustering performance"""
    inertia: float  # Sum of squared distances to centers
    constraint_violations: int  # Number of constraint violations
    total_cost: float  # Combined objective function value
    violated_constraints: List[Tuple[int, int]]  # List of violated constraints

class ConstrainedKMeans:
    """
    Implementation of KMeans clustering with Cannot-Link constraints
    with enhanced violation tracking
    """

    def __init__(self,
                 n_clusters: int,
                 initializer: BaseInitializer,
                 w_cl: float = 1.0,
                 max_iter: int = 100,
                 tol: float = 1e-4,
                 early_stopping_tol: int = 10,
                 random_state: Optional[int] = None):
        """
        Initialize the Constrained KMeans algorithm with violation tracking

        Args:
            n_clusters: Number of clusters
            initializer: Cluster center initializer
            w_cl: Weight for cannot-link constraints
            max_iter: Maximum number of iterations
            tol: Convergence tolerance for centroid movement
            early_stopping_tol: Number of iterations with no improvement before early stopping
            random_state: Random seed
        """
        self.n_clusters = n_clusters
        self.initializer = initializer
        self.w_cl = w_cl
        self.max_iter = max_iter
        self.tol = tol
        self.early_stopping_tol = early_stopping_tol
        self.random_state = random_state

        # Attributes that will be set during fitting
        self.cluster_centers_: Optional[npt.NDArray[np.float64]] = None
        self.labels_: Optional[npt.NDArray[np.int64]] = None
        self.n_iter_: int = 0
        self.history_: List[ClusteringMetrics] = []

        # Enhanced tracking attributes
        self.violation_counts: Dict[Tuple[int, int], int] = {}
        self.violations_per_iteration: List[int] = []
        self.persistent_violations: List[Tuple[int, int]] = []
        self.all_violations_history: List[List[Tuple[int, int]]] = []

    def _compute_distances(self, X: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Compute distances between points and cluster centers"""
        return np.array([
            np.sum((X - center) ** 2, axis=1)
            for center in self.cluster_centers_
        ])

    def _build_constraint_graph(self,
                              n_samples: int,
                              cl_constraints: List[Tuple[int, int]]) -> Dict[int, Set[int]]:
        """Build constraint graph for efficient lookup"""
        constraint_graph = {i: set() for i in range(n_samples)}

        # Sort constraint pairs to ensure consistency
        sorted_constraints = []
        for i, j in cl_constraints:
            if i > j:
                sorted_constraints.append((j, i))
            else:
                sorted_constraints.append((i, j))

        for i, j in sorted_constraints:
            constraint_graph[i].add(j)
            constraint_graph[j].add(i)

        return constraint_graph, sorted_constraints

    def _find_violations(self,
                       assignments: npt.NDArray[np.int64],
                       sorted_constraints: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        """Find all violated constraints"""
        violations = []

        for i, j in sorted_constraints:
            if assignments[i] == assignments[j]:
                violations.append((i, j))

        return violations

    def _count_violations(self,
                         assignments: npt.NDArray[np.int64],
                         constraint_graph: Dict[int, Set[int]],
                         sorted_constraints: List[Tuple[int, int]]) -> Tuple[int, List[Tuple[int, int]]]:
        """Count number of constraint violations and return list of violated constraints"""
        violations = self._find_violations(assignments, sorted_constraints)
        return len(violations), violations

    def _compute_metrics(self,
                        X: npt.NDArray[np.float64],
                        assignments: npt.NDArray[np.int64],
                        constraint_graph: Dict[int, Set[int]],
                        sorted_constraints: List[Tuple[int, int]]) -> ClusteringMetrics:
        """Compute clustering metrics"""
        # Calculate inertia
        distances = self._compute_distances(X)
        min_distances = np.min(distances, axis=0)
        inertia = np.sum(min_distances)

        # Count constraint violations and get list of violated constraints
        num_violations, violated_constraints = self._count_violations(
            assignments, constraint_graph, sorted_constraints
        )

        # Calculate total cost
        total_cost = inertia + self.w_cl * num_violations

        return ClusteringMetrics(inertia, num_violations, total_cost, violated_constraints)

    def _assign_points(self,
                      X: npt.NDArray[np.float64],
                      constraint_graph: Dict[int, Set[int]]) -> npt.NDArray[np.int64]:
        """Assign points to clusters considering cannot-link constraints"""
        n_samples = X.shape[0]
        assignments = np.zeros(n_samples, dtype=np.int64)

        # Calculate base distances to all centers
        distances = self._compute_distances(X)

        # Assign points one by one
        for i in range(n_samples):
            cluster_costs = distances[:, i].copy()

            # Add penalty for constraint violations
            for j in constraint_graph[i]:
                if j < i:  # Only consider already assigned points
                    cluster_costs[assignments[j]] += self.w_cl

            assignments[i] = np.argmin(cluster_costs)

        return assignments

    def _update_centers(self,
                       X: npt.NDArray[np.float64],
                       assignments: npt.NDArray[np.int64]) -> npt.NDArray[np.float64]:
        """Update cluster centers"""
        new_centers = np.zeros_like(self.cluster_centers_)

        for k in range(self.n_clusters):
            mask = (assignments == k)
            if np.any(mask):
                new_centers[k] = np.mean(X[mask], axis=0)
            else:
                # If cluster is empty, keep old center
                new_centers[k] = self.cluster_centers_[k]

        return new_centers

    def _update_violation_statistics(self, violated_constraints: List[Tuple[int, int]]):
        """Update violation tracking statistics"""
        # Update count of total violations per iteration
        self.violations_per_iteration.append(len(violated_constraints))

        # Update all violations history
        self.all_violations_history.append(violated_constraints)

        # Update violation counts for each constraint
        for constraint in violated_constraints:
            if constraint in self.violation_counts:
                self.violation_counts[constraint] += 1
            else:
                self.violation_counts[constraint] = 1

    def fit(self,
            X: npt.NDArray[np.float64],
            cl_constraints: List[Tuple[int, int]]) -> 'ConstrainedKMeans':
        """
        Fit the Constrained KMeans clustering model

        Args:
            X: Training data
            cl_constraints: Cannot-link constraints

        Returns:
            self: Fitted model
        """
        # Initialize tracking structures
        self.violation_counts = {}
        self.violations_per_iteration = []
        self.all_violations_history = []

        # Initialize centers
        self.cluster_centers_ = self.initializer.initialize(
            X, self.n_clusters, cl_constraints, self.random_state
        )

        # Build constraint graph and get sorted constraints
        constraint_graph, sorted_constraints = self._build_constraint_graph(len(X), cl_constraints)

        # Initialize history and tracking variables
        self.history_ = []
        best_cost = float('inf')
        best_centers = None
        best_labels = None
        iterations_without_improvement = 0

        # Main clustering loop
        for iteration in range(self.max_iter):
            # Assign points to clusters
            new_assignments = self._assign_points(X, constraint_graph)

            # Update centers
            new_centers = self._update_centers(X, new_assignments)

            # Compute metrics
            metrics = self._compute_metrics(X, new_assignments, constraint_graph, sorted_constraints)
            self.history_.append(metrics)

            # Update violation statistics
            self._update_violation_statistics(metrics.violated_constraints)

            # Check for improvement
            if metrics.total_cost < best_cost:
                best_cost = metrics.total_cost
                best_centers = new_centers.copy()
                best_labels = new_assignments.copy()
                iterations_without_improvement = 0
            else:
                iterations_without_improvement += 1

            # Early stopping checks
            if iterations_without_improvement >= self.early_stopping_tol:
                break

            # Check for convergence
            if iteration > 0:
                center_shift = np.sum((new_centers - self.cluster_centers_) ** 2)
                if center_shift < self.tol:
                    break

            self.cluster_centers_ = new_centers
            self.labels_ = new_assignments

        # Set best found solution
        if best_centers is not None:
            self.cluster_centers_ = best_centers
            self.labels_ = best_labels

        self.n_iter_ = iteration + 1

        # Identify persistent violations (violated in all iterations)
        if self.all_violations_history:
            min_iterations = min(len(self.all_violations_history), self.n_iter_)
            if min_iterations > 0:
                # Count how many iterations each constraint was violated
                persistent_candidates = [constraint for constraint, count
                                      in self.violation_counts.items()
                                      if count >= min_iterations]

                # Verify they were violated in all iterations up to convergence
                self.persistent_violations = []
                for constraint in persistent_candidates:
                    all_iterations = True
                    for i in range(min_iterations):
                        if constraint not in self.all_violations_history[i]:
                            all_iterations = False
                            break
                    if all_iterations:
                        self.persistent_violations.append(constraint)

        return self

    def predict(self, X: npt.NDArray[np.float64]) -> npt.NDArray[np.int64]:
        """Predict cluster labels for new data"""
        if self.cluster_centers_ is None:
            raise ValueError("Model must be fitted before making predictions")

        distances = self._compute_distances(X)
        return np.argmin(distances, axis=0)

    def get_violation_statistics(self) -> Dict:
        """Get comprehensive violation statistics"""
        if not hasattr(self, 'violation_counts') or not self.violation_counts:
            return {"error": "Model has not been fitted yet"}

        # Calculate frequency of violations
        total_iterations = len(self.violations_per_iteration)
        violation_frequency = {constraint: count / total_iterations
                             for constraint, count in self.violation_counts.items()}

        # Find most frequently violated constraints
        most_violated = sorted(self.violation_counts.items(),
                              key=lambda x: x[1], reverse=True)[:10]

        # Analyze violation trends
        violation_trend = "decreasing" if self.violations_per_iteration[-1] < self.violations_per_iteration[0] else "increasing"
        if len(self.violations_per_iteration) > 2:
            # Check if violations are consistently decreasing
            is_decreasing = all(self.violations_per_iteration[i] >= self.violations_per_iteration[i+1]
                               for i in range(len(self.violations_per_iteration)-1))
            if is_decreasing:
                violation_trend = "consistently decreasing"

            # Check if violations are consistently increasing
            is_increasing = all(self.violations_per_iteration[i] <= self.violations_per_iteration[i+1]
                              for i in range(len(self.violations_per_iteration)-1))
            if is_increasing:
                violation_trend = "consistently increasing"

        return {
            "total_unique_violations": len(self.violation_counts),
            "violations_per_iteration": self.violations_per_iteration,
            "persistent_violations": self.persistent_violations,
            "most_violated_constraints": most_violated,
            "violation_trend": violation_trend,
            "violation_frequency": violation_frequency
        }

    def save(self, config):
        """Save the model to a file."""

        output = {
            "cluster_centers": self.cluster_centers_,
            "labels": self.labels_,
            "violations": self.get_violation_statistics()
        }

        with open(config["clusters_path"], 'wb') as f:
            pickle.dump(output, f, protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PCKmeans Clustering')
    parser.add_argument('-c', metavar='CONF', default='base', help='configuration (see config.conf)')
    parser.add_argument('-k', metavar='N_CLUSTERS', default=5, type=int, help='number of clusters')
    parser.add_argument('-i', metavar='MAX_ITER', default=100, type=int, help='maximum number of iterations')

    args = parser.parse_args()
    config = ConfigFactory.parse_file('./config.conf')[args.c]

    print("Loading data for clustering...", flush=True)

    # Load data
    with open(config["cluster_embs_path"], 'rb') as f:
        data = pickle.load(f)

    print("Clustering...", flush=True)

    # Create initializer (either standard or constraint-aware)
    initializer = KMeansPlusPlusInit(
        strategy=InitializationStrategy.CONSTRAINT_AWARE,
        w_cl=1.0
    )

    # Create and fit the model
    model = ConstrainedKMeans(
        n_clusters=args.k,
        initializer=initializer,
        w_cl=1.0,
        max_iter=args.i,
        tol=1e-4,
        early_stopping_tol=10,
        random_state=config['seed']
    )

    # Fit the model
    model.fit(data['embs'], data['constraints'])

    model.save(config)