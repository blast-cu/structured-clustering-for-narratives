import argparse
import pickle
from dataclasses import dataclass
from typing import Optional, List, Tuple, Dict, Set

import numpy as np
import numpy.typing as npt
from pyhocon import ConfigFactory
from sklearn.cluster import KMeans
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from clustering.initializer.base_initializer import BaseInitializer
from clustering.initializer.cl_kmeans_plus_plus import KMeansPlusPlusInit, InitializationStrategy
from clustering.metrics.purity import Purity
from models.regression import RegressionModel
from utils.constraint_flat_db import ConstraintFlatDB
from utils.constraints_graph_db import ConstraintGraphDB


@dataclass
class ClusteringMetrics:
    """Metrics for tracking clustering performance"""

    inertia: float  # Sum of squared distances to centers
    constraint_violations: int  # Number of constraint violations
    total_cost: float  # Combined objective function value
    violated_constraints: List[Tuple[int, int]]  # List of violated constraints

class ConstrainedKMeans:
    """
    Implementation of KMeans clustering with Pairwise Cannot-Link constraints
    """

    def __init__(self,
                 n_clusters: int,
                 initializer: BaseInitializer = None,
                 w_cl: float = 1.0,
                 max_iter: int = 100,
                 tol: float = 1e-4,
                 early_stopping_tol: int = 10,
                 random_state: Optional[int] = None,
                 centroid_percentile: Optional[float] = None,
                 pairwise_percentile: Optional[float] = None):
        """
        Initialize the Pairwise Constrained KMeans algorithm

        Args:
            n_clusters: Number of clusters
            initializer: Cluster center initializer
            w_cl: Weight for cannot-link constraints
            max_iter: Maximum number of iterations
            tol: Convergence tolerance for centroid movement
            early_stopping_tol: Number of iterations with no improvement before early stopping
            random_state: Random seed
            centroid_percentile: Percentile threshold for distance to cluster centers (e.g., 25 for top 25%)
            pairwise_percentile: Percentile threshold for pairwise distances (e.g., 10 for bottom 10%)
        """

        self.n_clusters = n_clusters
        self.initializer = initializer
        self.w_cl = w_cl
        self.max_iter = max_iter
        self.tol = tol
        self.early_stopping_tol = early_stopping_tol
        self.random_state = random_state
        self.centroid_percentile = centroid_percentile
        self.pairwise_percentile = pairwise_percentile

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
    
    def _compute_dual_thresholds(self, X: npt.NDArray[np.float64], assignments: npt.NDArray[np.int64]) -> Tuple[float, float]:
        """Compute global centroid and pairwise distance thresholds using efficient numpy operations"""
        
        # Compute global centroid threshold
        centroid_threshold = None
        if self.centroid_percentile is not None:
            # Compute distance from each point to its assigned cluster center
            assigned_centers = self.cluster_centers_[assignments]  # (n_samples, n_features)
            distances_to_assigned_centers = np.linalg.norm(X - assigned_centers, axis=1)
            centroid_threshold = np.percentile(distances_to_assigned_centers, self.centroid_percentile)
        
        # Compute global pairwise threshold
        pairwise_threshold = None
        if self.pairwise_percentile is not None:
            # Use vectorized pairwise distance computation on a sample
            n_samples = min(1000, len(X))  # Limit for performance
            sample_indices = np.random.choice(len(X), n_samples, replace=False)
            sample_X = X[sample_indices]
            
            # Vectorized pairwise distance computation
            # Broadcasting: (n, 1, d) - (1, n, d) -> (n, n, d)
            diff = sample_X[:, np.newaxis, :] - sample_X[np.newaxis, :, :]
            # Compute squared distances: (n, n)
            squared_distances = np.sum(diff ** 2, axis=2)
            # Take square root and get upper triangular part (avoid duplicates)
            distances = np.sqrt(squared_distances)
            upper_triangle = np.triu_indices(n_samples, k=1)
            pairwise_distances = distances[upper_triangle]
            
            pairwise_threshold = np.percentile(pairwise_distances, self.pairwise_percentile)
        
        return centroid_threshold, pairwise_threshold

    def _get_constraint_eligible_items(self, X: npt.NDArray[np.float64], assignments: npt.NDArray[np.int64], centroid_threshold: Optional[float]) -> Set[int]:
        """Get items eligible for constraint enforcement based on centroid threshold."""
        
        # Filter items based on centroid threshold
        if centroid_threshold is not None:
            # Get distance from each point to its assigned cluster center
            assigned_centers = self.cluster_centers_[assignments]
            distances_to_assigned_centers = np.linalg.norm(X - assigned_centers, axis=1)
            
            # Items closer than threshold are eligible
            eligible_mask = distances_to_assigned_centers <= centroid_threshold
            return set(np.where(eligible_mask)[0])
        else:
            # If no centroid threshold, all items are eligible
            return set(range(len(X)))

    @staticmethod
    def _build_constraint_graph(n_samples: int,
                            cl_constraints: List[Tuple[int, int]]) -> Dict[int, Set[int]]:
        """Build constraint graph for efficient lookup"""

        constraint_graph = {i: set() for i in range(n_samples)}
        
        # Pre-sort constraints once to avoid repeated sorting
        sorted_constraints = []
        
        for i, j in tqdm(cl_constraints):
            # Ensure i < j ordering consistently
            if i > j:
                sorted_constraints.append((int(j), int(i)))
            else:
                sorted_constraints.append((int(i), int(j)))
            
            # Add to constraint graph
            constraint_graph[int(i)].add(int(j))
            constraint_graph[int(j)].add(int(i))

        return constraint_graph, sorted_constraints

    def _find_violations(self, X: npt.NDArray[np.float64], assignments: npt.NDArray[np.int64],
                       eligible_items: Set[int], pairwise_threshold: Optional[float]) -> List[Tuple[int, int]]:
        """Find all violated constraints, filtered by dual threshold eligibility"""

        violations = []

        # for i, j in self.sorted_constraints.read_all_tuples():
        for i, j in self.sorted_constraints:
            if assignments[i] == assignments[j]:
                # Check dual threshold conditions
                constraint_applies = True
                
                # Check centroid threshold (via eligible items)
                constraint_applies = constraint_applies and (i in eligible_items and j in eligible_items)
                
                # Check pairwise threshold
                if constraint_applies and pairwise_threshold is not None:
                    pairwise_distance = np.linalg.norm(X[i] - X[j])
                    constraint_applies = constraint_applies and (pairwise_distance <= pairwise_threshold)
                
                if constraint_applies:
                    violations.append((i, j))

        return violations

    def _count_violations(self,
                         X: npt.NDArray[np.float64],
                         assignments: npt.NDArray[np.int64],
                         eligible_items: Set[int],
                         pairwise_threshold: Optional[float]) -> Tuple[int, List[Tuple[int, int]]]:
        """Count number of constraint violations and return list of violated constraints"""

        violations = self._find_violations(X, assignments, eligible_items, pairwise_threshold)
        return len(violations), violations

    def _compute_metrics(self,
                        X: npt.NDArray[np.float64],
                        assignments: npt.NDArray[np.int64]) -> ClusteringMetrics:
        """Compute clustering metrics"""

        # Calculate inertia
        distances = self._compute_distances(X)
        min_distances = np.min(distances, axis=0)
        inertia = np.sum(min_distances)

        # Only check violations if constraints have weight
        if self.w_cl > 0:
            # Compute thresholds once
            centroid_threshold, pairwise_threshold = self._compute_dual_thresholds(X, assignments)
            
            # Get eligible items based on centroid threshold
            eligible_items = self._get_constraint_eligible_items(X, assignments, centroid_threshold)
            
            num_violations, violated_constraints = self._count_violations(X, assignments, eligible_items, pairwise_threshold)
        else:
            num_violations = 0
            violated_constraints = []

        # Calculate total cost
        total_cost = inertia + self.w_cl * num_violations

        return ClusteringMetrics(inertia, num_violations, total_cost, violated_constraints)

    def _assign_points(self,
                      X: npt.NDArray[np.float64]) -> npt.NDArray[np.int64]:
        """Assign points to clusters considering cannot-link constraints with dual thresholds"""

        n_samples = X.shape[0]
        assignments = np.zeros(n_samples, dtype=np.int64)

        # Calculate base distances to all centers
        distances = self._compute_distances(X)

        # Add small epsilon to prevent exactly equal distances
        distances += np.random.uniform(0, 1e-10, distances.shape)
        
        if self.w_cl == 0:
            # If no constraint weight, assign all points at once
            return np.argmin(distances, axis=0)

        # First pass: assign all points without constraints to get initial assignments
        initial_assignments = np.argmin(distances, axis=0)
        
        # Compute thresholds once based on initial assignments
        centroid_threshold, pairwise_threshold = self._compute_dual_thresholds(X, initial_assignments)
        eligible_items = self._get_constraint_eligible_items(X, initial_assignments, centroid_threshold)

        # Assign points one by one
        for i in range(n_samples):
            cluster_costs = distances[:, i].copy()

            # Add penalty for constraint violations based on dual thresholds
            # candidates = self.constraint_graph.get_set(i)
            candidates = self.constraint_graph.get(i, set())
            for j in candidates:
                if j < i:  # Only consider already assigned points
                    # Check dual threshold conditions
                    constraint_applies = True
                    
                    # Check centroid threshold (via eligible items)
                    constraint_applies = constraint_applies and (i in eligible_items and j in eligible_items)
                    
                    # Check pairwise threshold
                    if constraint_applies and pairwise_threshold is not None:
                        pairwise_distance = np.linalg.norm(X[i] - X[j])
                        constraint_applies = constraint_applies and (pairwise_distance <= pairwise_threshold)
                    
                    if constraint_applies:
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

        # Skip violation tracking if constraints have no weight
        if self.w_cl == 0:
            return
        
        # Convert to set of violated constraints for this iteration
        # Use frozenset for each constraint to ensure hashability
        violated_set = {frozenset(constraint) for constraint in violated_constraints}
        
        # Add to history (only count, not the actual constraints)
        self.violations_per_iteration.append(len(violated_set))
        
        # For persistent violations tracking, use set operations
        if not hasattr(self, 'potential_persistent_violations'):
            # First iteration - all current violations are potentially persistent
            self.potential_persistent_violations = violated_set
        else:
            # Keep only violations that appear in both current and previous sets
            self.potential_persistent_violations &= violated_set
        
        # Update violation counts (only if needed for statistics)
        # This is an expensive operation, so we could make it optional
        for constraint in violated_constraints:
            frozen_constraint = frozenset(constraint)
            self.violation_counts[frozen_constraint] = self.violation_counts.get(frozen_constraint, 0) + 1

    def fit(self,
            X: npt.NDArray[np.float64],
            sorted_constraints: List[Tuple[int, int]],
            constraint_graph: Dict[int, Set[int]],
            skip_init: bool = False) -> 'ConstrainedKMeans':
        """
        Fit the Constrained KMeans clustering model

        Args:
            X: Training data
            sorted_constraints_path: Path to the sorted constraints database
            constraint_graph_path: Path to the constraint graph database
            skip_init: If True, use existing cluster centers for initialization

        Returns:
            self: Fitted model
        """

        # Initialize centers

        if skip_init:
            # Ensure cluster_centers_ has already been set
            if self.cluster_centers_ is None:
                raise ValueError("Custom initialization requires cluster_centers_ to be set")
        else:
            print("Initializing cluster centers...", flush=True)
            self.cluster_centers_ = self.initializer.initialize(
                X, self.n_clusters, sorted_constraints, self.random_state
            )

        # Initialize tracking variables
        self.violation_counts = {}
        self.violations_per_iteration = []
        self.all_violations_history = []

        # if self.w_cl > 0:
        #     # Process constraints only once
        #     print("Building constraint graph...", flush=True)
        #     if self.constraint_graph == None and self.sorted_constraints == None:
        #         self.constraint_graph, self.sorted_constraints = self._build_constraint_graph(len(X), cl_constraints)
        # else:
        #     # Empty placeholders when w_cl = 0
        #     self.constraint_graph = {}
        #     self.sorted_constraints = []

        # self.sorted_constraints = ConstraintFlatDB(sorted_constraints_path)
        # self.constraint_graph = ConstraintGraphDB(constraint_graph_path)

        # Initialize constraint graph and sorted constraints
        self.sorted_constraints = sorted_constraints
        self.constraint_graph = constraint_graph

        # Initialize history and tracking variables
        self.history_ = []
        best_cost = float('inf')
        best_centers = None
        best_labels = None
        iterations_without_improvement = 0

        # Main clustering loop
        print("Starting clustering iterations...", flush=True)
        for iteration in tqdm(range(self.max_iter)):
            # Get new assignments
            new_assignments = self._assign_points(X)
            
            # Update centers
            new_centers = self._update_centers(X, new_assignments)
            
            # Compute metrics
            metrics = self._compute_metrics(X, new_assignments)
            self.history_.append(metrics)
            
            # Print constraint violations for this iteration
            print(f"Iteration {iteration + 1}: Constraint violations = {metrics.constraint_violations}", flush=True)
            
            # Update violation statistics only if needed
            if self.w_cl > 0:
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

        if self.w_cl > 0 and hasattr(self, 'potential_persistent_violations'):
            self.persistent_violations = [tuple(sorted(v)) for v in self.potential_persistent_violations]
        else:
            self.persistent_violations = []

        # self.sorted_constraints.close()
        # self.constraint_graph.close()

        return self

    def predict(self, X: npt.NDArray[np.float64]) -> npt.NDArray[np.int64]:
        """Predict cluster labels for new data"""

        if self.cluster_centers_ is None:
            raise ValueError("Model must be fitted before making predictions")

        distances = self._compute_distances(X)
        return np.argmin(distances, axis=0)

    def get_violation_statistics(self) -> Dict:
        """Get comprehensive violation statistics"""

        if self.w_cl <= 0:
            return None
        
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

    def save(self, config, embeddings, skip_init):
        """Save the model to a file."""

        print("Saving clustering results...", flush=True)

        output = {
            "number_cluster": self.n_clusters,
            "w_cl": self.w_cl,
            "centroid_percentile":self.centroid_percentile,
            "pairwise_percentile":self.pairwise_percentile,
            "cluster_centers": self.cluster_centers_,
            "labels": self.labels_,
            "embeddings": embeddings,
            "violations": self.get_violation_statistics()
        }

        if skip_init:
            skip="skip"
        else:
            skip=""
        with open(config["clusters_path"] + f"clusters_{self.n_clusters}_{self.w_cl}_{skip}.pickle", 'wb') as f:
            pickle.dump(output, f, protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PCKmeans Clustering')
    parser.add_argument('-c', metavar='CONF', default='base', help='configuration (see config.conf)')
    parser.add_argument('-k', metavar='N_CLUSTERS', default=5, type=int, help='number of clusters')
    parser.add_argument('-i', metavar='MAX_ITER', default=100, type=int, help='maximum number of iterations')
    parser.add_argument('-w', metavar='W_CL', default=1.0, type=float, help='weight for cannot-link constraints')
    parser.add_argument('--centroid_percentile', metavar='CENTROID_PERCENTILE', default=None, type=float, help='percentile threshold for distance to cluster centers (e.g., 25 for top 25%)')
    parser.add_argument('--pairwise_percentile', metavar='PAIRWISE_PERCENTILE', default=None, type=float, help='percentile threshold for pairwise distances (e.g., 10 for bottom 10%)')
    parser.add_argument('--skip_init', action='store_true', help='skip initialization and use scikit-learn KMeans for initialization')

    args = parser.parse_args()
    config = ConfigFactory.parse_file('./config.conf')[args.c]

    print("Running Weighted PCKmeans Clustering with the following parameters:", flush=True)

    print("N_CLUSTERS: " + str(args.k), flush=True)
    print("W_CL: " + str(args.w), flush=True)
    if args.centroid_percentile is not None:
        print("CENTROID_PERCENTILE: " + str(args.centroid_percentile), flush=True)
    if args.pairwise_percentile is not None:
        print("PAIRWISE_PERCENTILE: " + str(args.pairwise_percentile), flush=True)

    print("Loading data for clustering...", flush=True)

    # Load data
    with open(config["processed_chains_path"], 'rb') as f:
        data = pickle.load(f)
    chain_sents = data['chain_sents']

    sbert_model = SentenceTransformer(config["cluster_model"])

    embeddings = sbert_model.encode(
        chain_sents, batch_size=32, show_progress_bar=True, normalize_embeddings=True
    )

    # Create initializer (either standard or constraint-aware)
    initializer = KMeansPlusPlusInit(
        strategy=InitializationStrategy.CONSTRAINT_AWARE,
        w_cl=args.w
    )

    # Create and fit the model
    model = ConstrainedKMeans(
        n_clusters=args.k,
        initializer=initializer,
        w_cl=args.w,
        max_iter=args.i,
        tol=1e-4,
        early_stopping_tol=5,
        random_state=config['seed'],
        centroid_percentile=args.centroid_percentile,
        pairwise_percentile=args.pairwise_percentile
    )

    sorted_constraints = ConstraintFlatDB(config["constraints_flat_path"]).get_all_tuples_as_list()
    constraint_graph = ConstraintGraphDB(config["constraints_graph_path"]).get_all_sets_as_dict()
    print("Number of constraints: " + str(len(sorted_constraints)), flush=True)
    print("skip_init: " + str(args.skip_init), flush=True)

    if args.skip_init:
        print("Using scikit-learn KMeans for initialization...", flush=True)
        sk_kmeans = KMeans(n_clusters=args.k, random_state=config["seed"], init='k-means++', n_init=5)
        sk_kmeans.fit(embeddings)
        model.cluster_centers_ = sk_kmeans.cluster_centers_

    # Fit the model
    model.fit(X=embeddings,
              sorted_constraints=sorted_constraints,
              constraint_graph=constraint_graph,
              skip_init=args.skip_init)

    # Compute and print purity results after clustering is complete
    print("\n=== Purity Results ===", flush=True)
    clustering_data = {
        "number_cluster": model.n_clusters,
        "labels": model.labels_,
        "embeddings": embeddings,
        "cluster_centers": model.cluster_centers_
    }
    purity_calculator = Purity(config["processed_chains_path"], clustering_data)
    purity_calculator.compute_purity()
    purity_calculator.print_results()
    print("======================\n", flush=True)

    # Run regression model after purity computation
    print("\n=== Regression Results ===", flush=True)
    regression_model = RegressionModel(config)
    regression_model.run_regression(config, clustering_data)
    print("==========================\n", flush=True)

    model.save(config, embeddings, args.skip_init)