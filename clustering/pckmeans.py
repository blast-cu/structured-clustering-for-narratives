import argparse
import pickle
from collections import defaultdict

import numpy as np
from pyhocon import ConfigFactory
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances
from tqdm import tqdm


class PCKMeans:
    def __init__(self, n_clusters, max_iter=100, tol=1e-4, random_state=42):
        """
        Initialize Pairwise Constrained KMeans with cannot-link constraints.

        Parameters:
        -----------
        n_clusters : int
            Number of clusters
        max_iter : int, default=100
            Maximum number of iterations
        tol : float, default=1e-4
            Relative tolerance with regards to Frobenius norm of the difference
            in the cluster centers of two consecutive iterations to declare convergence
        random_state : int, default=None
            Random state for reproducibility
        """
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.kmeans = KMeans(
            n_clusters=n_clusters,
            init='k-means++',
            max_iter=1,  # We'll handle iterations ourselves, we run once to find initial centroids
            random_state=random_state
        )

    def fit(self, X, cannot_link_constraints):
        """
        Fit the Pairwise Constrained KMeans model.

        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Training data
        cannot_link_constraints : array-like of shape (n_constraints, 2)
            Pairs of point indices that should not be in the same cluster

        Returns:
        --------
        self : object
            Fitted estimator
        """
        self.X = X
        self.cannot_link_constraints = cannot_link_constraints
        self.n_samples = X.shape[0]

        # Initialize clusters using k-means++
        self.kmeans.fit(X)
        self.cluster_centers_ = self.kmeans.cluster_centers_
        self.labels_ = self.kmeans.labels_

        # Initialize constraint violation tracking
        self.violation_history = defaultdict(list)  # {constraint: [iterations_violated]}
        self.violations_per_iteration = []  # List of violations for each iteration

        # Initialize inertia
        self.inertia_ = np.inf

        # Main loop
        for iteration in tqdm(range(self.max_iter)):
            old_centers = self.cluster_centers_.copy()
            old_inertia = self.inertia_

            # Update cluster assignments
            self._update_assignments(iteration)

            # Update cluster centers
            self._update_centers()

            # Compute new inertia
            self.inertia_ = self._compute_inertia()

            # Check convergence using Frobenius norm of the difference in centers
            center_shift = np.linalg.norm(old_centers - self.cluster_centers_, ord='fro')

            if center_shift <= self.tol:
                break

        # Compute summary statistics for violations
        self.total_violations = sum(len(iterations) for iterations in self.violation_history.values())
        self.consistently_violated = [constraint for constraint, iterations in
                                      self.violation_history.items() if
                                      len(iterations) == len(self.violations_per_iteration)]

        return self

    def _compute_inertia(self):
        """Compute sum of squared distances of samples to their closest cluster center."""
        distances = euclidean_distances(self.X, self.cluster_centers_)
        return np.sum(np.min(distances, axis=1) ** 2)

    def _update_assignments(self, iteration):
        """Update cluster assignments considering cannot-link constraints."""
        current_violations = []
        distances = euclidean_distances(self.X, self.cluster_centers_)

        # First pass: assign points to their closest centers
        self.labels_ = np.argmin(distances, axis=1)

        # Second pass: handle constraints
        for i, j in self.cannot_link_constraints:
            constraint = tuple(sorted([i, j]))  # Ensure consistent ordering
            if self.labels_[i] == self.labels_[j]:
                # Record violation
                current_violations.append(constraint)
                self.violation_history[constraint].append(iteration)

                # Try to reassign one of the points to its next best cluster
                dist_i = distances[i]
                dist_j = distances[j]

                # Sort clusters by distance for both points
                sorted_clusters_i = np.argsort(dist_i)
                sorted_clusters_j = np.argsort(dist_j)

                # Try to find the best reassignment that satisfies the constraint
                reassigned = False

                # Try reassigning point i
                for cluster in sorted_clusters_i[1:]:  # Skip the current best cluster
                    if cluster != self.labels_[j]:
                        self.labels_[i] = cluster
                        reassigned = True
                        break

                # If we couldn't reassign point i, try reassigning point j
                if not reassigned:
                    for cluster in sorted_clusters_j[1:]:
                        if cluster != self.labels_[i]:
                            self.labels_[j] = cluster
                            break

        self.violations_per_iteration.append(current_violations)

    def _update_centers(self):
        """Update cluster centers."""
        for k in range(self.n_clusters):
            cluster_points = self.X[self.labels_ == k]
            if len(cluster_points) > 0:
                self.cluster_centers_[k] = cluster_points.mean(axis=0)

    def predict(self, X):
        """Predict cluster labels for new data."""
        return np.argmin(euclidean_distances(X, self.cluster_centers_), axis=1)

    def get_violation_statistics(self):
        """Get comprehensive statistics about constraint violations."""
        return {
            'total_violations': self.total_violations,
            'violations_per_iteration': [len(v) for v in self.violations_per_iteration],
            'consistently_violated_constraints': self.consistently_violated,
            'violation_history': dict(self.violation_history)
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
    parser = argparse.ArgumentParser(description='Post Processing')
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

    pckmeans = PCKMeans(n_clusters=args.k, random_state=config['seed'])
    pckmeans.fit(data['embs'], data['constraints'])

    pckmeans.save(config)