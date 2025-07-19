import pickle

import numpy as np
from sklearn.metrics import silhouette_score, calinski_harabasz_score


class ClusterMetrics:
    def __init__(self, clustering_data):
        self.embeddings = clustering_data["embeddings"]
        self.labels = clustering_data["labels"]
        self.cluster_centers = clustering_data["cluster_centers"]
        self.n_clusters = clustering_data["number_cluster"]
        self.n_samples = len(self.embeddings)
        
    def compute_cohesion(self):
        """Compute Within-Cluster Sum of Squares (WCSS)"""
        # Vectorized computation: get centers for each point based on labels
        centers_per_point = self.cluster_centers[self.labels]
        # Compute squared distances all at once
        wcss = np.sum((self.embeddings - centers_per_point) ** 2)
        return wcss
    
    def compute_separation(self):
        """Compute Between-Cluster Sum of Squares (BCSS)"""
        overall_center = np.mean(self.embeddings, axis=0)
        # Vectorized computation: get cluster sizes for all clusters at once
        cluster_sizes = np.bincount(self.labels, minlength=self.n_clusters)
        # Compute squared distances from each center to overall center
        center_distances = np.sum((self.cluster_centers - overall_center) ** 2, axis=1)
        # Multiply by cluster sizes and sum
        bcss = np.sum(cluster_sizes * center_distances)
        return bcss
    
    def compute_calinski_harabasz(self):
        """Compute Calinski-Harabasz Index"""
        return calinski_harabasz_score(self.embeddings, self.labels)
    
    def compute_silhouette(self):
        """Compute Silhouette Score"""
        if self.n_clusters == 1:
            return 0.0
        return silhouette_score(self.embeddings, self.labels)
    
    def compute_all_metrics(self):
        """Compute all clustering metrics"""
        cohesion = self.compute_cohesion()
        separation = self.compute_separation()
        calinski_harabasz = self.compute_calinski_harabasz()
        silhouette = self.compute_silhouette()
        
        return {
            "cohesion": cohesion,
            "separation": separation,
            "calinski_harabasz": calinski_harabasz,
            "silhouette": silhouette
        }
    
    def print_results(self):
        """Print all metrics with interpretation guides"""
        metrics = self.compute_all_metrics()
        
        print("=== CLUSTERING QUALITY METRICS ===")
        print(f"Number of clusters: {self.n_clusters}")
        print(f"Number of samples: {self.n_samples}")
        print()
        
        print("--- COHESION (Within-Cluster Sum of Squares) ---")
        print(f"WCSS: {metrics['cohesion']:.4f}")
        print("Interpretation: Lower values indicate tighter clusters")
        print("Guide: Compare across different k values - lower is better")
        print()
        
        print("--- SEPARATION (Between-Cluster Sum of Squares) ---")
        print(f"BCSS: {metrics['separation']:.4f}")
        print("Interpretation: Higher values indicate better separated clusters")
        print("Guide: Compare across different k values - higher is better")
        print()
        
        print("--- CALINSKI-HARABASZ INDEX ---")
        print(f"CH Index: {metrics['calinski_harabasz']:.4f}")
        print("Interpretation: Higher values indicate better clustering")
        if metrics['calinski_harabasz'] > 100:
            print("Guide: Very good clustering (CH > 100)")
        elif metrics['calinski_harabasz'] > 10:
            print("Guide: Reasonable clustering (CH > 10)")
        else:
            print("Guide: Poor clustering (CH < 10)")
        print()
        
        print("--- SILHOUETTE SCORE ---")
        print(f"Silhouette Score: {metrics['silhouette']:.4f}")
        print("Interpretation: Range [-1, 1], higher values indicate better clustering")
        if metrics['silhouette'] > 0.7:
            print("Guide: Excellent clustering (> 0.7)")
        elif metrics['silhouette'] > 0.5:
            print("Guide: Good clustering (> 0.5)")
        elif metrics['silhouette'] > 0.25:
            print("Guide: Weak clustering (> 0.25)")
        else:
            print("Guide: Poor clustering (< 0.25)")
        print()
        
        print("=== RECOMMENDATIONS ===")
        print("- Use Calinski-Harabasz for k-selection (find peak value)")
        print("- Use Silhouette Score for validation (aim for > 0.5)")
        print("- Lower WCSS + Higher BCSS = Better clustering")
        print("===================================")


def main(clustering_data):
    """Main function to compute and display clustering metrics"""
    metrics_calculator = ClusterMetrics(clustering_data)
    metrics_calculator.print_results()
    return metrics_calculator.compute_all_metrics()


if __name__ == "__main__":
    # Example usage
    with open("./data/mfc/immigration/clustering/clusters_150_0.0.pickle", "rb") as f:
        clustering_data = pickle.load(f)

    main(clustering_data)