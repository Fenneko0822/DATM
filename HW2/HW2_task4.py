import numpy as np
from sklearn import metrics
from sklearn.metrics import pairwise_distances

def silhouette_score(X, labels):
    """Computes silhouette score for clustering."""
    # Compute pairwise distances between all points
    distances = pairwise_distances(X)
    
    unique_labels = np.unique(labels)  # Get unique labels for clusters
    
    # Compute intra-cluster distance (a)
    a = []
    for i in range(len(X)):
        same_cluster = labels == labels[i]  # Mask for same cluster points
        intra_distances = distances[i, same_cluster]  # Get distances to same cluster points
        avg_distance = np.mean(intra_distances)  # Compute mean value of intra-cluster distances
        a.append(avg_distance)
    a = np.array(a)
    
    # Compute nearest-cluster distance (b)
    b = []
    for i in range(len(X)):
        other_cluster_distances = []  # Store avg distances to other clusters
        for l in unique_labels:
            if l != labels[i]:  # Skip the same cluster
                # Mask for other cluster points
                other_cluster_mask = labels == l
                cluster_distances = distances[i, other_cluster_mask]  # Extract distances with other cluster points
                if cluster_distances.size > 0:  # Ensure it's not empty
                    avg_distance = np.mean(cluster_distances)  # Compute mean distance to other cluster
                    other_cluster_distances.append(avg_distance)
        # Handle the case where other_cluster_distances is empty (shouldn't happen in normal cases)
        if other_cluster_distances:
            min_distance = np.min(other_cluster_distances)  # Compute minimum of other cluster distances
        else:
            min_distance = 0  # Default value if no other clusters exist (rare case)
        b.append(min_distance)
    b = np.array(b)
    
    # Return the Silhouette score
    return np.mean((b - a) / np.maximum(a, b))

from sklearn import metrics
def purity_score(y_true, y_pred):
    # compute contingency matrix (also called confusion matrix)
    contingency_matrix = metrics.cluster.contingency_matrix(y_true, y_pred)
    # return purity
    return np.sum(np.amax(contingency_matrix, axis=0)) / np.sum(contingency_matrix)
