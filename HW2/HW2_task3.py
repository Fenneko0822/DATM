import numpy as np
from scipy.spatial.distance import pdist, squareform

def compute_distance_matrix(X):
    """Computes the pairwise distance matrix."""
    pairwise_distances = pdist(X, metric='euclidean')
    distance_matrix = squareform(pairwise_distances)
    return distance_matrix

def find_closest_clusters(distances):
    """Finds the indices of the two closest clusters."""
    min_dist_index = np.argmin(distances)  # Find the smallest distance (ignoring diagonal)
    i, j = np.unravel_index(min_dist_index, distances.shape)  # Convert index to row and column
    return i, j

def update_distances(distances, clusters, i, j):
    """Updates the distance matrix after merging clusters i and j using single linkage."""
    cluster_j_points = clusters[j]  # Store points in cluster j
    
    for idx in clusters:
        if idx != i and idx != j:  # Skip already merged clusters
            # Compute the minimum distance between points in clusters i and j
            new_dist = np.min([distances[p1, p2] for p1 in clusters[i] for p2 in cluster_j_points])
            distances[idx, i] = distances[i, idx] = new_dist  # Update distances for cluster i
            
    distances[:, j] = distances[j, :] = np.inf  # Set all distances to/from cluster j to infinity to avoid re-merging

def agglomerative_clustering(X, k):
    """Performs agglomerative hierarchical clustering with single linkage."""
    num_points = len(X)
    clusters = {i: [i] for i in range(num_points)}  # Initialize clusters with each point as its own cluster
    distances = compute_distance_matrix(X)  # Compute the initial pairwise distance matrix
    
    while len(clusters) > k:
        i, j = find_closest_clusters(distances)  # Find the two closest clusters
        
        if i not in clusters or j not in clusters:  # Skip if the clusters no longer exist
            continue
        
        # Merge cluster j into cluster i
        clusters[i].extend(clusters[j])  
        update_distances(distances, clusters, i, j)  # Update the distance matrix after merging
        del clusters[j]  # Remove cluster j after merging
        
    # Renumber clusters from 0, 1, ..., N-1
    unique_clusters = list(clusters.keys())
    cluster_mapping = {old: new for new, old in enumerate(unique_clusters)}  # Mapping old cluster IDs to new ones

    labels = np.zeros(num_points, dtype=int)  # Initialize an array to store the cluster labels
    for cluster_id, points in clusters.items():
        for point in points:
            labels[point] = cluster_mapping[cluster_id]  # Assign the correct label to each point
    
    return labels
