import numpy as np

def initialize_centroids(X, k):
    """Initializes centroids using the KMeans++ strategy."""
    np.random.seed(42)
    centroids = []
    first_centroid = X[np.random.choice(X.shape[0])]
    centroids.append(first_centroid)
    
    for _ in range(1, k):
        distances = np.array([min(np.linalg.norm(x - c)**2 for c in centroids) for x in X])
        
        prob = distances / distances.sum() 
        next_centroid_idx = np.random.choice(X.shape[0], p=prob)
        centroids.append(X[next_centroid_idx])
    
    return np.array(centroids)

def assign_clusters(X, centroids):
    """Assigns each point to the nearest centroid."""
    distances = []
    for c in centroids:
        dist = np.linalg.norm(X - c, axis=1)  # Calculate distance
        distances.append(dist)
    
    distances = np.array(distances).T  
    labels = np.argmin(distances, axis=1) 
    return labels

def update_centroids(X, labels, k):
    """Recomputes centroids as the mean of assigned points."""
    new_centroids = []
    for i in range(k):
        cluster_points = X[labels == i]
        new_centroid = cluster_points.mean(axis=0)
        new_centroids.append(new_centroid)
    
    return np.array(new_centroids)

def kmeans(X, k, max_iters=100):
    """Performs KMeans clustering."""
    centroids = initialize_centroids(X, k)
    
    for _ in range(max_iters):
        labels = assign_clusters(X, centroids)
        new_centroids = update_centroids(X, labels, k)
        
        if np.all(centroids == new_centroids):
            break
        
        centroids = new_centroids
    
    return labels
