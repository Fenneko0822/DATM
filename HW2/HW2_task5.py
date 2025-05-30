kmeans_spherical = kmeans(spherical_data, 3, max_iters=100)
kmeans_nonspherical = kmeans(nonspherical_data, 2, max_iters=100)
    
agg_spherical = agglomerative_clustering(spherical_data, 3)
agg_nonspherical = agglomerative_clustering(nonspherical_data, 2)


print(f'kmeans_spherical_silhouette: {silhouette_score(spherical_data, kmeans_spherical.labels_):.4f}')
print(f'kmeans_spherical_purity: {purity_score(spherical_labels, kmeans_spherical.labels_):.4f}')

print(f'kmeans_nonspherical_silhouette: {silhouette_score(nonspherical_data, kmeans_nonspherical.labels_):.4f}')
print(f'kmeans_nonspherical_purity: {purity_score(nonspherical_labels, kmeans_nonspherical.labels_):.4f}')

print(f'agg_spherical_silhouette: {silhouette_score(spherical_data, agg_spherical.labels_):.4f}')
print(f'agg_spherical_purity: {purity_score(spherical_labels, agg_spherical.labels_):.4f}')

print(f'agg_nonspherical_silhouette: {silhouette_score(nonspherical_data, agg_nonspherical.labels_):.4f}')
print(f'agg_nonspherical_purity: {purity_score(nonspherical_labels, agg_nonspherical.labels_):.4f}')

