import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs, make_moons

def generate_datasets():
    """Generates two datasets: one spherical and one non-spherical."""
    X_spherical, y_spherical = make_blobs(n_samples=300, centers=3, random_state=42)
    
    X_nonspherical, y_nonspherical = make_moons(n_samples=300, noise=0.05, random_state=42)
    
    return (X_spherical, y_spherical), (X_nonspherical, y_nonspherical)

# Load datasets
spherical_data, spherical_labels = generate_datasets()[0]
nonspherical_data, nonspherical_labels = generate_datasets()[1]