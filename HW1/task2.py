import numpy as np

def linear_kernel(x1, x2):
    """
    Compute the linear kernel between two vectors.
    k(x1, x2) = x1.T @ x2
    """
    # TODO: implement the function
    return np.dot(x1, x2)

def polynomial_kernel(x1, x2, degree=3, coef0=1):
    """
    Compute the polynomial kernel between two vectors.
    k(x1, x2) = (x1.T @ x2 + coef0)^degree
    """
    # TODO: implement the function
    return (np.dot(x1, x2) + coef0) ** degree

def rbf_kernel(x1, x2, gamma=0.1):
    """
    Compute the Gaussian (RBF) kernel between two vectors.
    k(x1, x2) = exp(-gamma * ||x1 - x2||^2)
    """
    # TODO: implement the function
    return np.exp(-gamma * np.linalg.norm(x1 - x2) ** 2)

def sigmoid_kernel(x1, x2, alpha=0.1, coef0=0):
    """
    Compute the sigmoid kernel between two vectors.
    k(x1, x2) = tanh(alpha * (x1.T @ x2) + coef0)
    """
    # TODO: implement the function
    return np.tanh(alpha * np.dot(x1, x2) + coef0)