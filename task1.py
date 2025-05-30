import numpy as np
from collections import Counter
from sklearn.tree import DecisionTreeClassifier


class RandomForest:
    """
    A basic implementation of a Random Forest algorithm for classification.

    This class builds an ensemble of decision trees using bootstrap sampling and random feature selection. 
    It combines predictions from individual trees using majority voting to produce a robust classification model.

    Attributes:
    ----------
    n_estimators : int
        Number of decision trees in the forest.
    max_depth : int or None
        Maximum depth of each tree. If None, the trees grow until pure leaves or until reaching `min_samples_split`.
    min_samples_split : int
        Minimum number of samples required to split an internal node in a tree.
    max_features : str or int
        Number of features to consider when looking for the best split.
        - "sqrt": Square root of the total number of features.
        - "log2": Base-2 logarithm of the total number of features.
        - int: A specific number of features to use.
    seed : int or None
        Random seed for reproducibility. If None, results will vary between runs.
    trees : list
        A list containing tuples of trained decision trees and their selected feature indices/names.
    """

    def __init__(self, n_estimators=10, max_depth=None, min_samples_split=2, max_features="sqrt", seed=None):
        """
        Initializes the Random Forest classifier.

        Parameters:
        ----------
        n_estimators : int, default=10
            Number of decision trees to build in the forest.
        max_depth : int or None, default=None
            Maximum depth of each decision tree. If None, trees grow until they are pure or cannot be split further.
        min_samples_split : int, default=2
            Minimum number of samples required to split an internal node.
        max_features : str or int, default="sqrt"
            Number of features to consider when splitting nodes:
            - "sqrt": Square root of the total number of features.
            - "log2": Base-2 logarithm of the total number of features.
            - int: Specific number of features.
        seed : int or None, default=None
            Random seed for reproducibility. If None, results may vary across runs.
        """
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.seed = seed
        self.trees = []

    def _bootstrap_sample(self, X, y):
        """
        Generates a bootstrap sample of the input dataset by sampling with replacement.

        Parameters:
        ----------
        X : numpy.ndarray
            Feature matrix with shape (n_samples, n_features).
        y : numpy.ndarray
            Target vector with shape (n_samples,).

        Returns:
        -------
        tuple
            A tuple (X_sample, y_sample, indices) where:
            - X_sample is the feature matrix of the bootstrap sample.
            - y_sample is the target vector of the bootstrap sample.
            - indices are the indices of the selected samples.
        """
        # TODO implement this function
        # Step 1: Use np.random.choice() to sample indices from X.
        #indices = ...
        n_samples = X.shape[0]
        indices = np.random.choice(n_samples, size=n_samples, replace=True) 
        # Step 2: Select the corresponding rows from X and y using the sampled indices.
        X_sample = X[indices] 
        y_sample = y[indices]

        return X_sample, y_sample, indices

    def _select_random_features(self, X, n_features):
        """
        Selects a random subset of features to use for a decision tree.

        Parameters:
        ----------
        X : numpy.ndarray
            Feature matrix with shape (n_samples, n_features).
        n_features : int
            Total number of features in the dataset.

        Returns:
        -------
        tuple
            A tuple (X_subset, feature_indices) where:
            - X_subset is the feature matrix containing only the selected features.
            - feature_indices are the indices of the selected features.
        """
        # TODO implement this function
        # Step 1: Compute the number of features to select (max_features).
        if self.max_features == "sqrt":
            max_features = int(np.sqrt(n_features))
        elif self.max_features == "log2":
            max_features = int(np.log2(n_features))
        else:
            max_features = n_features

        # Step 2: Randomly choose max_features feature indices using np.random.choice().
        feature_indices = np.random.choice(n_features, size=max_features, replace=False)

        # Step 3: Extract the corresponding columns from X using these indices.
        X_subset = X[:, feature_indices]

        return X_subset, feature_indices


    def fit(self, X, y):
        """
        Trains the Random Forest model by creating an ensemble of decision trees.

        Each tree is trained on a bootstrap sample of the data and a random subset of features.

        Parameters:
        ----------
        X : numpy.ndarray
            Feature matrix with shape (n_samples, n_features).
        y : numpy.ndarray
            Target vector with shape (n_samples,).
        """
        self.feature_names = None

        self.trees = []
        n_features = X.shape[1]
        np.random.seed(self.seed)

        for _ in range(self.n_estimators):
            # TODO implement this function
            # Create bootstrap sample
            # Step 1: Call _bootstrap_sample() to generate a training sample.
            X_sample, y_sample, _ = self._bootstrap_sample(X, y)
            
            # Step 2: Call _select_random_features() to select features for training.
            X_subset, feature_indices = self._select_random_features(X_sample, n_features)

            # Step 3: Train a DecisionTreeClassifier on the sampled data.
            # NOTE: For DecisionTreeClassifier
            #  - Only set max_depth (from self.) and min_samples_split from (self.)
            #  - Set random_state=0
            
            tree = DecisionTreeClassifier(max_depth=self.max_depth, 
                                          min_samples_split=self.min_samples_split, 
                                          random_state=0)
            tree.fit(X_subset, y_sample)

            # Step 4: Store the trained tree and the selected feature indices in self.trees.
            selected_features = feature_indices
            self.trees.append((tree, selected_features))


    def predict(self, X):
        """
        Predicts class labels for the input data using the trained Random Forest.

        Predictions are made by aggregating votes from all individual trees (majority voting).

        Parameters:
        ----------
        X : numpy.ndarray
            Feature matrix with shape (n_samples, n_features).

        Returns:
        -------
        numpy.ndarray
            Predicted class labels of shape (n_samples,).
        """
        predictions = np.zeros((X.shape[0], len(self.trees)))
        for i, (tree, selected_features) in enumerate(self.trees):
            feature_indices = selected_features

            predictions[:, i] = tree.predict(X[:, feature_indices])

        return np.apply_along_axis(lambda x: Counter(x).most_common(1)[0][0], axis=1, arr=predictions)