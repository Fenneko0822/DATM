import numpy as np
from sklearn.metrics import accuracy_score
from itertools import product

class NestedCrossValidator:
    def __init__(self, model_class, param_grid, outer_splits=5, inner_splits=3, random_seed=None):
        """
        Initialize the nested cross-validator.

        :param model_class: The model class to be used (e.g., SVC, RandomForestClassifier).
        :param param_grid: Dictionary of hyperparameters and their values to search.
        :param outer_splits: Number of splits for the outer cross-validation.
        :param inner_splits: Number of splits for the inner cross-validation.
        :param random_seed: Random seed for reproducibility.
        """
        self.model_class = model_class
        self.param_grid = param_grid
        self.outer_splits = outer_splits
        self.inner_splits = inner_splits
        self.random_seed = random_seed

    def _create_folds(self, X, y, n_splits, data_fraction=1.0):
        """
        Manually create folds for cross-validation.

        :param X: Feature matrix.
        :param y: Target vector.
        :param n_splits: Number of splits for cross-validation.
        :param data_fraction: Fraction of data to use (between 0.0 and 1.0).
        :return: List of (train_indices, test_indices) for each fold.
        """
        # Validate data_fraction
        if not 0.0 < data_fraction <= 1.0:
            raise ValueError("data_fraction must be between 0.0 and 1.0 (exclusive).")

        np.random.seed(self.random_seed) # DO NOT DELETE THIS LINE
        total_indices = np.arange(len(X))
        subset_size = int(len(X) * data_fraction)
        # TODO: implement this function
        # Step 1: Select a subset of data indices based on data_fraction. (use np.random.choice function)
        selected_indices = np.random.choice(total_indices, subset_size, replace=False)

        # Step 2: Shuffle the selected indices. (use np.random.shuffle function)
        np.random.shuffle(selected_indices)

        # Step 3: Calculate the size for each fold split
        fold_size = subset_size // n_splits
        folds = []

        for i in range(n_splits):
            # Step 3: Split the shuffled indices into n_splits folds. 
            test_indices = selected_indices[i * fold_size:(i + 1) * fold_size]
            train_indices = np.setdiff1d(selected_indices, test_indices)
            folds.append((train_indices, test_indices))

        return folds
    


    def fit(self, X, y, outer_data_fraction=1.0, inner_data_fraction=0.5):
        """
        Perform nested cross-validation to evaluate model performance.

        :param X: Feature matrix.
        :param y: Target vector.
        :param outer_data_fraction: Fraction of data to use for outer loop.
        :param inner_data_fraction: Fraction of data to use for inner loop.
        :return: List of outer fold accuracies and mean accuracy.
        """
        # TODO: implement this function
        # Step 1: Create outer folds using _create_folds().
        outer_folds = self._create_folds(X, y, self.outer_splits, outer_data_fraction)

        # Step 2: Initialize a list to store outer fold results.
        outer_results = []

        # Step 3: Loop through each outer fold.
        for outer_idx, (train_idx, test_idx) in enumerate(outer_folds):
            # Step 4: Split data into outer train/test sets.
            X_train_outer, y_train_outer = X[train_idx], y[train_idx]
            X_test_outer, y_test_outer = X[test_idx], y[test_idx]

            # Inner loop for hyperparameter tuning
            # Step 5: Initialize best parameters and best score.
            best_params = None
            best_score = -np.inf

            # Step 6: Create inner folds from outer training data. (Don't forget inner_data_fraction here)
            inner_folds = self._create_folds(X_train_outer, y_train_outer, self.inner_splits, inner_data_fraction)

            # Step 7: Iterate over hyperparameter combinations.
            for param_combination in product(*self.param_grid.values()):
                params = dict(zip(self.param_grid.keys(), param_combination))
                inner_scores = []

                # Step 8: Train model using inner train sets and evaluate using validation sets.
                for inner_train_idx, inner_val_idx in inner_folds:
                    X_train_inner, y_train_inner = X_train_outer[inner_train_idx], y_train_outer[inner_train_idx]
                    X_val_inner, y_val_inner = X_train_outer[inner_val_idx], y_train_outer[inner_val_idx]

                    model = self.model_class(**params)
                    # Train model with the current parameters
                    model.fit(X_train_inner, y_train_inner)

                    # Evaluate on validation set and update inner_scores
                    y_pred_inner = model.predict(X_val_inner)
                    inner_scores.append(accuracy_score(y_val_inner, y_pred_inner))

                # Step 9: Select the best hyperparameter combination.
                # Compute mean score and update best parameters
                mean_inner_score = np.mean(inner_scores)
                if mean_inner_score > best_score:
                    best_score = mean_inner_score
                    best_params = params

            # Step 10: Train the final model on outer training data using best parameters.
            final_model = self.model_class(**best_params)
            final_model.fit(X_train_outer, y_train_outer)

            # Step 11: Evaluate the final model on outer test set and store the result.
            y_pred_outer = final_model.predict(X_test_outer)
            outer_accuracy = accuracy_score(y_test_outer, y_pred_outer)
            outer_results.append(outer_accuracy)

            print(f"Outer Fold {outer_idx+1}: Accuracy = {outer_accuracy:.4f} with Params {best_params}")

        mean_outer_accuracy = np.mean(outer_results)
        print(f"Mean Outer Accuracy: {mean_outer_accuracy:.4f}")

        return outer_results, mean_outer_accuracy
