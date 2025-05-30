from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from itertools import product

def find_best_kernel(kernel_list, X, y, test_size=0.2, random_state=0):
    """
    Finds the best kernel for an SVM model using a single validation split.

    Parameters:
    - kernel_list (list): List of kernel names to test (e.g., ['linear', 'poly', 'rbf', 'sigmoid'])
    - X (array-like): Feature matrix
    - y (array-like): Target labels
    - test_size (float): Proportion of the dataset to use as the test set (default = 0.2)
    - random_state (int): Random seed for reproducibility (default = 0)

    Returns:
    - best_kernel (str): The kernel with the highest validation accuracy
    - best_score (float): The highest validation accuracy
    """
    # TODO: implement the function

    # 1. Use `train_test_split()` to split `X` and `y` into training (1-test_size) and testing (test_size) sets.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    best_score = 0
    best_kernel = None

    # 2. Iterate over each kernel in `kernel_list`.
    for kernel in kernel_list:
        # Train the SVM model
        # 3. Train an `SVC` model using the current kernel.
        model = SVC(kernel=kernel)
        model.fit(X_train, y_train) 

        # 4. Make predictions on the test set.
        y_pred = model.predict(X_test)

        # 5. Calculate accuracy using `accuracy_score()`.
        score = accuracy_score(y_test, y_pred)

        # 6. Keep track of the kernel with the highest accuracy.
        if score > best_score:
            best_score = score
            best_kernel = kernel

    return best_kernel, best_score
    



def find_best_svm_params(param_dict, X, y, test_size=0.2, random_state=0):
    """
    Finds the best SVM parameters using a single validation split.

    Parameters:
    - param_dict (dict): Dictionary containing parameter lists to test.
                         Example: {'kernel': ['linear', 'rbf'], 'C': [0.1, 1, 10]}
    - X (array-like): Feature matrix
    - y (array-like): Target labels
    - test_size (float): Proportion of the dataset to use as the test set (default = 0.2)
    - random_state (int): Random seed for reproducibility (default = 0)

    Returns:
    - best_params (dict): Dictionary of the best parameter combination
    - best_score (float): The highest validation accuracy
    """
    # TODO: implement the function

    # 1. Split the dataset into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    best_score = 0
    best_params = None

    # 2. Extract the hyperparameter keys from `param_dict`.
    keys = list(param_dict.keys())
    values = list(param_dict.values())
    # 3. Use `itertools.product()` to generate all possible parameter combinations.
    # 4. Iterate over each parameter combination:
    #     - Train an `SVC` model with the given parameters.
    #     - Make predictions on the test set.
    #     - Compute accuracy.
    #     - Store the best-performing parameter combination.
    # 5. Return the best hyperparameter set and its corresponding accuracy.

    # Generate all possible combinations of parameters
    for param_combination in product(*values):
        params = dict(zip(keys, param_combination))  
        model = SVC(**params, random_state=random_state)  
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)  
        score = accuracy_score(y_test, y_pred)  

        if score > best_score:
            best_score = score
            best_params = params

    return best_params, best_score


