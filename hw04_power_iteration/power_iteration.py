import numpy as np

def get_dominant_eigenvalue_and_eigenvector(data, num_steps):
    """
    data: np.ndarray – symmetric diagonalizable real-valued matrix
    num_steps: int – number of power method steps
    
    Returns:
    eigenvalue: float – dominant eigenvalue estimation after `num_steps` steps
    eigenvector: np.ndarray – corresponding eigenvector estimation
    """
    current_vector = np.ones(data.shape[1])
    for i in range(num_steps):
        data_mult_current = np.dot(data, current_vector)
        current_vector = data_mult_current / np.sqrt(np.dot(data_mult_current, data_mult_current))
    return float(np.dot(current_vector, np.dot(data, current_vector)) / np.dot(current_vector, current_vector)), current_vector