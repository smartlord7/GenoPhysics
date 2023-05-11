import numpy as np

def sigmoid(predicted, real):
    """
    Calculates the sigmoid of the error between predicted and real values.
    Used for maximization problems.

    Parameters:
    predicted : numpy array
        Array of predicted values.
    real : numpy array
        Array of actual (real) values.

    Returns:
    float
        Sigmoid of the error between predicted and real values.
    """
    error = np.sum(np.abs(predicted - real))
    return 1 / (1 + error)

def sse(predicted, real):
    """
    Calculates the sum of squared errors between predicted and real values.
    Used for minimization problems.

    Parameters:
    predicted : numpy array
        Array of predicted values.
    real : numpy array
        Array of actual (real) values.

    Returns:
    float
        Sum of squared errors between predicted and real values.
    """
    return np.sum((predicted - real) ** 2)

def mse(predicted, real):
    """
    Calculates the mean squared error between predicted and real values.
    Used for minimization problems.

    Parameters:
    predicted : numpy array
        Array of predicted values.
    real : numpy array
        Array of actual (real) values.

    Returns:
    float
        Mean squared error between predicted and real values.
    """
    return 1 / len(predicted) * np.sum((predicted - real) ** 2)

def rmse(predicted, real):
    """
    Calculates the root mean squared error between predicted and real values.
    Used for minimization problems.

    Parameters:
    predicted : numpy array
        Array of predicted values.
    real : numpy array
        Array of actual (real) values.

    Returns:
    float
        Root mean squared error between predicted and real values.
    """
    return np.sqrt(1 / len(predicted) * np.sum((predicted - real) ** 2))
