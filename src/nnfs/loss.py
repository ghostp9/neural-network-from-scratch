import numpy as np


def mse(y_true, y_pred):
    """Mean Squared Error."""
    return np.mean((y_pred - y_true) ** 2)


def mse_derivative(y_true, y_pred):
    """Gradient of MSE with respect to y_pred."""
    return 2 * (y_pred - y_true) / y_true.size
