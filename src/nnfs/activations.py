import numpy as np


def sigmoid(x):
    """Sigmoid: maps any real number to (0, 1)."""
    z = np.exp(-np.abs(x))
    return np.where(x >= 0, 1 / (1 + z), z / (1 + z))


def sigmoid_derivative(x):
    """Derivative of sigmoid with respect to its input x."""
    s = sigmoid(x)
    return s * (1 - s)


def relu(x):
    """ReLU: returns x if x > 0, otherwise 0."""
    return np.maximum(0, x)


def relu_derivative(x):
    """Derivative of ReLU with respect to its input x."""
    return (x > 0).astype(float)
