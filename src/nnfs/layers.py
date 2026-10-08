import numpy as np
from nnfs.activations import sigmoid, sigmoid_derivative, relu, relu_derivative


class Dense:
    def __init__(self, n_inputs, n_neurons, seed=None):
        rng = np.random.default_rng(seed)
        self.W = rng.standard_normal((n_inputs, n_neurons)) * np.sqrt(2.0 / n_inputs)
        self.b = np.zeros((1, n_neurons))
        self.x = None
        self.dW = None
        self.db = None

    def forward(self, x):
        self.x = x
        return x @ self.W + self.b

    def backward(self, grad_out):
        self.dW = self.x.T @ grad_out
        self.db = np.sum(grad_out, axis=0, keepdims=True)
        return grad_out @ self.W.T


class Activation:
    def __init__(self, fn, fn_derivative):
        self.fn = fn
        self.fn_derivative = fn_derivative
        self.z = None

    def forward(self, z):
        self.z = z
        return self.fn(z)

    def backward(self, grad_out):
        return grad_out * self.fn_derivative(self.z)


class Sigmoid(Activation):
    def __init__(self):
        super().__init__(sigmoid, sigmoid_derivative)


class ReLU(Activation):
    def __init__(self):
        super().__init__(relu, relu_derivative)
