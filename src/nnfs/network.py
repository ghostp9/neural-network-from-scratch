import numpy as np


class Network:
    def __init__(self, layers, loss, loss_derivative, optimizer):
        self.layers = layers
        self.loss = loss
        self.loss_derivative = loss_derivative
        self.optimizer = optimizer

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, grad):
        for layer in reversed(self.layers):
            grad = layer.backward(grad)

    def fit(self, X, y, epochs=1000, verbose=100):
        history = []
        for epoch in range(1, epochs + 1):
            y_pred = self.forward(X)
            loss = self.loss(y, y_pred)
            history.append(loss)

            self.backward(self.loss_derivative(y, y_pred))
            self.optimizer.step(self.layers)

            if verbose and epoch % verbose == 0:
                print(f"epoch {epoch:5d}  loss {loss:.6f}")
        return history

    def predict(self, X):
        return self.forward(X)

    def predict_classes(self, X, threshold=0.5):
        return (self.predict(X) >= threshold).astype(int)
