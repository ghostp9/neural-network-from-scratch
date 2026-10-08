import numpy as np
from nnfs.layers import Dense, Sigmoid
from nnfs.loss import mse, mse_derivative
from nnfs.optimizer import SGD
from nnfs.network import Network

X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
y = np.array([[0], [1], [1], [0]], dtype=float)

net = Network(
    layers=[Dense(2, 4, seed=1), Sigmoid(), Dense(4, 1, seed=2), Sigmoid()],
    loss=mse,
    loss_derivative=mse_derivative,
    optimizer=SGD(lr=1.0),
)

net.fit(X, y, epochs=10000, verbose=1000)

print("\npredictions:")
print(np.round(net.predict(X), 3))
print("classes:", net.predict_classes(X).ravel())
