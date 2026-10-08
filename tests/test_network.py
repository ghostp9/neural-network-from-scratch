import numpy as np
from nnfs.layers import Dense, Sigmoid
from nnfs.loss import mse, mse_derivative
from nnfs.optimizer import SGD
from nnfs.network import Network


def test_xor_learns():
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
    y = np.array([[0], [1], [1], [0]], dtype=float)
    net = Network(
        [Dense(2, 4, seed=1), Sigmoid(), Dense(4, 1, seed=2), Sigmoid()],
        mse, mse_derivative, SGD(lr=1.0),
    )
    net.fit(X, y, epochs=10000, verbose=0)
    assert list(net.predict_classes(X).ravel()) == [0, 1, 1, 0]
