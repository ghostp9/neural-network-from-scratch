# neural-network-from-scratch

A small neural network I built from scratch using only Python and NumPy, no PyTorch or TensorFlow. I made it to understand what actually happens inside a neural network instead of just calling `model.fit()`.

It learns XOR, which a single neuron can't solve, so it's a nice first test that the hidden layer and backpropagation really work.

## What's inside

```text
src/nnfs/
├── activations.py
├── layers.py
├── loss.py
├── optimizer.py
└── network.py
examples/xor.py
tests/
```

- `activations.py`: sigmoid, relu and their derivatives
- `layers.py`: Dense layer and activation layers (forward and backward)
- `loss.py`: mean squared error
- `optimizer.py`: plain gradient descent (SGD)
- `network.py`: ties everything together, training loop

## Setup

```bash
git clone https://github.com/ghostp9/neural-network-from-scratch.git
cd neural-network-from-scratch
python -m venv .venv
source .venv/bin/activate      # on Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
```

## Run it

```bash
python examples/xor.py
```

The loss goes down while training, and at the end the network predicts `[0 1 1 0]` for the four XOR inputs.

Run the tests:

```bash
python -m pytest -v
```

## Quick example

```python
from nnfs.layers import Dense, Sigmoid
from nnfs.loss import mse, mse_derivative
from nnfs.optimizer import SGD
from nnfs.network import Network

net = Network(
    layers=[Dense(2, 4, seed=1), Sigmoid(), Dense(4, 1, seed=2), Sigmoid()],
    loss=mse,
    loss_derivative=mse_derivative,
    optimizer=SGD(lr=1.0),
)

net.fit(X, y, epochs=10000)
print(net.predict(X))
```

## How it works

Every training step does four things:

1. **Forward pass:** the input goes through each layer (`xW + b`, then an activation).
2. **Loss:** compare the output with the correct answer.
3. **Backward pass:** use the chain rule to find how much each weight contributed to the error.
4. **Update:** nudge every weight a little in the direction that reduces the error (`W = W - lr * dW`).

Repeat this thousands of times and the network gets better.

## Notes

- If XOR training gets stuck at a loss around 0.125, try different `seed` values. Bad random starting weights can trap it.
- Things I might add later: mini-batches, more optimizers (momentum, Adam), a real dataset like MNIST, gradient checking.
