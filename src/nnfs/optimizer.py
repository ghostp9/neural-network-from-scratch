class SGD:
    def __init__(self, lr=0.1):
        self.lr = lr

    def step(self, layers):
        for layer in layers:
            if hasattr(layer, "W"):
                layer.W -= self.lr * layer.dW
                layer.b -= self.lr * layer.db
