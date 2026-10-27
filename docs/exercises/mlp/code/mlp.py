"""MLP written from scratch with NumPy (no autodiff, no ML frameworks).

The same class is used in Exercises 2, 3 and 4. Only *arguments* change:
    - layer_sizes : e.g. [2, 8, 1]  (Ex. 2), [4, 16, 3] (Ex. 3), [4, 16, 16, 3] (Ex. 4)
    - output      : "sigmoid" (binary cross-entropy) or "softmax" (categorical cross-entropy)

Hidden layers use tanh. For both output/loss pairs the gradient of the loss with
respect to the output pre-activation is the same expression, (p - y) / N, so the
forward / backward / update code is shared without any change.
"""
import numpy as np


class MLP:
    def __init__(self, layer_sizes, output="sigmoid", rng=None):
        self.sizes = list(layer_sizes)
        self.output = output
        rng = rng if rng is not None else np.random.default_rng(42)
        self.W, self.b = [], []
        for n_in, n_out in zip(self.sizes[:-1], self.sizes[1:]):
            # Xavier/Glorot initialisation (suited to tanh)
            limit = np.sqrt(6.0 / (n_in + n_out))
            self.W.append(rng.uniform(-limit, limit, size=(n_in, n_out)))
            self.b.append(np.zeros(n_out))

    # ---------------------------------------------------------------- forward
    def forward(self, X):
        """Returns the list of activations [X, h1, ..., p] (p = output probabilities)."""
        acts = [X]
        a = X
        L = len(self.W)
        for l in range(L):
            u = a @ self.W[l] + self.b[l]
            if l < L - 1:
                a = np.tanh(u)                       # hidden: tanh
            elif self.output == "sigmoid":
                a = 1.0 / (1.0 + np.exp(-u))         # binary output
            else:
                u = u - u.max(axis=1, keepdims=True) # numerical stability
                e = np.exp(u)
                a = e / e.sum(axis=1, keepdims=True) # softmax
            acts.append(a)
        return acts

    # ------------------------------------------------------------------- loss
    def loss(self, p, Y):
        """Mean cross-entropy. Y: (N,1) in {0,1} for sigmoid, (N,K) one-hot for softmax."""
        eps = 1e-12
        if self.output == "sigmoid":
            return float(-np.mean(Y * np.log(p + eps) + (1 - Y) * np.log(1 - p + eps)))
        return float(-np.mean(np.sum(Y * np.log(p + eps), axis=1)))

    # --------------------------------------------------------------- backward
    def backward(self, acts, Y):
        """Backpropagation. Returns (dW list, db list)."""
        N = Y.shape[0]
        L = len(self.W)
        dW, db = [None] * L, [None] * L
        # For sigmoid+BCE and softmax+CCE: dL/du_out = (p - y) / N
        delta = (acts[-1] - Y) / N
        for l in reversed(range(L)):
            dW[l] = acts[l].T @ delta
            db[l] = delta.sum(axis=0)
            if l > 0:
                dh = delta @ self.W[l].T                    # dL/dh
                delta = dh * (1.0 - acts[l] ** 2)           # tanh'(z) = 1 - tanh^2
        return dW, db

    # ----------------------------------------------------------------- update
    def update(self, dW, db, lr):
        for l in range(len(self.W)):
            self.W[l] -= lr * dW[l]
            self.b[l] -= lr * db[l]

    # ------------------------------------------------------------------ train
    def fit(self, X, Y, lr=0.1, epochs=1000):
        """Full-batch gradient descent. Returns the training loss of every epoch."""
        history = []
        for _ in range(epochs):
            acts = self.forward(X)
            history.append(self.loss(acts[-1], Y))
            dW, db = self.backward(acts, Y)
            self.update(dW, db, lr)
        return history

    def predict(self, X):
        p = self.forward(X)[-1]
        if self.output == "sigmoid":
            return (p[:, 0] >= 0.5).astype(int)
        return p.argmax(axis=1)

    def n_params(self):
        return sum(w.size + b.size for w, b in zip(self.W, self.b))
