"""Deep-learning component: a denoising autoencoder for tabular data.

It is trained on the (scaled) numeric columns and learns what a "normal" row looks like.
Rows / cells it cannot reconstruct well are anomalies.  Uses PyTorch when installed and falls
back to a scikit-learn neural network (MLPRegressor) so the app still runs without torch.
"""
from __future__ import annotations

import numpy as np

try:  # PyTorch is optional at import time
    import torch
    import torch.nn as nn

    TORCH_OK = True
except Exception:  # pragma: no cover
    torch = None
    nn = None
    TORCH_OK = False


class TabularAutoencoder:
    def __init__(self, epochs: int = 60, noise: float = 0.05, seed: int = 42, max_train_rows: int = 20000):
        self.epochs = epochs
        self.noise = noise
        self.seed = seed
        self.max_train_rows = max_train_rows
        self.backend = "pytorch" if TORCH_OK else "sklearn-mlp"
        self.history: list[float] = []
        self._net = None

    @staticmethod
    def _sizes(d: int) -> tuple[int, int]:
        hidden = max(8, min(64, d * 2))
        bottleneck = max(1, d // 2)
        return hidden, bottleneck

    def fit(self, X: np.ndarray) -> "TabularAutoencoder":
        X = np.asarray(X, dtype=np.float32)
        rng = np.random.default_rng(self.seed)
        if len(X) > self.max_train_rows:
            X = X[rng.choice(len(X), self.max_train_rows, replace=False)]
        n, d = X.shape
        hidden, bottleneck = self._sizes(d)

        if TORCH_OK:
            torch.manual_seed(self.seed)
            net = nn.Sequential(
                nn.Linear(d, hidden), nn.ReLU(),
                nn.Linear(hidden, bottleneck), nn.ReLU(),
                nn.Linear(bottleneck, hidden), nn.ReLU(),
                nn.Linear(hidden, d),
            )
            opt = torch.optim.Adam(net.parameters(), lr=3e-3, weight_decay=1e-5)
            loss_fn = nn.MSELoss()
            data = torch.from_numpy(X)
            bs = 256
            for _ in range(self.epochs):
                net.train()
                perm = torch.randperm(n)
                total = 0.0
                for i in range(0, n, bs):
                    xb = data[perm[i:i + bs]]
                    noisy = xb + self.noise * torch.randn_like(xb)  # denoising objective
                    opt.zero_grad()
                    loss = loss_fn(net(noisy), xb)
                    loss.backward()
                    opt.step()
                    total += loss.item() * len(xb)
                self.history.append(total / n)
            net.eval()
            self._net = net
        else:
            from sklearn.neural_network import MLPRegressor

            net = MLPRegressor(hidden_layer_sizes=(hidden, bottleneck, hidden), activation="relu",
                               max_iter=300, learning_rate_init=3e-3, random_state=self.seed,
                               early_stopping=False)
            noisy = X + self.noise * rng.standard_normal(X.shape).astype(np.float32)
            net.fit(noisy, X)
            self.history = [float(v) for v in net.loss_curve_]
            self._net = net
        return self

    def reconstruct(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float32)
        if TORCH_OK:
            with torch.no_grad():
                return self._net(torch.from_numpy(X)).numpy().astype(np.float64)
        return np.asarray(self._net.predict(X), dtype=np.float64)

    def loss_curve(self, points: int = 30) -> list[float]:
        h = self.history
        if len(h) <= points:
            return [float(v) for v in h]
        idx = np.linspace(0, len(h) - 1, points).astype(int)
        return [float(h[i]) for i in idx]
