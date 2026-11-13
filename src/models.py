import numpy as np
import torch
from sklearn.base import BaseEstimator, ClassifierMixin
from torch import nn


class TorchMLP(ClassifierMixin, BaseEstimator):
    """Feedforward network with the scikit-learn fit/predict_proba interface."""
    """epochs=20 chosen from the sweep in notebooks/04_calibration_tuning.ipynb"""
    def __init__(self, hidden=(32, 16), lr=1e-3, epochs=20,
                 batch_size=64, seed=42):
        self.hidden = hidden
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.seed = seed

    def fit(self, X, y):
        torch.manual_seed(self.seed)
        X = torch.tensor(np.asarray(X), dtype=torch.float32)
        y = torch.tensor(np.asarray(y), dtype=torch.float32).unsqueeze(1)
        layers, d = [], X.shape[1]
        for h in self.hidden:
            layers += [nn.Linear(d, h), nn.ReLU()]
            d = h
        layers.append(nn.Linear(d, 1))
        self.net_ = nn.Sequential(*layers)
        opt = torch.optim.Adam(self.net_.parameters(), lr=self.lr)
        loss_fn = nn.BCEWithLogitsLoss()
        n = len(X)
        for _ in range(self.epochs):
            perm = torch.randperm(n)
            for i in range(0, n, self.batch_size):
                idx = perm[i:i + self.batch_size]
                opt.zero_grad()
                loss_fn(self.net_(X[idx]), y[idx]).backward()
                opt.step()
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X):
        X = torch.tensor(np.asarray(X), dtype=torch.float32)
        with torch.no_grad():
            p = torch.sigmoid(self.net_(X)).squeeze(1).numpy()
        return np.column_stack([1 - p, p])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)