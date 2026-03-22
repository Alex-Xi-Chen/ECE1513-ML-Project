"""
Model definitions: Linear Regression, SVR, and MLP (PyTorch).
"""

import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR

import config


# ──────────────────────────────────────────────
# Scikit-learn models
# ──────────────────────────────────────────────

def build_linear_regression() -> LinearRegression:
    return LinearRegression()


def build_svr() -> SVR:
    return SVR(**config.SVR_PARAMS)


# ──────────────────────────────────────────────
# PyTorch MLP
# ──────────────────────────────────────────────

class MLP(nn.Module):
    """Multi-Layer Perceptron for regression."""

    def __init__(
        self,
        input_dim: int,
        hidden_sizes: list[int] | None = None,
        dropout: float = 0.2,
    ):
        super().__init__()
        if hidden_sizes is None:
            hidden_sizes = config.MLP_PARAMS["hidden_sizes"]

        layers: list[nn.Module] = []
        prev_dim = input_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.BatchNorm1d(h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = h
        layers.append(nn.Linear(prev_dim, 1))

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x).squeeze(-1)


class ExchangeRateDataset(torch.utils.data.Dataset):
    """Simple dataset wrapper for numpy arrays."""

    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int):
        return self.X[idx], self.y[idx]
