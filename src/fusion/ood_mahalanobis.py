"""
Task 25: OOD Detection via Mahalanobis Distance.

This module learns:
  - mean vector
  - covariance
  - inverse covariance

Then provides .score(x) = Mahalanobis distance.

Usage:
  ood = MahalanobisOOD()
  ood.fit(train_features)     # NxD normal data
  d = ood.score(x)            # Out-of-distribution score
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import json


class MahalanobisOOD(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("mean", None)
        self.register_buffer("cov_inv", None)
        self.dim = None

    def fit(self, feats: np.ndarray):
        """
        feats: NxD array of normal behavior features
        """
        if feats.ndim != 2:
            raise ValueError("Expected feats shape (N, D)")

        N, D = feats.shape
        self.dim = D

        mu = feats.mean(axis=0)
        cov = np.cov(feats, rowvar=False)

        # Regularize covariance (important!)
        cov += 1e-6 * np.eye(D)

        cov_inv = np.linalg.inv(cov)

        # Save as tensors
        self.mean = torch.tensor(mu, dtype=torch.float32)
        self.cov_inv = torch.tensor(cov_inv, dtype=torch.float32)

    def score(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (D,) or (B,D)
        Returns OOD distance
        """
        if x.ndim == 1:
            x = x.unsqueeze(0)

        diff = x - self.mean
        left = torch.matmul(diff, self.cov_inv)
        dist = torch.sqrt(torch.sum(left * diff, dim=1) + 1e-6)
        return dist  # (B,)


# ---------------- DEMO ---------------- #

def _demo():
    print("[DEMO] Running Mahalanobis OOD demo...")

    # Generate synthetic "normal" data
    train = np.random.normal(0, 1, (500, 16))

    # Fit model
    ood = MahalanobisOOD()
    ood.fit(train)

    # Test normal point
    normal_x = torch.tensor(train[0], dtype=torch.float32)
    d1 = ood.score(normal_x)
    print("[DEMO] Normal score:", float(d1))

    # Test abnormal point
    abnormal_x = torch.tensor(np.random.normal(8, 1, 16), dtype=torch.float32)
    d2 = ood.score(abnormal_x)
    print("[DEMO] Abnormal score:", float(d2))


if __name__ == "__main__":
    _demo()
