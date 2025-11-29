"""
Task 23: Sensor Reliability Weighting
-------------------------------------

We learn a module that estimates reliability weights for each modality
based on simple metadata such as:

  - temperature
  - SNR
  - brightness
  - motion blur level
  - RF noise floor
  - audio amplitude / clipping
  - thermal camera gain

This feeds into FusionNet from Task 22.
"""

import torch
import torch.nn as nn
from typing import Dict, Tuple


class ReliabilityNet(nn.Module):
    """
    Learnable reliability estimator.
    Input:
        reliability_features: dict(sensor -> tensor(batch, feature_dim))

        Example structure:
        {
            "video":   [brightness, blur_level],
            "audio":   [snr, noise],
            "rf":      [snr, noise_floor],
            "thermal": [temperature, variance]
        }

    Output:
        weights: dict(sensor -> scalar weight in [0, 1])
    """

    def __init__(self, dims: Dict[str, int], hidden: int = 32):
        """
        dims: dict mapping modality -> reliability feature dimension.
              Example: {"video":2, "audio":2, "rf":2, "thermal":2}
        """
        super().__init__()

        self.modalities = list(dims.keys())

        self.nets = nn.ModuleDict()
        for m, dim_in in dims.items():
            self.nets[m] = nn.Sequential(
                nn.Linear(dim_in, hidden),
                nn.ReLU(inplace=True),
                nn.Linear(hidden, 1),
                nn.Sigmoid()    # output is in [0, 1]
            )

    def forward(self, rel_feats: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        rel_feats: dict(sensor -> Tensor(batch, dim))
        Returns:
            weights: dict(sensor -> Tensor(batch, 1))
        """
        out = {}
        for m in self.modalities:
            out[m] = self.nets[m](rel_feats[m])  # (B,1)
        return out


# ---------------- DEMO ---------------- #

def _demo():
    print("[DEMO] Running ReliabilityNet demo...")

    dims = {
        "video": 2,    # brightness, blur
        "audio": 2,    # snr, noise
        "rf": 2,       # snr, noise_floor
        "thermal": 2   # temperature, variance
    }

    net = ReliabilityNet(dims)

    # Fake reliability indicators
    feats = {
        "video": torch.tensor([[0.6, 0.2]]),
        "audio": torch.tensor([[0.7, 0.3]]),
        "rf": torch.tensor([[0.4, 0.8]]),
        "thermal": torch.tensor([[0.9, 0.1]]),
    }

    weights = net(feats)
    print("[DEMO] Reliability weights:")
    for m, w in weights.items():
        print(f"  {m}: {float(w):.3f}")


if __name__ == "__main__":
    _demo()
