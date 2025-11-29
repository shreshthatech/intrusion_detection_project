"""
Task 29: Uncertainty quantification using Monte Carlo Dropout.

We:
  - wrap the FusionNet (Task 22)
  - enable dropout during inference
  - run N forward passes
  - compute mean, variance, confidence interval
"""
import sys, os

# Determine the path to the project root:
CURRENT_FILE = os.path.abspath(__file__)
FUSION_DIR = os.path.dirname(CURRENT_FILE)             # .../src/fusion
SRC_DIR = os.path.dirname(FUSION_DIR)                  # .../src
PROJECT_ROOT = os.path.dirname(SRC_DIR)                # .../intrusion_detection_project

# Add SRC_DIR to Python path (this is the correct root for imports)
sys.path.append(SRC_DIR)

from fusion.learnable_fusion import FusionNet
import torch
import torch.nn as nn

from typing import Dict, Tuple
from fusion.learnable_fusion import FusionNet


class MCDropoutWrapper(nn.Module):
    def __init__(self, fusion_model: FusionNet, dropout_p: float = 0.3):
        """
        fusion_model: FusionNet instance
        dropout_p: dropout probability (applied on projected features)
        """
        super().__init__()
        self.fusion_model = fusion_model

        # Insert dropout layers AFTER projection stage
        # We modify projectors to include dropout
        for m in fusion_model.projectors.values():
            m.proj.add_module("drop", nn.Dropout(p=dropout_p))

    def forward_pass(self, feats: Dict[str, torch.Tensor]):
        """
        Single forward pass WITH dropout.
        Returns fused_embedding (B, D)
        """
        self.train()  # Enables dropout at inference!
        fused, weights = self.fusion_model(feats)
        return fused

    @torch.no_grad()
    def mc_forward(self, feats: Dict[str, torch.Tensor], n_samples: int = 20):
        """
        Run MC forward passes and compute stats.
        """
        outputs = []

        for _ in range(n_samples):
            y = self.forward_pass(feats)  # (B,D)
            outputs.append(y.unsqueeze(0))

        # (N,B,D) → take mean + variance along N
        outputs = torch.cat(outputs, dim=0)
        mean = outputs.mean(dim=0)
        var = outputs.var(dim=0)

        return mean, var


# ---------------- DEMO ---------------- #

def _demo():
    print("[DEMO] MC Dropout Demo")

    # Example dims
    dims = {
        "video": 128,
        "audio": 64,
        "rf": 32,
        "thermal": 128
    }

    # Fusion model
    fusion = FusionNet(dims=dims, proj_dim=64, fused_dim=128)

    # Enable MC dropout
    mc = MCDropoutWrapper(fusion_model=fusion, dropout_p=0.3)

    # Fake batch of features
    feats = {
        "video": torch.randn(1, 128),
        "audio": torch.randn(1, 64),
        "rf": torch.randn(1, 32),
        "thermal": torch.randn(1, 128),
    }

    mean, var = mc.mc_forward(feats, n_samples=20)

    print("[DEMO] mean shape:", mean.shape)
    print("[DEMO] var shape:", var.shape)
    print("[DEMO] variance (first 5 dims):", var[0, :5])


if __name__ == "__main__":
    _demo()
