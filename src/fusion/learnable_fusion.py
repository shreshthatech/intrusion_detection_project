"""
Task 22: Learnable feature fusion across modalities (concat + attention).

We build a small PyTorch module:
  - Input: dict of modality tensors {"video": v, "audio": a, "rf": r, "thermal": th}
  - Fusion:
        step 1: individual linear projections
        step 2: concat -> joint vector
        step 3: attention network produces modality weights
        step 4: weighted sum -> fused embedding
  - Output:
        fused embedding (tensor), modality weights (dict)

This module will later plug into:
  - Task 23: reliability weighting
  - Task 24: threshold calibration
  - Task 25: OOD detection
"""

import torch
import torch.nn as nn
from typing import Dict, Tuple


class ModalityProjector(nn.Module):
    """
    Projects each modality's raw feature vector x to a common dimension D.
    """
    def __init__(self, input_dim: int, proj_dim: int):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(input_dim, proj_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.proj(x)


class FusionNet(nn.Module):
    """
    Learnable early-late hybrid fusion.

    Steps:
      1. Project each modality to proj_dim.
      2. Concatenate projected vectors -> joint_vec.
      3. Compute attention weights over modalities.
      4. weighted_sum = sum(weight[m] * proj_vec[m]).
      5. Optionally pass through final MLP -> fused embedding.
    """

    def __init__(self,
                 dims: Dict[str, int],
                 proj_dim: int = 64,
                 fused_dim: int = 128):
        """
        dims: dict mapping modality -> input_dim
              Example: {"video":128, "audio":64, "rf":32, "thermal":128}
        proj_dim: dimension of each projected modality
        fused_dim: dimension of final fused embedding
        """
        super().__init__()

        self.modalities = list(dims.keys())

        # Projectors
        self.projectors = nn.ModuleDict()
        for m, in_dim in dims.items():
            self.projectors[m] = ModalityProjector(in_dim, proj_dim)

        # Attention network
        self.att_mlp = nn.Sequential(
            nn.Linear(len(dims) * proj_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, len(dims))
        )

        # Final fusion MLP
        self.final_mlp = nn.Sequential(
            nn.Linear(proj_dim, fused_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, feats: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        feats: dict of modality -> tensor of shape (batch, dim)
        """
        batch = next(iter(feats.values())).shape[0]

        # 1. Project each modality
        proj = {}
        for m in self.modalities:
            proj[m] = self.projectors[m](feats[m])  # shape (B, proj_dim)

        # 2. Concat all projected feats
        concat = torch.cat([proj[m] for m in self.modalities], dim=1)  # (B, M*proj_dim)

        # 3. Attention weights
        att_logits = self.att_mlp(concat)              # (B, M)
        att_weights = torch.softmax(att_logits, dim=1) # (B, M)

        # Convert to dict for debugging
        weight_dict = {
            m: att_weights[:, i].mean().item()
            for i, m in enumerate(self.modalities)
        }

        # 4. Weighted sum of projectors
        fused = 0
        for i, m in enumerate(self.modalities):
            w = att_weights[:, i].view(batch, 1)       # (B,1)
            fused = fused + w * proj[m]                # (B, proj_dim)

        # 5. Final MLP
        fused_final = self.final_mlp(fused)            # (B, fused_dim)

        return fused_final, weight_dict


# ---------------- DEMO ---------------- #

def _demo():
    print("[DEMO] Running FusionNet demo...")

    dims = {"video": 128, "audio": 64, "rf": 32, "thermal": 128}

    model = FusionNet(dims=dims, proj_dim=64, fused_dim=128)

    # fake random feature batch
    feats = {
        "video": torch.randn(4, 128),
        "audio": torch.randn(4, 64),
        "rf": torch.randn(4, 32),
        "thermal": torch.randn(4, 128),
    }

    fused, w = model(feats)

    print("[DEMO] Fused embedding shape:", fused.shape)
    print("[DEMO] Attention weights:", w)


if __name__ == "__main__":
    _demo()
