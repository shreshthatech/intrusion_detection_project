"""
Task 64: Operator Fusion (Conv + BN + ReLU)

We demonstrate:
    - A model with Conv2d + BatchNorm2d + ReLU
    - Fuse them into a single Conv+ReLU (or Conv only) operator
    - Export fused model

This is EXACTLY what all real deployment toolchains do.
"""

import torch
import torch.nn as nn
from pathlib import Path


# ----------------------------- Model with BN ----------------------------- #

class ConvBNReLU(nn.Module):
    def __init__(self):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.seq(x)


def fuse_model(model):
    """
    Applies PyTorch official operator fusion.
    Conv2d + BatchNorm2d + ReLU → fused
    """
    print("[INFO] Fusing Conv + BN + ReLU …")

    # Fuse in-place
    torch.quantization.fuse_modules(
        model.seq,
        [["0", "1", "2"]],    # indices inside Sequential
        inplace=True
    )

    print("[SUCCESS] Layers fused.")
    return model


def main():
    print("[INFO] Task 64 – Operator Fusion")

    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    SAVE_PATH = PROJECT_ROOT / "models" / "vision" / "conv_bn_relu_fused.pt"

    model = ConvBNReLU().eval()

    # Fake weights initialization
    for p in model.parameters():
        nn.init.normal_(p, mean=0, std=0.02)

    # Fuse operators
    fused = fuse_model(model)

    # Save fused model
    torch.save(fused.state_dict(), SAVE_PATH)
    print(f"[SUCCESS] Saved fused model to: {SAVE_PATH}")

    # Test run
    x = torch.randn(1, 3, 224, 224)
    y = fused(x)
    print(f"[INFO] Output shape: {y.shape}")

    print("[DEMO] Task 64 completed.")


if __name__ == "__main__":
    main()
