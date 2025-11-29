"""
Task 33: Re-ID embedding (tiny CNN) for cross-camera continuity.

This defines:
  - TinyReID: small CNN that maps a person crop to a 64-D embedding.
  - Embeddings are L2-normalized so cosine similarity / dot product is meaningful.

Demo:
  - Runs forward on random "fake" images.
  - Prints distances between embeddings to show how it works.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TinyReID(nn.Module):
    def __init__(self, embedding_dim: int = 64):
        super().__init__()

        # Input assumed: (B,3,H,W), e.g. 3x128x64 person crop
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),   # H/2, W/2

            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),   # H/4, W/4

            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),   # H/8, W/8
        )

        # After 3 pools, if input is (3,128,64) → spatial becomes (128, 16, 8)
        # 128 * 16 * 8 = 16384 features.
        self.fc = nn.Sequential(
            nn.Linear(128 * 16 * 8, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, embedding_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B,3,H,W)
        returns L2-normalized embeddings: (B, embedding_dim)
        """
        feat = self.features(x)
        feat = feat.view(feat.size(0), -1)
        emb = self.fc(feat)
        emb = F.normalize(emb, p=2, dim=1)   # L2 normalize
        return emb


# -------------- DEMO -------------- #

def _demo():
    print("[DEMO] TinyReID demo")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TinyReID(embedding_dim=64).to(device)

    # Fake person crops: batch of 4 images (3x128x64)
    imgs = torch.randn(4, 3, 128, 64, device=device)

    with torch.no_grad():
        embs = model(imgs)  # (4,64)

    print("[DEMO] Embeddings shape:", embs.shape)

    # Compute pairwise distances between first two
    d01 = torch.dist(embs[0], embs[1]).item()
    d02 = torch.dist(embs[0], embs[2]).item()
    d03 = torch.dist(embs[0], embs[3]).item()

    print(f"[DEMO] dist(0,1) = {d01:.4f}")
    print(f"[DEMO] dist(0,2) = {d02:.4f}")
    print(f"[DEMO] dist(0,3) = {d03:.4f}")
    print("[DEMO] Smaller distance = more similar identity (after training).")


if __name__ == "__main__":
    _demo()
