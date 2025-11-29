"""
Task 43: Acoustic anomaly detection via autoencoder reconstruction error.

We:
  - Generate synthetic "normal" MFCC sequences (mostly one pattern).
  - Train a small 1D autoencoder to reconstruct them.
  - Generate "abnormal" sequences with different patterns/noise.
  - Compute reconstruction error and see abnormal > normal.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random


class AudioAutoencoder(nn.Module):
    def __init__(self, n_mfcc=13):
        super().__init__()

        # Encoder: (B, C, T) -> compressed latent
        self.encoder = nn.Sequential(
            nn.Conv1d(n_mfcc, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2),  # T/2

            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2),  # T/4
        )

        # Decoder: mirror the encoder (upsample back to T)
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(64, 32, kernel_size=4, stride=2, padding=1),  # T/2
            nn.ReLU(),
            nn.ConvTranspose1d(32, n_mfcc, kernel_size=4, stride=2, padding=1),  # T
        )

    def forward(self, x):
        z = self.encoder(x)
        out = self.decoder(z)
        return out


# ------------ Synthetic data generators (reuse idea from Task 42) ------------ #

def gen_normal_mfcc(T=200, n_mfcc=13):
    """
    Normal sounds (e.g. background hum, AC noise, mild wind):
    smooth-ish, low amplitude patterns.
    """
    base = np.linspace(0, 4, T)
    mfcc = np.array([
        0.3 * np.sin(base + np.random.uniform(0, 2*np.pi)) +
        0.05 * np.random.randn(T)
        for _ in range(n_mfcc)
    ], dtype=np.float32)
    return mfcc


def gen_abnormal_mfcc(T=200, n_mfcc=13):
    """
    Abnormal sounds (e.g. sudden loud bangs, sirens):
    spikes, high variance, abrupt changes.
    """
    mfcc = 0.5 * np.random.randn(n_mfcc, T).astype(np.float32)

    # Add a few strong spikes
    for _ in range(5):
        t0 = random.randint(0, T - 5)
        c = random.randint(0, n_mfcc - 1)
        mfcc[c, t0:t0+5] += np.random.uniform(2.0, 3.0)

    return mfcc


def build_normal_dataset(N=300, n_mfcc=13, T=200, device="cpu"):
    X = []
    for _ in range(N):
        X.append(gen_normal_mfcc(T=T, n_mfcc=n_mfcc))
    X = torch.tensor(X, dtype=torch.float32, device=device)  # (N, C, T)
    return X


# ---------------- DEMO TRAIN LOOP ---------------- #

def _demo():
    print("[DEMO] Audio Autoencoder (Task 43)")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[INFO] Using device:", device)

    n_mfcc = 13
    T = 200

    # Build "normal" training set
    X_train = build_normal_dataset(N=400, n_mfcc=n_mfcc, T=T, device=device)

    model = AudioAutoencoder(n_mfcc=n_mfcc).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    # Train for a few epochs
    for epoch in range(1, 9):
        model.train()
        optimizer.zero_grad()

        recon = model(X_train)
        loss = criterion(recon, X_train)
        loss.backward()
        optimizer.step()

        print(f"[EPOCH {epoch}] train_recon_loss={loss.item():.6f}")

    # ---- Evaluate on normal vs abnormal samples ---- #
    model.eval()
    with torch.no_grad():
        # Normal samples
        normals = build_normal_dataset(N=20, n_mfcc=n_mfcc, T=T, device=device)
        recon_normals = model(normals)
        err_normals = F.mse_loss(recon_normals, normals, reduction="none")
        err_normals = err_normals.mean(dim=(1, 2)).cpu().numpy()  # per-sample

        # Abnormal samples
        abnormals = []
        for _ in range(20):
            abnormals.append(gen_abnormal_mfcc(T=T, n_mfcc=n_mfcc))
        abnormals = torch.tensor(abnormals, dtype=torch.float32, device=device)
        recon_ab = model(abnormals)
        err_ab = F.mse_loss(recon_ab, abnormals, reduction="none")
        err_ab = err_ab.mean(dim=(1, 2)).cpu().numpy()

    print("\n[RESULTS] Reconstruction error:")
    print(f"  Normal   mean={err_normals.mean():.6f}, std={err_normals.std():.6f}")
    print(f"  Abnormal mean={err_ab.mean():.6f}, std={err_ab.std():.6f}")

    # Show a few individual errors
    print("\n[EXAMPLES]")
    for i in range(5):
        print(f"  normal[{i}] err={err_normals[i]:.6f}")
    for i in range(5):
        print(f"  abnormal[{i}] err={err_ab[i]:.6f}")

    print("\n[DEMO] Done.")


if __name__ == "__main__":
    _demo()
