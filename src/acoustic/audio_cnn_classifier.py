"""
Task 42: Tiny 1D-CNN to classify:
    - vehicle
    - footstep
    - wind
    - other

We simulate MFCC-like feature sequences and train a CNN.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random


LABELS = ["vehicle", "footstep", "wind", "other"]


# -----------------------------------------------------
# Tiny 1D CNN model
# -----------------------------------------------------

class AudioCNN(nn.Module):
    def __init__(self, n_mfcc=13, num_classes=4):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv1d(n_mfcc, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(32, 64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(64, 128, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )

        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        # x: (B, C, T)
        feat = self.net(x)      # (B, 128, 1)
        feat = feat.squeeze(-1) # (B, 128)
        logits = self.fc(feat)  # (B, num_classes)
        return logits


# -----------------------------------------------------
# Synthetic dataset generator
# -----------------------------------------------------

def generate_fake_mfcc(class_id, T=200, n_mfcc=13):
    """
    Generates simple MFCC-like patterns:
      - vehicle: low + steady
      - footstep: periodic peaks
      - wind: noise-like
      - other: random
    """

    if class_id == 0:
        # vehicle: steady low freq noise
        base = np.linspace(0, 1, T)
        mfcc = np.array([0.3 * np.sin(base * 6) + 0.1 * np.random.randn(T)
                         for _ in range(n_mfcc)])

    elif class_id == 1:
        # footstep: periodic impulses
        mfcc = np.zeros((n_mfcc, T))
        step_positions = np.arange(0, T, 25)
        for p in step_positions:
            for c in range(n_mfcc):
                if p < T:
                    mfcc[c, p] = 2.0 + 0.2 * np.random.randn()

        mfcc += 0.05 * np.random.randn(n_mfcc, T)

    elif class_id == 2:
        # wind: noisy with slow drift
        drift = np.cumsum(0.005 * np.random.randn(T))
        mfcc = drift + 0.15 * np.random.randn(n_mfcc, T)

    else:
        # other: pure random
        mfcc = 0.2 * np.random.randn(n_mfcc, T)

    return mfcc.astype(np.float32)


def build_dataset(N=400, n_mfcc=13, T=200, device="cpu"):
    X = []
    y = []
    for _ in range(N):
        cls = random.randint(0, 3)
        sample = generate_fake_mfcc(cls, T=T, n_mfcc=n_mfcc)
        X.append(sample)
        y.append(cls)

    X = torch.tensor(X, dtype=torch.float32, device=device)  # (N, C, T)
    y = torch.tensor(y, dtype=torch.long, device=device)

    return X, y


# -----------------------------------------------------
# DEMO Training Loop
# -----------------------------------------------------

def _demo():
    print("[DEMO] Tiny Audio CNN (Task 42)")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[INFO] Using device:", device)

    n_mfcc = 13
    T = 200
    num_classes = 4

    # Build dataset
    X, y = build_dataset(N=400, n_mfcc=n_mfcc, T=T, device=device)
    n_train = 320
    X_train, y_train = X[:n_train], y[:n_train]
    X_val, y_val = X[n_train:], y[n_train:]

    model = AudioCNN(n_mfcc=n_mfcc, num_classes=num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    # Train 8 epochs
    for epoch in range(1, 9):
        model.train()
        optimizer.zero_grad()
        logits = model(X_train)
        loss = criterion(logits, y_train)
        loss.backward()
        optimizer.step()

        # val
        model.eval()
        with torch.no_grad():
            val_logits = model(X_val)
            val_loss = criterion(val_logits, y_val)
            preds = val_logits.argmax(dim=1)
            acc = (preds == y_val).float().mean().item()

        print(f"[EPOCH {epoch}] train_loss={loss.item():.4f}  val_loss={val_loss.item():.4f}  val_acc={acc*100:.1f}%")

    # Test on 4 samples
    print("\n[DEMO] Testing…")

    test_samples = []
    for cid in range(4):
        sample = generate_fake_mfcc(cid, T=T, n_mfcc=n_mfcc)
        test_samples.append(sample)

    test = torch.tensor(test_samples, dtype=torch.float32, device=device)
    with torch.no_grad():
        out = model(test)
        pred = out.argmax(dim=1).cpu().tolist()

    for i, p in enumerate(pred):
        print(f"  True={LABELS[i]}  Pred={LABELS[p]}")

    print("\n[DEMO] Done.")


if __name__ == "__main__":
    _demo()
