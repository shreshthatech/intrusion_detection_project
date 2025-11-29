"""
Task 45: Shallow CNN to detect suspicious RF beacons/jammers.

We:
  - Generate synthetic RF signals (normal vs abnormal).
  - Compute log-power STFT waterfalls.
  - Train a small 2D CNN to classify:
        0 = normal
        1 = suspicious (jammer-like)
"""

from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.signal as spsig
import random


PROJECT_ROOT = Path(__file__).resolve().parents[2]


# ------------ Synthetic RF + STFT (reuse idea from Task 44) ------------ #

def gen_rf_signal(T=16000, mode="normal"):
    t = np.linspace(0, 1.0, T)
    x = 0.2 * np.random.randn(T)
    x += 0.05 * np.sin(2 * np.pi * 500 * t)
    x += 0.05 * np.sin(2 * np.pi * 900 * t)

    if mode == "abnormal":
        for _ in range(4):
            start = np.random.randint(0, T - 800)
            x[start:start+800] += np.random.uniform(2.0, 3.0) * np.random.randn(800)

    return x.astype(np.float32)


def compute_rf_waterfall(signal, n_fft=256, hop=128):
    f, t, Zxx = spsig.stft(signal, nperseg=n_fft, noverlap=n_fft-hop)
    power = np.abs(Zxx) ** 2
    log_power = np.log(power + 1e-6)
    return log_power  # shape: (freq_bins, time_frames)


def build_rf_dataset(N=200, n_fft=256, hop=128, device="cpu"):
    X = []
    y = []

    for _ in range(N):
        mode = "normal" if random.random() < 0.5 else "abnormal"
        sig = gen_rf_signal(mode=mode)
        W = compute_rf_waterfall(sig, n_fft=n_fft, hop=hop)
        # normalize
        W = (W - W.mean()) / (W.std() + 1e-6)
        # add channel dim
        X.append(W[None, :, :])  # (1, F, T)
        y.append(0 if mode == "normal" else 1)

    X = np.stack(X, axis=0)  # (N, 1, F, T)
    X = torch.tensor(X, dtype=torch.float32, device=device)
    y = torch.tensor(y, dtype=torch.long, device=device)

    return X, y


# ------------ CNN Model ------------ #

class RFCNN(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.fc = nn.Linear(64 * 4 * 4, num_classes)

    def forward(self, x):
        feat = self.conv(x)            # (B,64,4,4)
        feat = feat.view(feat.size(0), -1)
        logits = self.fc(feat)
        return logits


# ------------ DEMO TRAINING ------------ #

def _demo():
    print("[DEMO] RF CNN Classifier (Task 45)")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[INFO] Using device:", device)

    X, y = build_rf_dataset(N=240, device=device)
    n_train = 200
    X_train, y_train = X[:n_train], y[:n_train]
    X_val, y_val = X[n_train:], y[n_train:]

    model = RFCNN(num_classes=2).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(1, 9):
        model.train()
        optimizer.zero_grad()
        logits = model(X_train)
        loss = criterion(logits, y_train)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_logits = model(X_val)
            val_loss = criterion(val_logits, y_val)
            preds = val_logits.argmax(dim=1)
            acc = (preds == y_val).float().mean().item()

        print(f"[EPOCH {epoch}] train_loss={loss.item():.4f}  val_loss={val_loss.item():.4f}  val_acc={acc*100:.1f}%")

    # Test on explicit normal vs abnormal
    print("\n[DEMO] Testing on explicit examples")
    test_signals = []
    labels = []
    for mode in ["normal", "abnormal"]:
        sig = gen_rf_signal(mode=mode)
        W = compute_rf_waterfall(sig)
        W = (W - W.mean()) / (W.std() + 1e-6)
        test_signals.append(W[None, :, :])
        labels.append(mode)

    test = np.stack(test_signals, axis=0)
    test = torch.tensor(test, dtype=torch.float32, device=device)

    with torch.no_grad():
        out = model(test)
        pred = out.argmax(dim=1).cpu().tolist()

    for i, p in enumerate(pred):
        pred_label = "normal" if p == 0 else "abnormal"
        print(f"  True={labels[i]}  Pred={pred_label}")

    print("\n[DEMO] Done.")


if __name__ == "__main__":
    _demo()
