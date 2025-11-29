"""
Task 35: LSTM on detection/track sequences for intent prediction.

We build a small LSTM classifier that:
  - Input: per-frame features for a track (sequence)
  - Output: intent class:
        0 = normal_pass_through
        1 = approaching_restricted
        2 = loitering

For now, we generate synthetic sequences with simple rules.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import random


INTENT_LABELS = {
    0: "normal_pass_through",
    1: "approaching_restricted",
    2: "loitering",
}


class IntentLSTM(nn.Module):
    def __init__(self, input_dim=4, hidden_dim=64, num_layers=1, num_classes=3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
        )
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        """
        x: (B, T, D)
        """
        out, (h_n, c_n) = self.lstm(x)  # h_n: (num_layers, B, H)
        h_last = h_n[-1]                # (B, H)
        logits = self.fc(h_last)        # (B, num_classes)
        return logits


# -------------- Synthetic Data Generator -------------- #

def generate_sequence(seq_len=20):
    """
    Generate a synthetic track with simple (x, y, speed, dist_to_gate) features.

    We define:
      - restricted gate at x=1.0, y=0.5
      - positions in [0,1] range.

    Rules:
      1) approaching_restricted:
         - x increases toward 1.0
         - dist_to_gate becomes small
         - speed moderate

      2) loitering:
         - stays around some point (almost no movement)
         - speed very low
         - dist_to_gate medium/high

      3) normal_pass_through:
         - moves across scene but not strongly toward gate
         - dist_to_gate not very small at end
    """
    gate_x, gate_y = 1.0, 0.5

    intent_type = random.choice([0, 1, 2])  # 0=normal,1=approach,2=loiter

    xs, ys = [], []

    if intent_type == 1:
        # approaching restricted: start left, move right toward gate
        x = random.uniform(0.0, 0.3)
        y = gate_y + random.uniform(-0.1, 0.1)
        for t in range(seq_len):
            x += random.uniform(0.02, 0.06)
            x = min(x, 1.0)
            y += random.uniform(-0.01, 0.01)
            xs.append(x)
            ys.append(y)
    elif intent_type == 2:
        # loitering: stay near some random point, small jitter
        x = random.uniform(0.3, 0.7)
        y = random.uniform(0.2, 0.8)
        for t in range(seq_len):
            x += random.uniform(-0.005, 0.005)
            y += random.uniform(-0.005, 0.005)
            xs.append(x)
            ys.append(y)
    else:
        # normal_pass_through: move diagonally, not directly to gate
        x = random.uniform(0.0, 0.3)
        y = random.uniform(0.0, 0.3)
        dx = random.uniform(0.01, 0.04)
        dy = random.uniform(0.01, 0.04)
        for t in range(seq_len):
            x += dx + random.uniform(-0.01, 0.01)
            y += dy + random.uniform(-0.01, 0.01)
            x = max(0.0, min(x, 1.0))
            y = max(0.0, min(y, 1.0))
            xs.append(x)
            ys.append(y)

    # Compute speed and dist_to_gate
    feats = []
    for i in range(seq_len):
        x = xs[i]
        y = ys[i]

        if i == 0:
            vx = 0.0
            vy = 0.0
        else:
            vx = xs[i] - xs[i - 1]
            vy = ys[i] - ys[i - 1]

        speed = (vx**2 + vy**2) ** 0.5
        dist_gate = ((x - gate_x) ** 2 + (y - gate_y) ** 2) ** 0.5

        feats.append([x, y, speed, dist_gate])

    # Heuristic label adjustment to keep things more consistent
    avg_speed = sum(f[2] for f in feats) / seq_len
    final_dist = feats[-1][3]

    if avg_speed < 0.01:
        label = 2  # loitering
    elif final_dist < 0.2:
        label = 1  # approaching_restricted
    else:
        label = intent_type if intent_type == 0 else 0  # fallback to normal for ambiguous

    return feats, label


def build_dataset(num_sequences=300, seq_len=20, device="cpu"):
    X = []
    y = []
    for _ in range(num_sequences):
        feats, label = generate_sequence(seq_len)
        X.append(feats)
        y.append(label)

    X = torch.tensor(X, dtype=torch.float32, device=device)  # (N,T,D)
    y = torch.tensor(y, dtype=torch.long, device=device)     # (N,)
    return X, y


# -------------- DEMO TRAIN LOOP -------------- #

def _demo():
    print("[DEMO] IntentLSTM (Task 35)")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[INFO] Using device:", device)

    seq_len = 20
    input_dim = 4
    num_classes = 3

    model = IntentLSTM(input_dim=input_dim, hidden_dim=64, num_layers=1, num_classes=num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    # Build dataset
    X, y = build_dataset(num_sequences=400, seq_len=seq_len, device=device)
    print("[INFO] Dataset:", X.shape, y.shape)

    # Simple train/val split
    n_train = int(0.8 * X.size(0))
    X_train, y_train = X[:n_train], y[:n_train]
    X_val, y_val = X[n_train:], y[n_train:]

    for epoch in range(1, 11):
        model.train()
        optimizer.zero_grad()

        logits = model(X_train)   # (N, num_classes)
        loss = criterion(logits, y_train)
        loss.backward()
        optimizer.step()

        # Validation
        model.eval()
        with torch.no_grad():
            val_logits = model(X_val)
            val_loss = criterion(val_logits, y_val)
            preds = val_logits.argmax(dim=1)
            acc = (preds == y_val).float().mean().item()

        print(f"[EPOCH {epoch}] train_loss={loss.item():.4f}  val_loss={val_loss.item():.4f}  val_acc={acc*100:.1f}%")

    # Test on a few hand-crafted sequences
    print("\n[DEMO] Testing on example sequences:")

    # Approaching gate
    feats_approach, _ = generate_sequence(seq_len)
    # Loitering
    feats_loiter, _ = generate_sequence(seq_len)
    # Normal pass
    feats_normal, _ = generate_sequence(seq_len)

    samples = torch.tensor([feats_approach, feats_loiter, feats_normal], dtype=torch.float32, device=device)
    with torch.no_grad():
        out = model(samples)
        pred = out.argmax(dim=1).cpu().tolist()

    for i, p in enumerate(pred):
        print(f"  Sample {i}: predicted={INTENT_LABELS[p]} (class {p})")

    print("\n[DEMO] Done.")


if __name__ == "__main__":
    _demo()
