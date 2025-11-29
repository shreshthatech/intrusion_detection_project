"""
Task 16 (part 2): Early-fusion CNN using RGB + thermal channels.

Input: (2, H, W)  -> [gray, thermal]
Output: (1, H, W) motion mask

Uses data from:
  data/processed/motion_masks/fusion_motion_dataset.npz
"""

from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_FILE = PROJECT_ROOT / "data" / "processed" / "motion_masks" / "fusion_motion_dataset.npz"

MODEL_DIR = PROJECT_ROOT / "models" / "vision"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
MODEL_PATH = MODEL_DIR / "fusion_motion_cnn.pt"


class FusionMotionDataset(Dataset):
    def __init__(self, npz_path: Path):
        data = np.load(npz_path)
        self.inputs = data["inputs"]  # (N, 2, H, W)
        self.masks = data["masks"]    # (N, 1, H, W)

    def __len__(self):
        return self.inputs.shape[0]

    def __getitem__(self, idx):
        x = self.inputs[idx]   # float32
        y = self.masks[idx]

        x = torch.from_numpy(x)   # [2, H, W]
        y = torch.from_numpy(y)   # [1, H, W]
        return x, y


class FusionMotionCNN(nn.Module):
    """
    Tiny early-fusion CNN (2-channel input -> 1-channel mask).
    """
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(2, 8, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, kernel_size=1)  # logits
        )

    def forward(self, x):
        return self.net(x)


def train_model(epochs=3, batch_size=16, lr=1e-3):
    print("[INFO] Loading fusion dataset from:", DATA_FILE)
    ds = FusionMotionDataset(DATA_FILE)

    n_total = len(ds)
    n_train = int(0.8 * n_total)
    n_val = n_total - n_train
    train_ds, val_ds = torch.utils.data.random_split(ds, [n_train, n_val])

    print(f"[INFO] Dataset split: train={len(train_ds)}, val={len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[INFO] Using device:", device)

    model = FusionMotionCNN().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0

        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * x.size(0)

        train_loss /= len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device)
                y = y.to(device)
                logits = model(x)
                loss = criterion(logits, y)
                val_loss += loss.item() * x.size(0)

        val_loss /= len(val_loader.dataset)
        print(f"[EPOCH {epoch}] train_loss={train_loss:.4f}  val_loss={val_loss:.4f}")

    torch.save(model.state_dict(), MODEL_PATH)
    print("[SUCCESS] Saved FusionMotionCNN model to:", MODEL_PATH)


if __name__ == "__main__":
    train_model()
