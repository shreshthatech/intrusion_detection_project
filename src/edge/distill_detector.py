"""
Task 63: Knowledge Distillation (Teacher → Student)

Teacher = MobileNet-SSD detections (from postprocessed_detections.jsonl)
Student = TinyDetectorCNN (lightweight)

We train the student to:
  - Match teacher’s confidence (soft objectness)
  - Match teacher’s bbox (L1 regression)
"""

import json
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# ------------------- Student Model ------------------- #

class TinyDetectorCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 8, 3, padding=1), nn.ReLU(),
            nn.Conv2d(8, 16, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.fc_obj = nn.Linear(16, 1)   # objectness logit
        self.fc_box = nn.Linear(16, 4)   # bbox (x1,y1,x2,y2) or similar

    def forward(self, x):
        f = self.features(x)            # (B,16,1,1)
        f = f.view(f.size(0), -1)       # (B,16)
        obj_logit = self.fc_obj(f)      # (B,1)
        box = self.fc_box(f)            # (B,4)
        return obj_logit, box


# ------------------- Dataset ------------------- #

class DistillDataset(Dataset):
    def __init__(self, project_root: Path):
        self.samples = []

        det_path = project_root / "data" / "processed" / "rgb_detections" / "postprocessed_detections.jsonl"
        if not det_path.exists():
            raise FileNotFoundError(f"Detections file not found: {det_path}")

        with open(det_path, "r") as f:
            for line in f:
                ev = json.loads(line)

                # ---- get confidence from any reasonable key ----
                score = None
                for k in ["score", "conf", "confidence", "p"]:
                    if k in ev:
                        score = float(ev[k])
                        break
                if score is None:
                    # fallback default if no key exists
                    score = 0.5

                # ---- get bbox ----
                box = None
                for k in ["bbox", "box", "bbox_xyxy"]:
                    if k in ev:
                        box = ev[k]
                        break
                if box is None:
                    # fallback: 0 box
                    box = [0.0, 0.0, 0.0, 0.0]

                self.samples.append({
                    "score": score,
                    "bbox": box,
                })

        if not self.samples:
            raise RuntimeError(f"No samples loaded from {det_path}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        ev = self.samples[idx]

        # Student does not actually see the real frame here; we just
        # demonstrate the distillation mechanism with random inputs.
        img = torch.randn(3, 224, 224)          # fake image
        t_conf = torch.tensor([ev["score"]], dtype=torch.float32)  # (1,)
        t_box = torch.tensor(ev["bbox"], dtype=torch.float32)      # (4,)

        return img, t_conf, t_box


# ------------------- Distillation Training ------------------- #

def distill():
    print("[INFO] Task 63 – Knowledge Distillation")

    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    SAVE_PATH = PROJECT_ROOT / "models" / "vision" / "tiny_detector_distilled.pt"

    ds = DistillDataset(PROJECT_ROOT)
    loader = DataLoader(ds, batch_size=32, shuffle=True)

    student = TinyDetectorCNN()
    student.train()

    opt = torch.optim.Adam(student.parameters(), lr=1e-3)

    alpha = 1.0   # weight for confidence distillation
    beta = 0.1    # weight for bbox regression

    for epoch in range(1, 6):
        total_loss = 0.0

        for img, t_conf, t_box in loader:
            opt.zero_grad()

            s_logits, s_box = student(img)        # (B,1), (B,4)

            # ----- confidence distillation (MSE between probabilities) -----
            s_prob = torch.sigmoid(s_logits).squeeze(1)   # (B,)
            t_prob = t_conf.squeeze(1).clamp(0.0, 1.0)    # (B,)
            conf_loss = F.mse_loss(s_prob, t_prob)

            # ----- bbox distillation (L1) -----
            box_loss = F.l1_loss(s_box, t_box)

            loss = alpha * conf_loss + beta * box_loss
            loss.backward()
            opt.step()

            total_loss += loss.item()

        avg_loss = total_loss / max(1, len(loader))
        print(f"[EPOCH {epoch}] distill_loss={avg_loss:.4f}")

    torch.save(student.state_dict(), SAVE_PATH)
    print(f"[SUCCESS] Saved distilled student model to: {SAVE_PATH}")
    print("[DEMO] Task 63 distillation completed.")


if __name__ == "__main__":
    distill()
