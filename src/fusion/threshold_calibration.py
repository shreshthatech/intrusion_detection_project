"""
Task 24: ROC-based per-sensor threshold calibration.

We compute:
  - true labels (intruder/non-intruder) based on IoU >= 0.5
  - predicted scores (confidence values)
  - ROC curve
  - best threshold using Youden's J statistic: J = TPR - FPR

We support multiple sensors, but for now we calibrate:
    - video (using postprocessed detections)
"""

import json
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_curve
import math


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DETS_FILE = PROJECT_ROOT / "data" / "processed" / "rgb_detections" / "postprocessed_detections.jsonl"
LABELS_FILE = PROJECT_ROOT / "data" / "processed" / "synthetic_labels.txt"

OUT_DIR = PROJECT_ROOT / "models" / "fusion"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_THRESHOLDS = OUT_DIR / "sensor_thresholds.json"


# ---------------- helpers ---------------- #

def load_intruder_boxes():
    """
    Returns: dict frame -> (x1,y1,x2,y2)
    """
    gt = {}
    with open(LABELS_FILE, "r") as f:
        for line in f:
            if not line.strip():
                continue
            parts = line.strip().split(",")
            frame = int(parts[0])
            x, y, w, h = map(int, parts[1:])
            gt[frame] = (x, y, x+w, y+h)
    return gt


def iou(a, b):
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])

    if x2 <= x1 or y2 <= y1:
        return 0.0

    inter = (x2 - x1) * (y2 - y1)
    areaA = (a[2]-a[0]) * (a[3]-a[1])
    areaB = (b[2]-b[0]) * (b[3]-b[1])
    return inter / (areaA + areaB - inter + 1e-6)


def load_video_scores(gt_boxes):
    """
    Builds:
        scores: list of detection confidence
        labels: list of GT labels (1 = intruder, 0 = negative)
    """
    scores = []
    labels = []

    with open(DETS_FILE, "r") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)

            frame = obj["frame_idx"]
            if frame not in gt_boxes:
                continue

            gt = gt_boxes[frame]
            box = obj["bbox"]

            ov = iou(gt, box)

            label = 1 if ov >= 0.5 else 0
            score = float(obj["conf_calibrated"])

            scores.append(score)
            labels.append(label)

    return np.array(scores), np.array(labels)


def compute_best_threshold(scores, labels):
    """
    Compute ROC curve and pick threshold via Youden's J statistic.

    Returns: best_thresh, (fpr, tpr thresholds arrays)
    """
    if len(np.unique(labels)) < 2:
        print("[WARN] Only one class present. Using fallback threshold = 0.5")
        return 0.5, None

    fpr, tpr, thresh = roc_curve(labels, scores)
    J = tpr - fpr
    best_idx = np.argmax(J)
    best_thresh = thresh[best_idx]
    return best_thresh, (fpr, tpr, thresh)


# ---------------- main pipeline ---------------- #

def calibrate():
    print("[INFO] Loading GT intruder boxes...")
    gt_boxes = load_intruder_boxes()

    print("[INFO] Loading video detection scores...")
    scores, labels = load_video_scores(gt_boxes)
    print(f"[INFO] Loaded {len(scores)} video samples")

    # Compute threshold
    print("[INFO] Computing ROC threshold (video)...")
    best_thresh, roc_data = compute_best_threshold(scores, labels)

    thresholds = {
        "video": float(best_thresh),
        "audio": None,     # future tasks
        "rf": None,
        "thermal": None
    }

    with open(OUT_THRESHOLDS, "w") as f:
        json.dump(thresholds, f, indent=2)

    print("[SUCCESS] Saved thresholds to:", OUT_THRESHOLDS)
    print("[INFO] Best video threshold:", best_thresh)


if __name__ == "__main__":
    calibrate()
