"""
Task 18: Post-processing – NMS + confidence calibration (Platt-style).

Steps:
  1. Load raw detections from rgb_detections/detections_sample.jsonl
  2. Load synthetic intruder ground-truth boxes
  3. Build a calibration dataset: (raw_confidence -> is_intruder)
  4. Fit a logistic regression (Platt-like sigmoid) as calibrator
     - If only one class exists, fallback to "no calibration" (raw = calibrated)
  5. Apply per-frame NMS on 'person' detections
  6. Apply calibrated confidence and save postprocessed JSONL
"""

from pathlib import Path
import json
import numpy as np
from sklearn.linear_model import LogisticRegression
import joblib


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DETS_IN = PROJECT_ROOT / "data" / "processed" / "rgb_detections" / "detections_sample.jsonl"
LABELS_PATH = PROJECT_ROOT / "data" / "processed" / "synthetic_labels.txt"

DETS_OUT = PROJECT_ROOT / "data" / "processed" / "rgb_detections" / "postprocessed_detections.jsonl"
MODEL_DIR = PROJECT_ROOT / "models" / "vision"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
CALIBRATOR_PATH = MODEL_DIR / "conf_calibrator.joblib"


# ---------- helpers ---------- #

def load_intruder_boxes():
    """
    synthetic_labels.txt: frame_id,x,y,w,h
    returns dict: frame_id -> (x1,y1,x2,y2)
    """
    intruder = {}
    with open(LABELS_PATH, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(",")
            frame_id = int(parts[0])
            x = int(parts[1])
            y = int(parts[2])
            w = int(parts[3])
            h = int(parts[4])
            intruder[frame_id] = (x, y, x + w, y + h)
    return intruder


def load_detections():
    """
    returns dict: frame_idx -> list of detection dicts
    """
    dets_by_frame = {}
    with open(DETS_IN, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            frame_idx = int(obj["frame_idx"])
            dets_by_frame.setdefault(frame_idx, []).extend(obj["detections"])
    return dets_by_frame


def iou(boxA, boxB):
    (x1A, y1A, x2A, y2A) = boxA
    (x1B, y1B, x2B, y2B) = boxB

    inter_x1 = max(x1A, x1B)
    inter_y1 = max(y1A, y1B)
    inter_x2 = min(x2A, x2B)
    inter_y2 = min(y2A, y2B)

    if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
        return 0.0

    inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
    areaA = (x2A - x1A) * (y2A - y1A)
    areaB = (x2B - x1B) * (y2B - y1B)

    union = areaA + areaB - inter_area
    if union <= 0:
        return 0.0
    return inter_area / union


def nms(boxes, scores, iou_thresh=0.45):
    """
    Simple NMS implementation.
    boxes: Nx4 (x1,y1,x2,y2)
    scores: N
    returns: indices of boxes to keep
    """
    if len(boxes) == 0:
        return []

    boxes = np.array(boxes)
    scores = np.array(scores)

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h

        iou_vals = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
        inds = np.where(iou_vals <= iou_thresh)[0]
        order = order[inds + 1]

    return keep


# ---------- calibration ---------- #

def fit_calibrator(dets_by_frame, intruder_boxes):
    """
    Build dataset: raw_confidence -> is_intruder (1/0)
    Use IoU >= 0.5 as positive.
    If only one class present, return None (fallback: identity calibration).
    """
    X = []
    y = []

    for frame_idx, dets in dets_by_frame.items():
        intr_box = intruder_boxes.get(frame_idx, None)
        if intr_box is None:
            continue

        for d in dets:
            conf = float(d["confidence"])
            x1, y1, x2, y2 = d["bbox"]
            det_box = (x1, y1, x2, y2)
            overlap = iou(det_box, intr_box)
            label = 1 if overlap >= 0.5 else 0
            X.append([conf])
            y.append(label)

    if not X:
        print("[WARN] No calibration samples found. Using raw confidences as-is.")
        joblib.dump(None, CALIBRATOR_PATH)
        return None

    X = np.array(X)
    y = np.array(y)

    positives = int(y.sum())
    negatives = int((y == 0).sum())
    print(f"[INFO] Calibration dataset: {X.shape[0]} samples (positives={positives}, negatives={negatives})")

    # If only one class present, skip fitting
    if len(np.unique(y)) < 2:
        print("[WARN] Only one class present in calibration data. Skipping calibration.")
        joblib.dump(None, CALIBRATOR_PATH)
        return None

    clf = LogisticRegression()
    clf.fit(X, y)

    joblib.dump(clf, CALIBRATOR_PATH)
    print("[SUCCESS] Saved confidence calibrator to:", CALIBRATOR_PATH)

    return clf


# ---------- main pipeline ---------- #

def run_postprocessing(conf_thresh=0.3, nms_iou=0.45):
    print("[INFO] Loading intruder boxes...")
    intruder_boxes = load_intruder_boxes()

    print("[INFO] Loading raw detections...")
    dets_by_frame = load_detections()
    print("[INFO] Frames with detections:", len(dets_by_frame))

    print("[INFO] Fitting confidence calibrator (Platt-style)...")
    calibrator = fit_calibrator(dets_by_frame, intruder_boxes)

    print("[INFO] Running NMS + calibration...")
    out_f = open(DETS_OUT, "w")
    total_kept = 0

    for frame_idx, dets in dets_by_frame.items():
        # focus on 'person' detections
        person_boxes = []
        person_scores = []
        person_indices = []

        for idx, d in enumerate(dets):
            if d["label"] != "person":
                continue
            score = float(d["confidence"])
            if score < conf_thresh:
                continue
            x1, y1, x2, y2 = d["bbox"]
            person_boxes.append([x1, y1, x2, y2])
            person_scores.append(score)
            person_indices.append(idx)

        if not person_boxes:
            continue

        keep_local = nms(person_boxes, person_scores, iou_thresh=nms_iou)

        for ki in keep_local:
            orig_idx = person_indices[ki]
            d = dets[orig_idx]
            raw_conf = float(d["confidence"])

            # If we couldn't train calibrator, use raw_conf directly
            if calibrator is None:
                cal_prob = raw_conf
            else:
                cal_prob = float(calibrator.predict_proba([[raw_conf]])[0, 1])

            out_obj = {
                "frame_idx": frame_idx,
                "label": d["label"],
                "bbox": d["bbox"],
                "conf_raw": raw_conf,
                "conf_calibrated": cal_prob,
            }
            out_f.write(json.dumps(out_obj) + "\n")
            total_kept += 1

    out_f.close()
    print(f"[SUCCESS] Postprocessed detections saved to: {DETS_OUT}")
    print(f"[INFO] Total kept detections after NMS: {total_kept}")


if __name__ == "__main__":
    run_postprocessing()
