"""
Task 17: Hard negative mining (non-intruder detections).

We:
  - load detections from rgb_detections/detections_sample.jsonl
  - load synthetic intruder labels (frame_id,x,y,w,h)
  - for each detection bbox, if IoU with intruder bbox < 0.1,
    treat it as a hard negative
  - crop that region from the original RGB frame
  - save crops + metadata

Output:
  data/processed/hard_negatives/
    neg_00001.jpg
    ...
    hard_negatives_meta.jsonl
"""

from pathlib import Path
import json
import cv2
import os

PROJECT_ROOT = Path(__file__).resolve().parents[2]

VIDEO_PATH = PROJECT_ROOT / "data" / "processed" / "synthetic_intrusion.mp4"
DETS_PATH = PROJECT_ROOT / "data" / "processed" / "rgb_detections" / "detections_sample.jsonl"
LABELS_PATH = PROJECT_ROOT / "data" / "processed" / "synthetic_labels.txt"

OUT_DIR = PROJECT_ROOT / "data" / "processed" / "hard_negatives"
OUT_DIR.mkdir(parents=True, exist_ok=True)
META_PATH = OUT_DIR / "hard_negatives_meta.jsonl"


def load_intruder_boxes():
    """
    synthetic_labels.txt: frame_id,x,y,w,h
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
            intruder[frame_id] = (x, y, x + w, y + h)  # (x1,y1,x2,y2)
    return intruder


def load_detections():
    dets_by_frame = {}
    if not DETS_PATH.exists():
        print("[WARN] Detections JSONL not found:", DETS_PATH)
        return dets_by_frame

    with open(DETS_PATH, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            frame_idx = int(obj["frame_idx"])
            dets_by_frame.setdefault(frame_idx, []).extend(obj["detections"])
    return dets_by_frame


def iou(boxA, boxB):
    """
    Intersection over Union of two boxes:
    box = (x1,y1,x2,y2)
    """
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


def main():
    print("[INFO] Loading intruder boxes...")
    intruder_boxes = load_intruder_boxes()
    print(f"[INFO] Loaded intruder boxes for {len(intruder_boxes)} frames.")

    print("[INFO] Loading detections...")
    dets_by_frame = load_detections()
    print(f"[INFO] Frames with detections: {len(dets_by_frame)}")

    print("[INFO] Opening video:", VIDEO_PATH)
    cap = cv2.VideoCapture(str(VIDEO_PATH))
    if not cap.isOpened():
        raise RuntimeError("Could not open video")

    neg_count = 0
    meta_out = open(META_PATH, "w")

    for frame_idx, dets in dets_by_frame.items():
        # Set video position to this frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            print(f"[WARN] Could not read frame {frame_idx}")
            continue

        h, w = frame.shape[:2]

        intr_box = intruder_boxes.get(frame_idx, None)

        for det in dets:
            label = det["label"]
            conf = float(det["confidence"])
            x1, y1, x2, y2 = det["bbox"]

            # Clip to image bounds
            x1 = max(0, min(x1, w - 1))
            y1 = max(0, min(y1, h - 1))
            x2 = max(0, min(x2, w - 1))
            y2 = max(0, min(y2, h - 1))

            if x2 <= x1 or y2 <= y1:
                continue

            det_box = (x1, y1, x2, y2)

            # If we have an intruder box for this frame, check IoU
            if intr_box is not None:
                overlap = iou(det_box, intr_box)
                # If big overlap, it's probably the intruder, skip
                if overlap >= 0.1:
                    continue

            # Now this is a hard negative candidate
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            neg_count += 1
            fname = f"neg_{neg_count:05d}.jpg"
            out_path = OUT_DIR / fname
            cv2.imwrite(str(out_path), crop)

            meta = {
                "file": fname,
                "frame_idx": frame_idx,
                "label": label,
                "confidence": conf,
                "bbox": [x1, y1, x2, y2],
            }
            meta_out.write(json.dumps(meta) + "\n")

    meta_out.close()
    cap.release()

    print(f"[SUCCESS] Saved {neg_count} hard negative crops to:", OUT_DIR)
    print("[SUCCESS] Metadata JSONL:", META_PATH)


if __name__ == "__main__":
    main()
