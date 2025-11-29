"""
Task 14: Lightweight object detector (MobileNet-SSD) on RGB.

This script:
  - loads a pre-trained MobileNet-SSD model (Caffe)
  - runs it on synthetic_intrusion.mp4
  - draws detections (esp. 'person')
  - saves sample annotated frames to data/processed/rgb_detections/
"""

from pathlib import Path
import cv2
import numpy as np
import json


PROJECT_ROOT = Path(__file__).resolve().parents[2]

VIDEO_PATH = PROJECT_ROOT / "data" / "processed" / "synthetic_intrusion.mp4"

MODEL_DIR = PROJECT_ROOT / "models" / "vision" / "mobilenet_ssd"
PROTO_PATH = MODEL_DIR / "MobileNetSSD_deploy.prototxt"
MODEL_PATH = MODEL_DIR / "MobileNetSSD_deploy.caffemodel"

OUT_DIR = PROJECT_ROOT / "data" / "processed" / "rgb_detections"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DETS_JSON = OUT_DIR / "detections_sample.jsonl"

# MobileNet-SSD class labels (standard list)
CLASS_LABELS = [
    "background", "aeroplane", "bicycle", "bird", "boat",
    "bottle", "bus", "car", "cat", "chair",
    "cow", "diningtable", "dog", "horse", "motorbike",
    "person", "pottedplant", "sheep", "sofa", "train",
    "tvmonitor"
]


def load_model():
    if not PROTO_PATH.exists() or not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Missing MobileNet-SSD model files.\n"
            f"Expected:\n  {PROTO_PATH}\n  {MODEL_PATH}"
        )

    print("[INFO] Loading MobileNet-SSD from:")
    print("  Proto:", PROTO_PATH)
    print("  Model:", MODEL_PATH)

    net = cv2.dnn.readNetFromCaffe(str(PROTO_PATH), str(MODEL_PATH))
    return net


def run_detector(conf_threshold=0.4, save_every_n=30):
    net = load_model()

    print("[INFO] Opening video:", VIDEO_PATH)
    cap = cv2.VideoCapture(str(VIDEO_PATH))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {VIDEO_PATH}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"[INFO] FPS={fps:.2f}, total_frames={total_frames}")

    frame_idx = 0
    saved_count = 0

    det_out = open(DETS_JSON, "w")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        (h, w) = frame.shape[:2]

        # Prepare input blob (typical SSD preprocessing)
        blob = cv2.dnn.blobFromImage(
            cv2.resize(frame, (300, 300)),
            0.007843,  # scale factor
            (300, 300),
            127.5      # mean subtraction
        )

        net.setInput(blob)
        detections = net.forward()  # shape: [1, 1, N, 7]

        frame_dets = []

        for i in range(detections.shape[2]):
            confidence = float(detections[0, 0, i, 2])
            if confidence < conf_threshold:
                continue

            class_id = int(detections[0, 0, i, 1])
            if class_id < 0 or class_id >= len(CLASS_LABELS):
                continue

            label = CLASS_LABELS[class_id]

            # bounding box (relative 0..1)
            box = detections[0, 0, i, 3:7]
            (x_min, y_min, x_max, y_max) = box
            # scale to frame size
            x1 = int(x_min * w)
            y1 = int(y_min * h)
            x2 = int(x_max * w)
            y2 = int(y_max * h)

            frame_dets.append({
                "frame_idx": frame_idx,
                "label": label,
                "confidence": confidence,
                "bbox": [x1, y1, x2, y2]
            })

            # Draw on frame (only for visualization)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            text = f"{label}:{confidence:.2f}"
            cv2.putText(frame, text, (x1, max(y1 - 5, 0)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # Save detections to JSONL (for now just a subset)
        if frame_dets:
            out_obj = {
                "frame_idx": frame_idx,
                "detections": frame_dets
            }
            det_out.write(json.dumps(out_obj) + "\n")

        # Save some sample annotated frames to disk
        if frame_idx % save_every_n == 0:
            out_path = OUT_DIR / f"frame_{frame_idx:04d}.jpg"
            cv2.imwrite(str(out_path), frame)
            saved_count += 1

        frame_idx += 1

    det_out.close()
    cap.release()

    print(f"[SUCCESS] Processed {frame_idx} frames.")
    print(f"[SUCCESS] Saved {saved_count} annotated frames to:", OUT_DIR)
    print(f"[SUCCESS] Sample detections JSONL:", DETS_JSON)


if __name__ == "__main__":
    run_detector()
