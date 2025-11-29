"""
Task 15: Run the same MobileNet-SSD detector on thermal images.

We:
  - open synthetic_intrusion.mp4
  - convert each frame to "thermal" using thermal_preprocess.normalize_thermal + CLAHE
  - stack to 3-channel (thermal, thermal, thermal)
  - run MobileNet-SSD
  - save annotated frames + detections JSONL

Later, if you have a real thermal video, just point VIDEO_PATH to that file.

"""
import sys
import os

# Add the current folder (src/vision) to Python path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from thermal_preprocess import normalize_thermal, apply_clahe




from pathlib import Path
import cv2
import numpy as np
import json




PROJECT_ROOT = Path(__file__).resolve().parents[2]

# For now, we still use the same RGB video as a source
VIDEO_PATH = PROJECT_ROOT / "data" / "processed" / "synthetic_intrusion.mp4"

MODEL_DIR = PROJECT_ROOT / "models" / "vision" / "mobilenet_ssd"
PROTO_PATH = MODEL_DIR / "MobileNetSSD_deploy.prototxt"
MODEL_PATH = MODEL_DIR / "MobileNetSSD_deploy.caffemodel"

OUT_DIR = PROJECT_ROOT / "data" / "processed" / "thermal_detections"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DETS_JSON = OUT_DIR / "detections_sample_thermal.jsonl"

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

    print("[INFO] Loading MobileNet-SSD for THERMAL from:")
    print("  Proto:", PROTO_PATH)
    print("  Model:", MODEL_PATH)

    net = cv2.dnn.readNetFromCaffe(str(PROTO_PATH), str(MODEL_PATH))
    return net


def rgb_to_fake_thermal(frame_bgr):
    """
    Convert an RGB frame to a 'thermal-like' 3-channel image:
      - grayscale
      - normalize_thermal
      - CLAHE
      - stack to (H, W, 3)
    """
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    norm = normalize_thermal(gray)
    clahe_img = apply_clahe(norm)

    # stack into 3 channels for MobileNet-SSD
    thermal_3ch = cv2.merge([clahe_img, clahe_img, clahe_img])
    return thermal_3ch


def run_thermal_detector(conf_threshold=0.4, save_every_n=30):
    net = load_model()

    print("[INFO] Opening video for thermal detection:", VIDEO_PATH)
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

        # Convert to "thermal"
        thermal_frame = rgb_to_fake_thermal(frame)

        # Prepare blob
        blob = cv2.dnn.blobFromImage(
            cv2.resize(thermal_frame, (300, 300)),
            0.007843,
            (300, 300),
            127.5
        )

        net.setInput(blob)
        detections = net.forward()

        frame_dets = []

        for i in range(detections.shape[2]):
            confidence = float(detections[0, 0, i, 2])
            if confidence < conf_threshold:
                continue

            class_id = int(detections[0, 0, i, 1])
            if class_id < 0 or class_id >= len(CLASS_LABELS):
                continue

            label = CLASS_LABELS[class_id]

            box = detections[0, 0, i, 3:7]
            (x_min, y_min, x_max, y_max) = box
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

            # draw on thermal_frame for visualization (green boxes)
            cv2.rectangle(thermal_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            text = f"{label}:{confidence:.2f}"
            cv2.putText(thermal_frame, text, (x1, max(y1 - 5, 0)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        if frame_dets:
            out_obj = {
                "frame_idx": frame_idx,
                "detections": frame_dets
            }
            det_out.write(json.dumps(out_obj) + "\n")

        if frame_idx % save_every_n == 0:
            out_path = OUT_DIR / f"thermal_frame_{frame_idx:04d}.jpg"
            cv2.imwrite(str(out_path), thermal_frame)
            saved_count += 1

        frame_idx += 1

    det_out.close()
    cap.release()

    print(f"[SUCCESS] Processed {frame_idx} frames for THERMAL.")
    print(f"[SUCCESS] Saved {saved_count} annotated thermal frames to:", OUT_DIR)
    print(f"[SUCCESS] Sample thermal detections JSONL:", DETS_JSON)


if __name__ == "__main__":
    run_thermal_detector()
