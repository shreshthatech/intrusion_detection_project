"""
Task 16 (part 1): Build RGB+thermal early-fusion dataset.
"""

from pathlib import Path
import cv2
import numpy as np
import os
import sys

# Add src/vision folder to PATH
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(CURRENT_DIR)

from thermal_preprocess import normalize_thermal, apply_clahe

PROJECT_ROOT = Path(__file__).resolve().parents[2]
VIDEO_PATH = PROJECT_ROOT / "data" / "processed" / "synthetic_intrusion.mp4"

OUT_DIR = PROJECT_ROOT / "data" / "processed" / "motion_masks"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_FILE = OUT_DIR / "fusion_motion_dataset.npz"

TARGET_W = 160
TARGET_H = 90

def main():
    print("[INFO] Opening video:", VIDEO_PATH)
    cap = cv2.VideoCapture(str(VIDEO_PATH))
    if not cap.isOpened():
        print("[ERROR] Could not open video!")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"[INFO] FPS={fps:.2f}, total_frames={total}")

    prev_gray = None
    inputs = []
    masks = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_small = cv2.resize(gray, (TARGET_W, TARGET_H))

        therm_norm = normalize_thermal(gray_small)
        therm = apply_clahe(therm_norm)

        if prev_gray is not None:
            diff = cv2.absdiff(gray_small, prev_gray)
            _, mask = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)

            fused = np.stack([
                gray_small.astype(np.float32) / 255.0,
                therm.astype(np.float32) / 255.0
            ], axis=0)

            m = (mask.astype(np.float32) / 255.0)[None, :, :]

            inputs.append(fused)
            masks.append(m)

        prev_gray = gray_small

    cap.release()

    inputs = np.stack(inputs)
    masks = np.stack(masks)

    print("[INFO] Saving dataset...")
    print("inputs:", inputs.shape)
    print("masks :", masks.shape)

    np.savez_compressed(OUT_FILE, inputs=inputs, masks=masks, fps=fps)
    print("[SUCCESS] Saved:", OUT_FILE)

if __name__ == "__main__":
    main()
