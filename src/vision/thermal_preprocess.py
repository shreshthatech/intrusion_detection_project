"""
Task 13: Thermal image normalization + CLAHE enhancement.

This module provides:
  - normalize_thermal(): contrast-stretches thermal frames
  - apply_clahe(): CLAHE on 8-bit thermal images
  - demo_thermal_pipeline(): demo using synthetic_intrusion.mp4
"""

from pathlib import Path
import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
VIDEO_PATH = PROJECT_ROOT / "data" / "processed" / "synthetic_intrusion.mp4"

OUT_DIR = PROJECT_ROOT / "data" / "processed" / "thermal_demo"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def normalize_thermal(img, clip_percent=2.0):
    """
    Normalize a thermal image to 8-bit using contrast stretching.

    img can be:
      - 16-bit single-channel (typical thermal)
      - 8-bit single-channel

    Steps:
      1. Compute lower/upper percentiles (e.g. 2% and 98%).
      2. Linearly map [low, high] -> [0, 255].
    """
    # Ensure it's a numpy array
    arr = img.astype(np.float32)

    # Compute percentiles to ignore extreme outliers
    low = np.percentile(arr, clip_percent)
    high = np.percentile(arr, 100.0 - clip_percent)

    if high <= low:
        # Degenerate case, just scale min/max
        low = arr.min()
        high = arr.max() if arr.max() > low else low + 1.0

    arr = (arr - low) / (high - low)
    arr = np.clip(arr, 0.0, 1.0)

    # Convert to 8-bit
    out = (arr * 255.0).astype(np.uint8)
    return out


def apply_clahe(img_8bit, clip_limit=2.0, tile_grid_size=(8, 8)):
    """
    Apply CLAHE (adaptive histogram equalization) to an 8-bit single-channel image.
    """
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    enhanced = clahe.apply(img_8bit)
    return enhanced


def demo_thermal_pipeline():
    """
    Demo:
      - open synthetic_intrusion.mp4
      - treat grayscale version as 'thermal'
      - apply normalize_thermal + CLAHE
      - save example frames
    """
    print("[INFO] Opening video for thermal demo:", VIDEO_PATH)
    cap = cv2.VideoCapture(str(VIDEO_PATH))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {VIDEO_PATH}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print("[INFO] Total frames:", total_frames)

    # Pick some frame indices to sample
    indices = np.linspace(0, total_frames - 1, 5, dtype=int)
    print("[INFO] Sampling frames at indices:", indices)

    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            print(f"[WARN] Could not read frame {idx}")
            continue

        # Convert RGB->GRAY as fake thermal
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Step 1: normalize thermal
        norm = normalize_thermal(gray)

        # Step 2: CLAHE
        clahe_img = apply_clahe(norm)

        # Save results
        base = f"thermal_{idx}"
        cv2.imwrite(str(OUT_DIR / f"{base}_orig_gray.jpg"), gray)
        cv2.imwrite(str(OUT_DIR / f"{base}_norm.jpg"), norm)
        cv2.imwrite(str(OUT_DIR / f"{base}_clahe.jpg"), clahe_img)

    cap.release()
    print("[SUCCESS] Thermal normalization + CLAHE demo images saved to:", OUT_DIR)


if __name__ == "__main__":
    demo_thermal_pipeline()
