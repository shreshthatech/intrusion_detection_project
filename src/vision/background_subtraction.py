"""
Task 12: Background subtraction with adaptive learning rate.

We:
  - Open synthetic_intrusion.mp4
  - Use OpenCV MOG2 background subtractor
  - Adjust learning rate based on detected motion
  - Save sample masks to data/processed/bg_sub/
  - Save foreground statistics to JSON
"""

from pathlib import Path
import cv2
import numpy as np
import json


PROJECT_ROOT = Path(__file__).resolve().parents[2]
VIDEO_PATH = PROJECT_ROOT / "data" / "processed" / "synthetic_intrusion.mp4"

OUT_DIR = PROJECT_ROOT / "data" / "processed" / "bg_sub"
OUT_DIR.mkdir(parents=True, exist_ok=True)

STATS_PATH = OUT_DIR / "bg_stats.json"


def main():
    print("[INFO] Opening video:", VIDEO_PATH)
    
    cap = cv2.VideoCapture(str(VIDEO_PATH))
    if not cap.isOpened():
        raise RuntimeError("Could not open video")

    # Background subtractor
    backsub = cv2.createBackgroundSubtractorMOG2(
        history=500,
        varThreshold=25,
        detectShadows=True
    )

    frame_count = 0
    fg_ratios = []
    sample_saved = 0

    motion_history = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Reduce size for speed
        small = cv2.resize(frame, (320, 180))

        # Detect current motion amount (rough proxy)
        # Average absdiff with previous frame
        if frame_count == 0:
            prev_small = small.copy()

        diff = cv2.absdiff(small, prev_small)
        motion = np.mean(diff)
        motion_history.append(motion)

        # Adaptive learning rate:
        # More motion → lower learning rate
        # Less motion → higher learning rate (background adapts)
        if motion > 20:
            lr = 0.001
        elif motion > 10:
            lr = 0.01
        else:
            lr = 0.05

        fg_mask = backsub.apply(small, learningRate=lr)

        # Threshold
        _, fg_bin = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)

        # Foreground ratio
        fg_ratio = np.mean(fg_bin / 255.0)
        fg_ratios.append(float(fg_ratio))

        # Save sample masks (first 10 frames)
        if sample_saved < 10:
            out_path = OUT_DIR / f"fg_{sample_saved}.jpg"
            cv2.imwrite(str(out_path), fg_bin)
            sample_saved += 1

        prev_small = small.copy()
        frame_count += 1

    cap.release()

    # Save stats
    stats = {
        "total_frames": frame_count,
        "avg_fg_ratio": float(np.mean(fg_ratios)),
        "max_fg_ratio": float(np.max(fg_ratios)),
        "min_fg_ratio": float(np.min(fg_ratios)),
        "motion_history_sample": motion_history[:50]
    }

    with open(STATS_PATH, "w") as f:
        json.dump(stats, f, indent=2)

    print("[SUCCESS] Background subtraction samples saved to:", OUT_DIR)
    print("[SUCCESS] Stats saved to:", STATS_PATH)


if __name__ == "__main__":
    main()
