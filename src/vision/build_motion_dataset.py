"""
Task 11 (part 1): Build motion-mask dataset from video.

We:
  - Read synthetic_intrusion.mp4
  - For each pair (frame_{t-1}, frame_t) compute abs difference
  - Threshold to get a motion mask (foreground/background)
  - Resize to small size (e.g. 160x90)
  - Save all pairs and masks into an .npz file

Outputs:
  data/processed/motion_masks/motion_dataset.npz
"""

from pathlib import Path
import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
VIDEO_PATH = PROJECT_ROOT / "data" / "processed" / "synthetic_intrusion.mp4"
OUT_DIR = PROJECT_ROOT / "data" / "processed" / "motion_masks"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_FILE = OUT_DIR / "motion_dataset.npz"

TARGET_W = 160
TARGET_H = 90   # 16:9 aspect approx


def main():
    print("[INFO] Opening video:", VIDEO_PATH)
    cap = cv2.VideoCapture(str(VIDEO_PATH))
    if not cap.isOpened():
        raise RuntimeError("Could not open video.")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"[INFO] FPS={fps:.2f}, total_frames={total_frames}")

    prev_gray_small = None

    inputs = []   # (2, H, W) – prev + current frames (grayscale)
    masks = []    # (1, H, W) – motion mask

    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Resize to small size
        gray_small = cv2.resize(gray, (TARGET_W, TARGET_H), interpolation=cv2.INTER_AREA)

        if prev_gray_small is not None:
            # Compute absolute difference
            diff = cv2.absdiff(gray_small, prev_gray_small)

            # Threshold to create binary mask
            # You can adjust threshold if motion is too little / too much
            _, mask = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)

            # Normalize to 0–1
            inp = np.stack([prev_gray_small, gray_small], axis=0).astype(np.float32) / 255.0
            m = (mask.astype(np.float32) / 255.0)[None, :, :]  # (1, H, W)

            inputs.append(inp)
            masks.append(m)

        prev_gray_small = gray_small
        frame_idx += 1

    cap.release()

    inputs = np.stack(inputs, axis=0)   # (N, 2, H, W)
    masks = np.stack(masks, axis=0)     # (N, 1, H, W)

    print("[INFO] Dataset shapes:")
    print("  inputs:", inputs.shape)
    print("  masks: ", masks.shape)

    np.savez_compressed(OUT_FILE, inputs=inputs, masks=masks, fps=fps)
    print("[SUCCESS] Saved motion dataset to:", OUT_FILE)


if __name__ == "__main__":
    main()
