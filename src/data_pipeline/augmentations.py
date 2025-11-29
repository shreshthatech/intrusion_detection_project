"""
Task 7: On-the-fly augmentation pipeline (night, fog, rain).

This module defines simple OpenCV-based augmentations that we can
later plug into a PyTorch Dataset.

For now, it:
  1. Loads synthetic_intrusion.mp4
  2. Samples some frames
  3. Applies night, fog, rain, and random augmentations
  4. Saves example images to data/processed/augmented_samples/
"""

from pathlib import Path
import cv2
import numpy as np
import random


PROJECT_ROOT = Path(__file__).resolve().parents[2]
VIDEO_PATH = PROJECT_ROOT / "data" / "processed" / "synthetic_intrusion.mp4"
OUT_DIR = PROJECT_ROOT / "data" / "processed" / "augmented_samples"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# --------- Augmentation functions --------- #

def night_effect(frame):
    """
    Darken the frame and add a slight blue tint to simulate night.
    """
    # Convert to float for safe multiplication
    img = frame.astype(np.float32) / 255.0

    # Darken
    darkness = random.uniform(0.3, 0.6)
    img = img * darkness

    # Add slight blue tint (increase blue channel)
    blue_boost = random.uniform(0.0, 0.2)
    img[:, :, 0] = np.clip(img[:, :, 0] + blue_boost, 0.0, 1.0)

    img = (img * 255.0).astype(np.uint8)
    return img


def fog_effect(frame):
    """
    Add a washed-out, low-contrast effect to simulate fog.
    """
    h, w, _ = frame.shape
    img = frame.astype(np.float32) / 255.0

    # Overlay semi-transparent white layer
    fog_strength = random.uniform(0.3, 0.6)
    white = np.ones_like(img)
    img = img * (1.0 - fog_strength) + white * fog_strength

    # Slight blur to soften edges
    ksize = random.choice([7, 9, 11])
    img = cv2.GaussianBlur(img, (ksize, ksize), 0)

    img = np.clip(img * 255.0, 0, 255).astype(np.uint8)
    return img


def rain_effect(frame):
    """
    Draw slanted bright streaks to simulate rain.
    """
    img = frame.copy()
    h, w, _ = img.shape

    num_drops = random.randint(300, 600)

    for _ in range(num_drops):
        # random start position
        x = random.randint(0, w - 1)
        y = random.randint(0, h - 1)

        length = random.randint(10, 25)
        thickness = 1

        # Slanted line (diagonal)
        x2 = x + random.randint(-3, 3)
        y2 = y + length

        color = (random.randint(200, 255),
                 random.randint(200, 255),
                 random.randint(200, 255))

        cv2.line(img, (x, y), (x2, y2), color, thickness)

    # Slight motion blur
    ksize = random.choice([3, 5])
    img = cv2.GaussianBlur(img, (ksize, ksize), 0)

    return img


def random_augment(frame):
    """
    Apply a random combination of night / fog / rain, or none.
    """
    img = frame.copy()
    # Decide which effects to apply
    effects = []

    if random.random() < 0.5:
        effects.append("night")
    if random.random() < 0.5:
        effects.append("fog")
    if random.random() < 0.5:
        effects.append("rain")

    # If no effect selected, sometimes leave as is, sometimes force one
    if not effects:
        if random.random() < 0.5:
            return img
        effects.append(random.choice(["night", "fog", "rain"]))

    for e in effects:
        if e == "night":
            img = night_effect(img)
        elif e == "fog":
            img = fog_effect(img)
        elif e == "rain":
            img = rain_effect(img)

    return img


# --------- Demo / sanity-check runner --------- #

def demo_augmentations():
    print("[INFO] Opening video:", VIDEO_PATH)
    cap = cv2.VideoCapture(str(VIDEO_PATH))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {VIDEO_PATH}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"[INFO] FPS={fps}, total_frames={total_frames}")

    # Sample N frames evenly across the video
    num_samples = 5
    indices = np.linspace(0, total_frames - 1, num_samples, dtype=int)

    print(f"[INFO] Sampling frames at indices: {indices}")

    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            print(f"[WARN] Could not read frame at index {idx}")
            continue

        # Save original
        base_name = f"frame_{idx}"
        cv2.imwrite(str(OUT_DIR / f"{base_name}_orig.jpg"), frame)

        # Night
        night = night_effect(frame)
        cv2.imwrite(str(OUT_DIR / f"{base_name}_night.jpg"), night)

        # Fog
        fog = fog_effect(frame)
        cv2.imwrite(str(OUT_DIR / f"{base_name}_fog.jpg"), fog)

        # Rain
        rain = rain_effect(frame)
        cv2.imwrite(str(OUT_DIR / f"{base_name}_rain.jpg"), rain)

        # Random combo
        rand = random_augment(frame)
        cv2.imwrite(str(OUT_DIR / f"{base_name}_random.jpg"), rand)

    cap.release()
    print("[SUCCESS] Saved augmented sample frames to:", OUT_DIR)


if __name__ == "__main__":
    demo_augmentations()
