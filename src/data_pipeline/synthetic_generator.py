"""
Task 2: Synthetic Intrusion Event Generator

This script takes a normal CCTV video and adds a fake intruder
(a moving colored rectangle) to simulate intrusion events.
"""

import cv2
import numpy as np
from pathlib import Path
import random


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_VIDEO = PROJECT_ROOT / "data" / "raw" / "video" / "demo.mp4"
OUTPUT_VIDEO = PROJECT_ROOT / "data" / "processed" / "synthetic_intrusion.mp4"
LABELS_PATH = PROJECT_ROOT / "data" / "processed" / "synthetic_labels.txt"


def generate_synthetic_intruder(frame, x, y, w=60, h=120):
    """
    Draws a fake intruder as a moving colored rectangle.
    """
    color = (0, 0, 255)  # RED intruder
    thickness = -1       # filled rectangle

    cv2.rectangle(frame, (x, y), (x + w, y + h), color, thickness)
    return frame


def generate_synthetic_video():
    print("[INFO] Loading video:", RAW_VIDEO)

    cap = cv2.VideoCapture(str(RAW_VIDEO))
    if not cap.isOpened():
        raise RuntimeError("Could not open raw video.")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out = cv2.VideoWriter(
        str(OUTPUT_VIDEO),
        cv2.VideoWriter_fourcc(*'mp4v'),
        fps,
        (width, height)
    )

    # Random starting position
    intruder_x = random.randint(50, width - 150)
    intruder_y = random.randint(50, height - 200)

    # Random movement speed
    dx = random.choice([-3, -2, 2, 3])
    dy = random.choice([-3, -2, 2, 3])

    labels = []  # store intruder (x,y) per frame

    frame_id = 0

    print("[INFO] Generating synthetic intrusion video...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Move intruder
        intruder_x += dx
        intruder_y += dy

        # Boundary check
        if intruder_x < 0 or intruder_x + 60 > width:
            dx = -dx
        if intruder_y < 0 or intruder_y + 120 > height:
            dy = -dy

        # Draw fake intruder
        frame = generate_synthetic_intruder(frame, intruder_x, intruder_y)

        # Save label
        labels.append(f"{frame_id},{intruder_x},{intruder_y},60,120")

        # Write frame
        out.write(frame)
        frame_id += 1

    cap.release()
    out.release()

    print("[INFO] Saving label file to:", LABELS_PATH)
    with open(LABELS_PATH, "w") as f:
        f.write("\n".join(labels))

    print("[SUCCESS] Synthetic intrusion video created!")
    print("         Video:", OUTPUT_VIDEO)
    print("         Labels:", LABELS_PATH)


if __name__ == "__main__":
    generate_synthetic_video()
