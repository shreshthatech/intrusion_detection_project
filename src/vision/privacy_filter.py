"""
Task 19: Privacy filter – blur faces (and later plates).

We:
  - read synthetic_intrusion.mp4
  - detect faces with Haar cascade
  - blur those regions
  - write a new video with blurred faces
  - save a few sample blurred frames
"""

from pathlib import Path
import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]

VIDEO_IN = PROJECT_ROOT / "data" / "processed" / "synthetic_intrusion.mp4"
VIDEO_OUT = PROJECT_ROOT / "data" / "processed" / "synthetic_intrusion_privacy.mp4"

SAMPLES_DIR = PROJECT_ROOT / "data" / "processed" / "privacy_samples"
SAMPLES_DIR.mkdir(parents=True, exist_ok=True)

HAAR_DIR = PROJECT_ROOT / "models" / "vision" / "haarcascades"
FACE_CASCADE_PATH = HAAR_DIR / "haarcascade_frontalface_default.xml"


def load_face_detector():
    if not FACE_CASCADE_PATH.exists():
        raise FileNotFoundError(
            f"Face cascade not found at {FACE_CASCADE_PATH}\n"
            "Please download 'haarcascade_frontalface_default.xml' "
            "and place it in that folder."
        )
    face_cascade = cv2.CascadeClassifier(str(FACE_CASCADE_PATH))
    return face_cascade


def blur_region(img, x, y, w, h, ksize=25):
    roi = img[y:y+h, x:x+w]
    if roi.size == 0:
        return img
    blurred = cv2.GaussianBlur(roi, (ksize | 1, ksize | 1), 0)
    img[y:y+h, x:x+w] = blurred
    return img


def run_privacy_filter():
    face_cascade = load_face_detector()

    print("[INFO] Opening input video:", VIDEO_IN)
    cap = cv2.VideoCapture(str(VIDEO_IN))
    if not cap.isOpened():
        raise RuntimeError("Could not open input video.")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"[INFO] FPS={fps:.2f}, size=({width}x{height}), frames={total_frames}")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(VIDEO_OUT), fourcc, fps, (width, height))

    frame_idx = 0
    sample_saved = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces (tweak scaleFactor/minNeighbors if needed)
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )

        # Blur each face
        for (x, y, w, h) in faces:
            frame = blur_region(frame, x, y, w, h, ksize=35)

        out.write(frame)

        # Save some sample frames with blur
        if sample_saved < 5 and len(faces) > 0:
            cv2.imwrite(str(SAMPLES_DIR / f"privacy_{frame_idx:04d}.jpg"), frame)
            sample_saved += 1

        frame_idx += 1

    cap.release()
    out.release()

    print("[SUCCESS] Privacy-filtered video saved to:", VIDEO_OUT)
    print("[SUCCESS] Sample blurred frames saved to:", SAMPLES_DIR)


if __name__ == "__main__":
    run_privacy_filter()
