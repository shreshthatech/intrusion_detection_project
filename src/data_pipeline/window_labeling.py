"""
Task 5: Window-level label smoothing + weak labeling.

We will:
1. Load sliding windows from Task 4.
2. Load synthetic labels (frame-level intrusions).
3. Convert frame_id -> timestamp using video FPS.
4. For each window, check if any intrusion occurs inside it.
5. Create smoothed labels.
6. Save final window-level labels to JSONL.
"""

from pathlib import Path
import json


PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Input files
WINDOWS_FILE = PROJECT_ROOT / "data" / "processed" / "windows" / "sliding_windows.jsonl"
LABELS_FILE = PROJECT_ROOT / "data" / "processed" / "synthetic_labels.txt"

# Output file
OUT_FILE = PROJECT_ROOT / "data" / "processed" / "windows" / "window_labels.jsonl"

# FPS of your synthetic video (same as original)
DEFAULT_FPS = 30.0   # adjust if your video has different FPS


def load_frame_labels():
    """
    Loads synthetic intruder labels from synthetic_labels.txt
    Format per line:
    frame_id,x,y,w,h
    """
    frame_to_time = {}
    with open(LABELS_FILE, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(",")
            frame_id = int(parts[0])
            timestamp = frame_id / DEFAULT_FPS
            frame_to_time[frame_id] = timestamp
    return frame_to_time


def load_windows():
    windows = []
    with open(WINDOWS_FILE, "r") as f:
        for line in f:
            windows.append(json.loads(line))
    return windows


def label_windows():
    print("[INFO] Loading frame-level intrusion timestamps...")
    frame_ts = load_frame_labels()

    print("[INFO] Loading window definitions...")
    windows = load_windows()

    print("[INFO] Labeling windows...")

    out_lines = []
    for w in windows:
        start_t = w["start_time"]
        end_t = w["end_time"]

        # Determine if this window contains any intruder
        intrusion = False
        for frame_id, t in frame_ts.items():
            if start_t <= t < end_t:
                intrusion = True
                break

        # Hard label
        label = 1 if intrusion else 0

        # Smoothed label
        label_smoothed = 0.9 if label == 1 else 0.1

        out_obj = {
            "window_id": w["window_id"],
            "start_time": start_t,
            "end_time": end_t,
            "label": label,
            "label_smoothed": label_smoothed
        }

        out_lines.append(out_obj)

    print(f"[INFO] Saving {len(out_lines)} labeled windows to {OUT_FILE}")
    with open(OUT_FILE, "w") as f:
        for item in out_lines:
            f.write(json.dumps(item) + "\n")

    print("[SUCCESS] Window labeling completed.")


if __name__ == "__main__":
    label_windows()
