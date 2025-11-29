"""
Task 4: Sliding-window dataset builder for streaming inference.

This script:
1. Loads aligned timestamps from Task 3.
2. Builds sliding windows (e.g. 2s window, 1s stride).
3. For each window, finds the index ranges of:
   - video frames
   - audio samples
   - RF rows
4. Saves everything as JSONL: one window per line.
"""

from pathlib import Path
import json
import math


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ALIGNED_PATH = PROJECT_ROOT / "data" / "processed" / "aligned" / "aligned_timestamps.json"
OUT_DIR = PROJECT_ROOT / "data" / "processed" / "windows"
OUT_DIR.mkdir(parents=True, exist_ok=True)

WINDOW_SIZE = 2.0   # seconds
STRIDE = 1.0        # seconds


def _find_index_range(timestamps, start_t, end_t):
    """
    Given a sorted list of timestamps, return (start_idx, end_idx)
    such that timestamps[i] is in [start_t, end_t) for i in that range.
    If no timestamps fall in the window, returns (None, None).
    """
    n = len(timestamps)
    start_idx = None
    end_idx = None

    # Linear scan is fine for now (your data is small).
    for i, t in enumerate(timestamps):
        if t >= start_t and start_idx is None:
            start_idx = i
        if t < end_t:
            end_idx = i
        if t >= end_t:
            break

    if start_idx is None or end_idx is None or start_idx > end_idx:
        return None, None

    return start_idx, end_idx


def build_windows():
    print("[INFO] Loading aligned timestamps from:", ALIGNED_PATH)
    with open(ALIGNED_PATH, "r") as f:
        aligned = json.load(f)

    video_ts = aligned["video"]
    audio_ts = aligned["audio"]
    rf_ts = aligned["rf"]

    last_video_time = video_ts[-1]
    print(f"[INFO] Last video timestamp: {last_video_time:.2f} s")

    windows = []
    window_id = 0
    start_t = 0.0

    print(f"[INFO] Building windows: size={WINDOW_SIZE}s, stride={STRIDE}s")

    while start_t + WINDOW_SIZE <= last_video_time + 1e-6:
        end_t = start_t + WINDOW_SIZE

        v_start, v_end = _find_index_range(video_ts, start_t, end_t)
        a_start, a_end = _find_index_range(audio_ts, start_t, end_t)
        r_start, r_end = _find_index_range(rf_ts, start_t, end_t)

        window = {
            "window_id": window_id,
            "start_time": start_t,
            "end_time": end_t,
            "video_index_start": v_start,
            "video_index_end": v_end,
            "audio_index_start": a_start,
            "audio_index_end": a_end,
            "rf_index_start": r_start,
            "rf_index_end": r_end,
        }
        windows.append(window)

        window_id += 1
        start_t += STRIDE

    out_path = OUT_DIR / "sliding_windows.jsonl"
    print("[INFO] Saving", len(windows), "windows to:", out_path)

    with open(out_path, "w") as f:
        for w in windows:
            f.write(json.dumps(w) + "\n")

    print("[SUCCESS] Sliding-window definition file created.")


if __name__ == "__main__":
    build_windows()
