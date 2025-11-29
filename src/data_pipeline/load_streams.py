# src/data_pipeline/load_streams.py

"""
Task 1: Load and validate synchronized sensor streams.

This script does 4 things:
1. Loads a sample video file.
2. Loads a sample audio file.
3. Loads a sample RF log (CSV).
4. Optionally loads a thermal video (if present).
5. Prints a summary + basic validation checks.

For now, we are NOT doing any ML. Just loading + checking.
"""

from pathlib import Path
import cv2
import librosa
import numpy as np
import pandas as pd


# ---------- Paths & config ----------

# This file is in: src/data_pipeline/load_streams.py
# parents[0] -> data_pipeline
# parents[1] -> src
# parents[2] -> project root (intrusion_detection_project)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"

VIDEO_PATH = RAW_DIR / "video" / "demo.mp4"
AUDIO_PATH = RAW_DIR / "audio" / "demo.wav"
RF_PATH = RAW_DIR / "rf" / "demo.csv"
THERMAL_PATH = RAW_DIR / "thermal" / "demo.mp4"  # optional


# ---------- Loader functions ----------

def load_video(path: Path) -> dict:
    """Load video metadata (fps, frame count, duration, size)."""
    if not path.exists():
        raise FileNotFoundError(f"Video file not found: {path}")

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise IOError(f"Failed to open video file: {path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    duration_sec = frame_count / fps if fps and fps > 0 else None

    cap.release()

    return {
        "path": path,
        "fps": fps,
        "frame_count": frame_count,
        "width": width,
        "height": height,
        "duration_sec": duration_sec,
    }


def load_audio(path: Path) -> dict:
    """Load audio metadata (sr, duration, channels)."""
    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {path}")

    # mono=False so we keep channels if stereo
    audio, sr = librosa.load(path, sr=None, mono=False)

    if audio.ndim == 1:
        # mono: shape (samples,)
        samples = audio.shape[0]
        channels = 1
    else:
        # stereo or more: shape (channels, samples)
        channels, samples = audio.shape

    duration_sec = samples / sr if sr and sr > 0 else None

    return {
        "path": path,
        "sample_rate": sr,
        "channels": channels,
        "samples": samples,
        "duration_sec": duration_sec,
    }


def load_rf_log(path: Path) -> dict:
    """Load RF CSV log and do basic checks."""
    if not path.exists():
        raise FileNotFoundError(f"RF log file not found: {path}")

    df = pd.read_csv(path)

    # Expected columns (for now we keep it simple)
    expected_cols = {"timestamp", "freq_mhz", "power_dbm"}
    missing_cols = expected_cols - set(df.columns)
    if missing_cols:
        print(f"[WARN] RF log is missing expected columns: {missing_cols}")

    n_rows = len(df)
    n_cols = len(df.columns)
    n_missing = int(df.isna().sum().sum())

    return {
        "path": path,
        "n_rows": n_rows,
        "n_cols": n_cols,
        "n_missing": n_missing,
        "columns": list(df.columns),
        "head": df.head(5),
    }


def load_thermal(path: Path) -> dict | None:
    """
    Load thermal video metadata. If file does not exist, return None.

    Later we will treat thermal differently (like grayscale images),
    but for Task 1 metadata is enough.
    """
    if not path.exists():
        print(f"[INFO] Thermal file not found, skipping: {path}")
        return None

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        print(f"[WARN] Failed to open thermal video: {path}")
        return None

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration_sec = frame_count / fps if fps and fps > 0 else None

    cap.release()

    return {
        "path": path,
        "fps": fps,
        "frame_count": frame_count,
        "width": width,
        "height": height,
        "duration_sec": duration_sec,
    }


# ---------- Validation logic ----------

def validate_durations(video_meta: dict, audio_meta: dict, thermal_meta: dict | None, rf_meta: dict):
    """Check if durations are roughly aligned (within a few seconds)."""

    durations = []

    if video_meta.get("duration_sec") is not None:
        durations.append(("video", video_meta["duration_sec"]))
    if audio_meta.get("duration_sec") is not None:
        durations.append(("audio", audio_meta["duration_sec"]))
    if thermal_meta and thermal_meta.get("duration_sec") is not None:
        durations.append(("thermal", thermal_meta["duration_sec"]))

    # RF logs don't always have exact duration; we infer from last timestamp
    # if 'timestamp' column exists
    rf_duration = None
    if "timestamp" in rf_meta.get("columns", []):
        # use RF head to get a sense of timestamp range (simple version)
        # better: read full column, but this is enough for Task 1
        rf_df_head = rf_meta["head"]
        # Not accurate duration – just a demo if you use small RF logs.
        # Later we will refine this if needed.
        try:
            # Try to infer from full file
            full_df = pd.read_csv(rf_meta["path"])
            t_min = full_df["timestamp"].min()
            t_max = full_df["timestamp"].max()
            rf_duration = float(t_max) - float(t_min)
            durations.append(("rf", rf_duration))
        except Exception as e:
            print(f"[WARN] Could not infer RF duration: {e}")

    if not durations:
        print("[WARN] No durations available to compare.")
        return

    # Find min and max durations
    names, values = zip(*durations)
    min_dur = min(values)
    max_dur = max(values)
    diff = max_dur - min_dur

    print("\n[INFO] Duration comparison (seconds):")
    for name, value in durations:
        print(f"  - {name}: {value:.2f} s")

    if diff > 3.0:
        print(f"[WARN] Streams differ in duration by more than 3 seconds (diff={diff:.2f} s)")
    else:
        print(f"[OK] Streams are roughly aligned in duration (diff={diff:.2f} s)")


# ---------- Main entry point ----------

def main():
    print("=== Task 1: Load & validate sensor streams ===\n")

    print(f"[INFO] Project root: {PROJECT_ROOT}")
    print(f"[INFO] Using video:   {VIDEO_PATH}")
    print(f"[INFO] Using audio:   {AUDIO_PATH}")
    print(f"[INFO] Using RF log:  {RF_PATH}")
    print(f"[INFO] Thermal (opt): {THERMAL_PATH}\n")

    # Load each stream
    video_meta = load_video(VIDEO_PATH)
    print("[OK] Loaded video.")
    print(f"     FPS = {video_meta['fps']}")
    print(f"     Frames = {video_meta['frame_count']}")
    print(f"     Size = {video_meta['width']}x{video_meta['height']}")
    print(f"     Duration ≈ {video_meta['duration_sec']:.2f} s\n")

    audio_meta = load_audio(AUDIO_PATH)
    print("[OK] Loaded audio.")
    print(f"     Sample rate = {audio_meta['sample_rate']} Hz")
    print(f"     Channels = {audio_meta['channels']}")
    print(f"     Samples = {audio_meta['samples']}")
    print(f"     Duration ≈ {audio_meta['duration_sec']:.2f} s\n")

    rf_meta = load_rf_log(RF_PATH)
    print("[OK] Loaded RF log.")
    print(f"     Rows = {rf_meta['n_rows']}")
    print(f"     Cols = {rf_meta['n_cols']}")
    print(f"     Missing values = {rf_meta['n_missing']}")
    print(f"     Columns = {rf_meta['columns']}\n")
    print("     Head of RF log:")
    print(rf_meta["head"])
    print()

    thermal_meta = load_thermal(THERMAL_PATH)
    if thermal_meta is not None:
        print("[OK] Loaded thermal.")
        print(f"     FPS = {thermal_meta['fps']}")
        print(f"     Frames = {thermal_meta['frame_count']}")
        print(f"     Size = {thermal_meta['width']}x{thermal_meta['height']}")
        print(f"     Duration ≈ {thermal_meta['duration_sec']:.2f} s\n")

    # Validate durations (rough synchronization)
    validate_durations(video_meta, audio_meta, thermal_meta, rf_meta)

    print("\n=== Task 1 completed: basic load & validate works. ===")


if __name__ == "__main__":
    main()
