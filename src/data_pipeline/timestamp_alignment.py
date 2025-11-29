"""
Task 3: Timestamp Alignment with Drift Correction

This script:
1. Reads video FPS to compute video timestamps.
2. Reads audio duration & sampling rate.
3. Reads RF timestamps from CSV.
4. Calculates offset differences.
5. Aligns all timestamps so video starts at 0.0s.
6. Saves aligned timestamps in JSON format.

"""

import json
import pandas as pd
from pathlib import Path
import cv2
import librosa


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
OUT_DIR = PROJECT_ROOT / "data" / "processed" / "aligned"
OUT_DIR.mkdir(parents=True, exist_ok=True)

VIDEO_PATH = RAW_DIR / "video" / "demo.mp4"
AUDIO_PATH = RAW_DIR / "audio" / "demo.wav"
RF_PATH = RAW_DIR / "rf" / "demo.csv"


def get_video_timestamps(path: Path):
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    timestamps = [i / fps for i in range(frames)]
    cap.release()

    return timestamps, fps, frames


def get_audio_timestamps(path: Path):
    audio, sr = librosa.load(path, sr=None, mono=False)

    if audio.ndim == 1:
        samples = audio.shape[0]
    else:
        samples = audio.shape[1]

    duration = samples / sr
    timestamps = [i / sr for i in range(samples)]

    return timestamps, sr, samples, duration


def get_rf_timestamps(path: Path):
    df = pd.read_csv(path)
    if "timestamp" not in df.columns:
        raise ValueError("RF CSV must contain 'timestamp' column")

    return df["timestamp"].tolist(), df


def align_timestamps():
    print("[INFO] Loading timestamps from sensors...")

    video_ts, fps, frames = get_video_timestamps(VIDEO_PATH)
    audio_ts, sr, samples, audio_dur = get_audio_timestamps(AUDIO_PATH)
    rf_ts, rf_df = get_rf_timestamps(RF_PATH)

    # Sensor start times
    video_start = video_ts[0]               # should be 0.0
    audio_start = audio_ts[0]
    rf_start = rf_ts[0]

    print(f"\nRaw Starts:")
    print(f"  Video start: {video_start:.4f}")
    print(f"  Audio start: {audio_start:.4f}")
    print(f"  RF start:    {rf_start:.4f}\n")

    # Compute offset to align audio and rf to video
    audio_offset = video_start - audio_start
    rf_offset = video_start - rf_start

    print("[INFO] Offsets computed:")
    print(f"  Apply audio_offset = {audio_offset:.4f} seconds")
    print(f"  Apply rf_offset    = {rf_offset:.4f} seconds\n")

    # Apply offsets
    audio_ts_aligned = [t + audio_offset for t in audio_ts]
    rf_ts_aligned = [t + rf_offset for t in rf_ts]

    # Save aligned data
    aligned_data = {
        "video": video_ts,
        "audio": audio_ts_aligned,
        "rf": rf_ts_aligned,
        "fps": fps,
        "audio_sr": sr,
        "rf_rows": len(rf_df)
    }

    out_path = OUT_DIR / "aligned_timestamps.json"
    with open(out_path, "w") as f:
        json.dump(aligned_data, f, indent=2)

    print("[SUCCESS] Aligned timestamps saved to:")
    print("         ", out_path)


if __name__ == "__main__":
    align_timestamps()
