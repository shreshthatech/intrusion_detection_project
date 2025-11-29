"""
Task 41: Audio feature extractor (MFCC, spectral rolloff, RMS).

This script:
  - Looks for a WAV file in data/raw/audio
  - If none found, it generates a demo sine-wave file.
  - Extracts MFCCs, spectral rolloff, RMS.
  - Saves features to NPZ and summary stats to JSON.

Paths:
  RAW:      data/raw/audio
  PROCESSED: data/processed/audio
"""

from pathlib import Path
import json
import numpy as np
import librosa
import soundfile as sf


# ---------------- Paths ---------------- #

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_AUDIO_DIR = PROJECT_ROOT / "data" / "raw" / "audio"
PROCESSED_AUDIO_DIR = PROJECT_ROOT / "data" / "processed" / "audio"

RAW_AUDIO_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_AUDIO_DIR.mkdir(parents=True, exist_ok=True)


# ---------------- Core Feature Extraction ---------------- #

def extract_audio_features(
    audio_path: Path,
    sr: int = 16000,
    n_mfcc: int = 13,
    hop_length: int = 512,
) -> dict:
    """
    Load audio and compute MFCCs, spectral rolloff, RMS.
    Returns a dict with arrays and summary stats.
    """
    print(f"[INFO] Loading audio: {audio_path}")
    y, sr = librosa.load(audio_path, sr=sr, mono=True)
    print(f"[INFO] Audio loaded: {len(y)} samples @ {sr} Hz")

    # MFCCs: shape (n_mfcc, T)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, hop_length=hop_length)

    # Spectral rolloff: shape (1, T)
    rolloff = librosa.feature.spectral_rolloff(
        y=y, sr=sr, hop_length=hop_length, roll_percent=0.85
    )

    # RMS energy: shape (1, T)
    rms = librosa.feature.rms(y=y, hop_length=hop_length)

    # Time axis (in seconds) for frames
    times = librosa.frames_to_time(
        np.arange(mfcc.shape[1]), sr=sr, hop_length=hop_length
    )

    # Summary stats
    summary = {
        "sr": sr,
        "n_samples": int(len(y)),
        "duration_sec": float(len(y) / sr),
        "n_frames": int(mfcc.shape[1]),
        "mfcc_mean": mfcc.mean(axis=1).tolist(),
        "mfcc_std": mfcc.std(axis=1).tolist(),
        "rolloff_mean": float(rolloff.mean()),
        "rolloff_std": float(rolloff.std()),
        "rms_mean": float(rms.mean()),
        "rms_std": float(rms.std()),
    }

    features = {
        "audio_path": str(audio_path),
        "sr": sr,
        "waveform": y,
        "times": times,
        "mfcc": mfcc,
        "rolloff": rolloff,
        "rms": rms,
        "summary": summary,
    }

    return features


def save_features(features: dict, base_name: str):
    """
    Save features to:
      - NPZ (arrays)
      - JSON (summary stats)
    """
    npz_path = PROCESSED_AUDIO_DIR / f"{base_name}_features.npz"
    json_path = PROCESSED_AUDIO_DIR / f"{base_name}_summary.json"

    print(f"[INFO] Saving arrays to: {npz_path}")
    np.savez_compressed(
        npz_path,
        waveform=features["waveform"],
        times=features["times"],
        mfcc=features["mfcc"],
        rolloff=features["rolloff"],
        rms=features["rms"],
        sr=features["sr"],
        audio_path=features["audio_path"],
    )

    print(f"[INFO] Saving summary to: {json_path}")
    with open(json_path, "w") as f:
        json.dump(features["summary"], f, indent=2)

    print("[SUCCESS] Audio features saved.")


# ---------------- Demo Helper ---------------- #

def ensure_demo_audio() -> Path:
    """
    If there is no WAV file in RAW_AUDIO_DIR, generate a demo sine-wave file.
    Returns the path to a WAV file.
    """
    wav_files = list(RAW_AUDIO_DIR.glob("*.wav"))
    if wav_files:
        print(f"[INFO] Found existing WAV file: {wav_files[0]}")
        return wav_files[0]

    print("[WARN] No WAV files found in data/raw/audio. Generating demo sine wave...")
    sr = 16000
    duration = 2.0
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    freq = 440.0  # A4
    y = 0.2 * np.sin(2 * np.pi * freq * t)

    demo_path = RAW_AUDIO_DIR / "demo_beep.wav"
    sf.write(demo_path, y, sr)
    print(f"[INFO] Demo audio saved to: {demo_path}")
    return demo_path


# ---------------- DEMO ---------------- #

def _demo():
    print("[DEMO] Audio feature extraction (Task 41)")

    audio_path = ensure_demo_audio()
    features = extract_audio_features(audio_path)
    base_name = audio_path.stem

    save_features(features, base_name)

    # Print a snippet of summary
    summary = features["summary"]
    print("\n[SUMMARY]")
    print(f"  duration_sec = {summary['duration_sec']:.2f}")
    print(f"  n_frames     = {summary['n_frames']}")
    print(f"  mfcc_mean[0] = {summary['mfcc_mean'][0]:.4f}")
    print(f"  rolloff_mean = {summary['rolloff_mean']:.2f}")
    print(f"  rms_mean     = {summary['rms_mean']:.6f}")

    print("\n[DEMO] Done.")


if __name__ == "__main__":
    _demo()
