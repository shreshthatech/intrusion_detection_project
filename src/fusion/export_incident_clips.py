"""
Task 50: Export acoustic/RF incident clips around flagged events.

We:
  - Generate demo long-duration audio + RF signals.
  - Define some incident timestamps (seconds).
  - Slice small windows around each event.
  - Save:
      - audio clips as WAV
      - RF clips as NPZ
"""

from pathlib import Path
import numpy as np
import soundfile as sf


PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_AUDIO_DIR = PROJECT_ROOT / "data" / "raw" / "audio"
PROCESSED_AUDIO_INCIDENT_DIR = PROJECT_ROOT / "data" / "processed" / "audio" / "incidents"
PROCESSED_RF_INCIDENT_DIR = PROJECT_ROOT / "data" / "processed" / "rf" / "incidents"

RAW_AUDIO_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_AUDIO_INCIDENT_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_RF_INCIDENT_DIR.mkdir(parents=True, exist_ok=True)


# --------- Demo signal generators --------- #

def generate_demo_audio(duration_sec=20.0, sr=16000):
    """
    Generate a fake ambient audio with occasional 'events' (louder bumps).
    """
    t = np.linspace(0, duration_sec, int(sr * duration_sec), endpoint=False)
    # background hum
    y = 0.01 * np.random.randn(len(t)) + 0.02 * np.sin(2 * np.pi * 60 * t)

    # add a few louder transient events
    for center in [5.0, 11.0, 16.0]:
        idx = int(center * sr)
        w = int(0.3 * sr)
        start = max(0, idx - w // 2)
        end = min(len(y), idx + w // 2)
        y[start:end] += 0.2 * np.hanning(end - start)

    return y.astype(np.float32), sr


def generate_demo_rf(duration_sec=20.0, sr_rf=8000):
    """
    Generate a fake RF IQ magnitude signal (1D power-like).
    """
    t = np.linspace(0, duration_sec, int(sr_rf * duration_sec), endpoint=False)
    x = 0.2 * np.random.randn(len(t))

    # occasional strong RF bursts
    for center in [6.0, 12.0, 17.5]:
        idx = int(center * sr_rf)
        w = int(0.5 * sr_rf)
        start = max(0, idx - w // 2)
        end = min(len(x), idx + w // 2)
        x[start:end] += 1.5 * np.random.randn(end - start)

    return x.astype(np.float32), sr_rf


# --------- Clip extraction --------- #

def extract_clip(signal, sr, center_time, window_before, window_after):
    """
    signal: 1D numpy array
    sr: sample rate
    center_time: event time in seconds
    window_before/after: seconds before/after event
    """
    N = len(signal)
    center_idx = int(center_time * sr)
    start_idx = max(0, int((center_time - window_before) * sr))
    end_idx = min(N, int((center_time + window_after) * sr))

    clip = signal[start_idx:end_idx]
    rel_start = (start_idx - center_idx) / sr
    rel_end = (end_idx - center_idx) / sr

    return clip, rel_start, rel_end


def _demo():
    print("[DEMO] Export Incident Clips (Task 50)")

    # 1) Generate demo signals
    audio, sr_audio = generate_demo_audio(duration_sec=20.0, sr=16000)
    rf_sig, sr_rf = generate_demo_rf(duration_sec=20.0, sr_rf=8000)

    print(f"[INFO] Audio length: {len(audio)/sr_audio:.2f} sec, RF length: {len(rf_sig)/sr_rf:.2f} sec")

    # Optionally save full demo audio to raw dir
    full_audio_path = RAW_AUDIO_DIR / "demo_full_incidents.wav"
    sf.write(full_audio_path, audio, sr_audio)
    print(f"[INFO] Full demo audio saved to: {full_audio_path}")

    # 2) Define incident timestamps (e.g., from fusion layer)
    incident_times = [5.0, 11.0, 16.0]  # seconds, matching where we injected events

    audio_window_before = 1.0
    audio_window_after = 1.0
    rf_window_before = 0.5
    rf_window_after = 0.5

    # 3) Extract & save clips
    for idx, t_evt in enumerate(incident_times):
        # Audio clip
        a_clip, a_rel_start, a_rel_end = extract_clip(
            audio, sr_audio, t_evt, audio_window_before, audio_window_after
        )
        audio_clip_path = PROCESSED_AUDIO_INCIDENT_DIR / f"audio_incident_{idx+1}.wav"
        sf.write(audio_clip_path, a_clip, sr_audio)

        # RF clip
        rf_clip, r_rel_start, r_rel_end = extract_clip(
            rf_sig, sr_rf, t_evt, rf_window_before, rf_window_after
        )
        rf_clip_path = PROCESSED_RF_INCIDENT_DIR / f"rf_incident_{idx+1}.npz"
        np.savez_compressed(
            rf_clip_path,
            rf_clip=rf_clip,
            sr_rf=sr_rf,
            center_time=t_evt,
            rel_start=r_rel_start,
            rel_end=r_rel_end,
        )

        print(f"\n[INCIDENT {idx+1}] t={t_evt:.2f} sec")
        print(f"  Audio clip: {audio_clip_path}  (len={len(a_clip)/sr_audio:.2f} sec)")
        print(f"  RF clip:    {rf_clip_path}  (len={len(rf_clip)/sr_rf:.2f} sec)")

    print("\n[SUCCESS] Exported all incident clips.")
    print("[DEMO] Done.")


if __name__ == "__main__":
    _demo()
