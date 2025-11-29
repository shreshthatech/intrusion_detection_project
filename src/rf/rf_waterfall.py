"""
Task 44: RF waterfall preprocessor (STFT + log-power spectrum)

We simulate:
  - normal RF: low noise, occasional small peaks
  - abnormal RF: strong bursts (jammer-like)

We compute:
  - STFT
  - log-power waterfall (time-frequency image)

We save:
  - NPZ containing the waterfall
  - Summary statistics
"""

from pathlib import Path
import numpy as np
import json
import scipy.signal as spsig


# ---------------- Paths ---------------- #

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_RF_DIR = PROJECT_ROOT / "data" / "processed" / "rf"
PROCESSED_RF_DIR.mkdir(parents=True, exist_ok=True)


# ---------------- Synthetic RF generator ---------------- #

def gen_rf_signal(T=16000, mode="normal"):
    """
    Generate simple RF-like signals.
    mode="normal": mostly background noise
    mode="abnormal": includes high-power bursts
    """
    t = np.linspace(0, 1.0, T)

    # Background noise
    x = 0.2 * np.random.randn(T)

    # Add some low-power sinusoidal components (normal)
    x += 0.05 * np.sin(2 * np.pi * 500 * t)
    x += 0.05 * np.sin(2 * np.pi * 900 * t)

    if mode == "abnormal":
        # Insert high-energy bursts (jammer-like)
        for _ in range(4):
            start = np.random.randint(0, T - 800)
            x[start:start+800] += np.random.uniform(2.0, 3.0) * np.random.randn(800)

    return x.astype(np.float32)


# ---------------- RF Waterfall ---------------- #

def compute_rf_waterfall(signal, n_fft=256, hop=128):
    """
    Compute STFT and return log-power waterfall matrix.
    """
    f, t, Zxx = spsig.stft(signal, nperseg=n_fft, noverlap=n_fft-hop)
    power = np.abs(Zxx) ** 2
    log_power = np.log(power + 1e-6)  # avoid log(0)

    return f, t, log_power


# ---------------- Save helper ---------------- #

def save_waterfall(log_power, mode):
    base = f"rf_waterfall_{mode}"
    npz_path = PROCESSED_RF_DIR / f"{base}.npz"
    json_path = PROCESSED_RF_DIR / f"{base}_summary.json"

    np.savez_compressed(npz_path, log_power=log_power)

    summary = {
        "log_power_mean": float(log_power.mean()),
        "log_power_std": float(log_power.std()),
        "shape": log_power.shape,
    }

    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"[SUCCESS] Saved RF waterfall to: {npz_path}")
    print(f"[SUCCESS] Summary saved to: {json_path}")


# ---------------- DEMO ---------------- #

def _demo():
    print("[DEMO] RF Waterfall (Task 44)")

    # Normal
    print("[INFO] Generating normal RF...")
    x_normal = gen_rf_signal(mode="normal")
    _, _, Wn = compute_rf_waterfall(x_normal)
    save_waterfall(Wn, "normal")

    # Abnormal
    print("[INFO] Generating abnormal RF (jammer)...")
    x_ab = gen_rf_signal(mode="abnormal")
    _, _, Wa = compute_rf_waterfall(x_ab)
    save_waterfall(Wa, "abnormal")

    print("\n[DEMO] Done.")


if __name__ == "__main__":
    _demo()
