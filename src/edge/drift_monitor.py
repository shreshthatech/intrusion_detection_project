"""
Task 69: Drift Monitor (KS-test + PSI) with auto-recalibration hook.

This monitor:
  - Loads reference distribution (baseline)
  - Computes KS test
  - Computes PSI
  - Decides drift severity
  - Logs results
"""

import numpy as np
import json
import time
from pathlib import Path
from scipy.stats import ks_2samp


def compute_psi(expected, actual, buckets=10):
    """
    Population Stability Index.
    PSI < 0.1     = no drift
    PSI < 0.25    = moderate drift
    PSI >= 0.25   = significant drift
    """
    expected = np.array(expected)
    actual = np.array(actual)

    # Avoid zero divide
    def safe_pct(x):
        return 0.0001 if x == 0 else x

    psi = 0
    quantiles = np.linspace(0, 100, buckets + 1)

    expected_bins = np.percentile(expected, quantiles)
    actual_bins = np.percentile(actual, quantiles)

    for i in range(buckets):
        e_low = expected_bins[i]
        e_high = expected_bins[i+1]

        a_low = actual_bins[i]
        a_high = actual_bins[i+1]

        e_pct = safe_pct(np.mean((expected >= e_low) & (expected < e_high)))
        a_pct = safe_pct(np.mean((actual >= a_low) & (actual < a_high)))

        psi += (e_pct - a_pct) * np.log(e_pct / a_pct)

    return psi


class DriftMonitor:
    def __init__(self, ref_data, project_root):
        self.ref_data = np.array(ref_data)
        self.project_root = project_root

        self.log_path = project_root / "data" / "processed" / "edge_metrics" / "drift_log.jsonl"
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def auto_recalibrate(self):
        # In a real system this might:
        # - retrain model
        # - update thresholds
        # - reload configurations
        print("[AUTO] Recalibration triggered! (simulated)")

    def run(self, new_data):
        new_data = np.array(new_data)

        # KS TEST
        ks_stat, ks_p = ks_2samp(self.ref_data, new_data)

        # PSI
        psi = compute_psi(self.ref_data, new_data)

        drift_level = "none"
        if psi >= 0.25 or ks_p < 0.05:
            drift_level = "major"
            self.auto_recalibrate()
        elif psi >= 0.1:
            drift_level = "moderate"
        else:
            drift_level = "none"

        # Log
        record = {
            "timestamp": time.time(),
            "ks_stat": float(ks_stat),
            "ks_p_value": float(ks_p),
            "psi": float(psi),
            "drift_level": drift_level,
        }

        with open(self.log_path, "a") as f:
            f.write(json.dumps(record) + "\n")

        print(f"[DEMO] KS={ks_stat:.4f}, p={ks_p:.4f}, PSI={psi:.4f} → Drift={drift_level}")

        return drift_level


def demo():
    print("[DEMO] Task 69 — Drift Monitor Demo")

    PROJECT_ROOT = Path(__file__).resolve().parents[2]

    # Reference distribution (normal behavior)
    ref = np.random.normal(0.0, 1.0, size=2000)

    # Simulated NEW data (shifted mean = drift)
    new = np.random.normal(0.8, 1.0, size=2000)

    monitor = DriftMonitor(ref, PROJECT_ROOT)
    monitor.run(new)

    print("[DEMO] Task 69 completed.")


if __name__ == "__main__":
    demo()
