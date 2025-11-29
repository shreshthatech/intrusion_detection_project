"""
Task 46: Change-point detection in RF spectrum occupancy.

We:
  - Generate synthetic RF power timeline (normal + abnormal bursts)
  - Compute sliding window averages
  - Detect sudden jumps in RF energy
  - Print detected change-points
"""

import numpy as np
from pathlib import Path


def gen_rf_power_sequence(T=600, mode="normal"):
    """
    Creates a fake RF occupancy timeline.
    Values represent average RF power each second.
    """
    power = 0.5 + 0.05 * np.random.randn(T)

    if mode == "abnormal":
        # Insert 3 abnormal jumps ≈ jammer starting
        for _ in range(3):
            idx = np.random.randint(200, T-50)
            power[idx:idx+50] += np.random.uniform(1.0, 2.0)

    return power.astype(np.float32)


def detect_changes(power, window=20, threshold=0.4):
    """
    Identify abrupt power changes using sliding window mean difference.
    """
    diffs = []
    change_points = []

    for i in range(window, len(power)):
        prev_mean = power[i-window:i].mean()
        if abs(power[i] - prev_mean) > threshold:
            change_points.append(i)
            diffs.append(power[i] - prev_mean)

    return change_points, diffs


def _demo():
    print("[DEMO] RF Change-Point Detection (Task 46)")

    # generate modes
    normal = gen_rf_power_sequence(mode="normal")
    abnormal = gen_rf_power_sequence(mode="abnormal")

    # detect
    cp_n, diff_n = detect_changes(normal)
    cp_a, diff_a = detect_changes(abnormal)

    print("\nNormal sequence ->")
    print(f"  Detected change-points: {len(cp_n)} (expected small ~0)")

    print("\nAbnormal sequence ->")
    print(f"  Detected change-points: {len(cp_a)} (expected > 0)")
    for i in range(min(10, len(cp_a))):
        print(f"  t={cp_a[i]}  delta={diff_a[i]:.3f}")

    print("\n[DEMO] Done.")


if __name__ == "__main__":
    _demo()
