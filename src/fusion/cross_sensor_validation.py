"""
Task 47: Cross-validate acoustic & RF cues with vision detections.

We simulate:
  - Vision detection score
  - Audio anomaly flag (0/1)
  - RF abnormal flag (0/1)

We compute:
  - Adjusted threat score based on cross-sensor agreement.
"""

import random
import numpy as np


def fuse_sensors(vision_score, audio_flag, rf_flag):
    """
    Simple rule-based fusion:
      - Start with vision_score
      - Add +0.10 if audio is anomalous
      - Add +0.15 if RF is abnormal
      - If all 3 agree (vision strong + audio + RF), boost more
    """
    score = vision_score

    if audio_flag:
        score += 0.10

    if rf_flag:
        score += 0.15

    # strong multisensor confirmation
    if vision_score > 0.55 and audio_flag and rf_flag:
        score += 0.20

    # clamp 0..1
    return min(score, 1.0)


def _demo():
    print("[DEMO] Cross-Sensor Validation (Task 47)")

    # simulate 6 events
    for i in range(6):
        vision = round(random.uniform(0.3, 0.9), 2)
        audio = random.choice([0, 1])
        rf = random.choice([0, 1])

        fused = fuse_sensors(vision, audio, rf)

        print(f"\nEvent {i}:")
        print(f"  vision={vision}  audio={audio}  rf={rf}")
        print(f"  ---> fused_score={fused:.3f}")

    print("\n[DEMO] Done.")


if __name__ == "__main__":
    _demo()
