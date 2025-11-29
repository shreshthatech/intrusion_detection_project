"""
Task 48:
Confidence arbitration when sensors disagree.

Rules:
  - If 2+ sensors agree → boosted confidence.
  - If only vision is high but audio/RF disagree → mild reduction.
  - If audio+RF show anomaly but vision low → trust audio/RF.
  - If strong conflict → mark as 'uncertain'.

Output:
  final_score, reason, dominant_sensor
"""

import random


def arbitrate(vision, audio_flag, rf_flag):
    """
    Returns:
      final_score: float
      reason: str
      dominant_sensor: str
    """

    # Count how many sensors vote "threat"
    votes = int(vision > 0.55) + int(audio_flag) + int(rf_flag)

    # Case 1: Strong agreement (2 or 3 sensors)
    if votes >= 2:
        final = min(vision + 0.20, 1.0)
        reason = "multi-sensor agreement"
        dominant = "fusion"
        return final, reason, dominant

    # Case 2: Only vision confident
    if vision > 0.65 and audio_flag == 0 and rf_flag == 0:
        final = vision - 0.05  # slight penalty
        reason = "vision-only, low acoustic/RF support"
        dominant = "vision"
        return final, reason, dominant

    # Case 3: Vision low, but audio+RF both suspicious
    if vision < 0.45 and audio_flag and rf_flag:
        final = 0.75
        reason = "audio+RF override vision"
        dominant = "audio-rf"
        return final, reason, dominant

    # Case 4: All sensors conflicting → uncertain
    final = 0.50
    reason = "sensor disagreement"
    dominant = "none"
    return final, reason, dominant


def _demo():
    print("[DEMO] Confidence Arbitration (Task 48)")

    for i in range(6):
        vision = round(random.uniform(0.3, 0.9), 2)
        audio_flag = random.choice([0, 1])
        rf_flag = random.choice([0, 1])

        final, reason, dominant = arbitrate(vision, audio_flag, rf_flag)

        print(f"\nEvent {i}:")
        print(f"  vision={vision}  audio={audio_flag}  rf={rf_flag}")
        print(f"  ---> final_score={final:.3f}")
        print(f"  reason={reason}")
        print(f"  dominant={dominant}")

    print("\n[DEMO] Done.")


if __name__ == "__main__":
    _demo()
