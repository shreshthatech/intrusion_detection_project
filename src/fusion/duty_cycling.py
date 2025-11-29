"""
Task 49: Low-power duty cycling policy for microphone + RF.

We simulate:
  - Vision activity (boolean)
  - Audio anomaly flag (0/1)
  - RF abnormal flag (0/1)

Policy:
  - If no sensor shows activity → switch to LOW POWER
  - If any sensor shows activity → switch to HIGH POWER

Output:
  - mode: "low_power" or "high_power"
  - sampling_interval: seconds between audio/RF reads
"""

import random


def duty_cycle_policy(vision_active, audio_flag, rf_flag):
    """
    Returns:
      mode (str)
      sampling_interval (float)
    """

    if not vision_active and audio_flag == 0 and rf_flag == 0:
        # all quiet → energy saving mode
        return "low_power", 5.0   # sample every 5 seconds

    # any activity → full monitoring
    return "high_power", 0.0     # continuous sampling


def _demo():
    print("[DEMO] Duty Cycling (Task 49)")

    for i in range(6):
        vision_active = random.choice([True, False])
        audio_flag = random.choice([0, 1])
        rf_flag = random.choice([0, 1])

        mode, interval = duty_cycle_policy(vision_active, audio_flag, rf_flag)

        print(f"\nEvent {i}:")
        print(f"  vision_active={vision_active}  audio={audio_flag}  rf={rf_flag}")
        print(f"  ---> mode={mode}, sampling_interval={interval}")

    print("\n[DEMO] Done.")


if __name__ == "__main__":
    _demo()
