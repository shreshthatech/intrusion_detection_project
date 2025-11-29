"""
Task 60 – What-If Planner (Sensor Outage + Weather Stress Test)

Simulates system response under:
  - camera/audio/RF outages
  - bad weather (fog, rain, night)
Produces:
  - expected detection drop %
  - expected response delay
  - recommended fallback strategy
"""

import random
import json
from pathlib import Path
from dataclasses import dataclass


@dataclass
class Scenario:
    name: str
    camera_ok: bool
    audio_ok: bool
    rf_ok: bool
    weather: str   # clear, fog, rain, night


# --------------------------------------------------------------------
# Effect models
# --------------------------------------------------------------------
def detection_drop(camera_ok, audio_ok, rf_ok, weather):
    """Return expected % drop in detection quality."""
    drop = 0.0

    if not camera_ok:
        drop += 40
    if not audio_ok:
        drop += 20
    if not rf_ok:
        drop += 15

    if weather == "fog":
        drop += 25
    elif weather == "rain":
        drop += 15
    elif weather == "night":
        drop += 20

    return min(drop, 95.0)


def response_delay(weather):
    """How much slower agents move."""
    if weather == "clear":
        return 0.0
    if weather == "rain":
        return 1.2
    if weather == "fog":
        return 1.4
    if weather == "night":
        return 1.5
    return 1.0


def fallback_strategy(camera_ok, audio_ok, rf_ok):
    if not camera_ok and not audio_ok and not rf_ok:
        return "Switch to perimeter-only patrols + manual escalation."
    if not camera_ok:
        return "Use thermal + RF + acoustic triangulation."
    if not audio_ok:
        return "Prioritise vision + RF confidence."
    if not rf_ok:
        return "Rely on vision/acoustics + geofence rules."
    return "All sensors normal."


# --------------------------------------------------------------------
# Main planning function
# --------------------------------------------------------------------
def run_planner():
    print("[DEMO] Task 60 – What-If Planner")

    scenarios = [
        Scenario("Camera outage + fog", False, True, True, "fog"),
        Scenario("Audio outage + rain", True, False, True, "rain"),
        Scenario("RF outage + night", True, True, False, "night"),
        Scenario("All sensors OK – rain", True, True, True, "rain"),
        Scenario("Total outage – night", False, False, False, "night"),
    ]

    results = []

    for s in scenarios:
        drop = detection_drop(s.camera_ok, s.audio_ok, s.rf_ok, s.weather)
        delay = response_delay(s.weather)
        strategy = fallback_strategy(s.camera_ok, s.audio_ok, s.rf_ok)

        entry = {
            "scenario": s.name,
            "camera_ok": s.camera_ok,
            "audio_ok": s.audio_ok,
            "rf_ok": s.rf_ok,
            "weather": s.weather,
            "detection_drop_percent": drop,
            "response_delay_factor": delay,
            "recommended_strategy": strategy,
        }

        results.append(entry)

        print(f"\nScenario: {s.name}")
        print(f"  Detection drop: {drop}%")
        print(f"  Response delay ×{delay}")
        print(f"  Strategy: {strategy}")

    # Save
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    out = PROJECT_ROOT / "data" / "processed" / "decision_support" / "what_if_report.json"
    out.parent.mkdir(parents=True, exist_ok=True)

    with open(out, "w") as f:
        json.dump(results, f, indent=2)

    print("\n[SUCCESS] Task 60 completed.")
    print(f"[INFO] Report saved to: {out}")


if __name__ == "__main__":
    run_planner()