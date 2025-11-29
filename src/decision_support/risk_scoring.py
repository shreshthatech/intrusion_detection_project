"""
Task 51: Risk scoring function (fused score + zone criticality).

We define:
  - zone criticality lookup
  - risk score = fused_score * zone_weight * context multipliers

This will later be used by alerting, queues, and dispatch logic.
"""

from dataclasses import dataclass
from typing import Optional
from pathlib import Path
import json
import time


# ----------------- Config & data structures ----------------- #

ZONE_WEIGHTS = {
    "gate": 1.2,
    "parking": 1.0,
    "corridor": 1.0,
    "lobby": 1.1,
    "server_room": 1.6,
    "control_room": 1.5,
    "perimeter_fence": 1.3,
    "unknown": 1.0,
}


@dataclass
class Event:
    """
    Minimal event structure used for risk scoring.
    """
    event_id: str
    fused_score: float        # 0..1 from fusion pipeline
    zone: str                 # e.g. "gate", "server_room"
    time_of_day: Optional[int] = None  # hour 0..23


def time_of_day_multiplier(hour: int) -> float:
    """
    Simple heuristic:
      - Night hours (22-6) -> higher risk
      - Daytime -> normal
    """
    if hour is None:
        return 1.0

    if hour >= 22 or hour < 6:
        return 1.2
    if 6 <= hour < 9:
        return 1.05
    if 18 <= hour < 22:
        return 1.1
    return 1.0


def compute_risk(event: Event) -> float:
    """
    Core risk scoring function.

    risk = fused_score * zone_weight * time_weight
    Clamped to [0, 1.5] for safety.
    """
    base = max(0.0, min(1.0, event.fused_score))

    zone = event.zone if event.zone in ZONE_WEIGHTS else "unknown"
    z_weight = ZONE_WEIGHTS[zone]

    hour = event.time_of_day if event.time_of_day is not None else time.localtime().tm_hour
    t_weight = time_of_day_multiplier(hour)

    risk = base * z_weight * t_weight

    # optional mild boost for very high fused scores
    if base > 0.8:
        risk *= 1.1

    # clamp
    risk = max(0.0, min(1.5, risk))
    return risk


# Optional: log risk scores to a file for inspection
def log_risk(event: Event, risk: float, project_root: Path):
    log_dir = project_root / "data" / "processed" / "decision_support"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "risk_scores.jsonl"

    record = {
        "event_id": event.event_id,
        "fused_score": event.fused_score,
        "zone": event.zone,
        "time_of_day": event.time_of_day,
        "risk": risk,
        "ts": time.time(),
    }

    with open(log_path, "a") as f:
        f.write(json.dumps(record) + "\n")


# ----------------- Demo ----------------- #

def _demo():
    print("[DEMO] Task 51 – Risk Scoring")

    project_root = Path(__file__).resolve().parents[2]

    events = [
        Event(event_id="e1", fused_score=0.65, zone="parking", time_of_day=14),
        Event(event_id="e2", fused_score=0.82, zone="gate", time_of_day=23),
        Event(event_id="e3", fused_score=0.40, zone="corridor", time_of_day=3),
        Event(event_id="e4", fused_score=0.95, zone="server_room", time_of_day=2),
    ]

    for ev in events:
        r = compute_risk(ev)
        log_risk(ev, r, project_root)
        print(f"  Event {ev.event_id}: fused={ev.fused_score:.2f}, zone={ev.zone}, hour={ev.time_of_day} -> risk={r:.3f}")

    print("\n[DEMO] Task 51 completed.")


if __name__ == "__main__":
    _demo()
