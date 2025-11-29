"""
Task 52: SLA-aware alert throttling (avoid alert storms)

This module enforces:
  - max_alerts_per_minute
  - burst_limit (alerts allowed instantly)
  - cooldown_after_burst

We simulate incoming events with a risk score (from Task 51).
"""

import time
from pathlib import Path
import json
from typing import List


class AlertThrottle:
    def __init__(self,
                 max_alerts_per_minute: int = 5,
                 burst_limit: int = 2,
                 cooldown_seconds: int = 10):
        """
        max_alerts_per_minute → SLA limit
        burst_limit → how many alerts can fire instantly
        cooldown_seconds → if burst exceeded, wait this long
        """
        self.max_alerts_per_minute = max_alerts_per_minute
        self.burst_limit = burst_limit
        self.cooldown_seconds = cooldown_seconds

        self.alert_timestamps: List[float] = []   # store timestamps of fired alerts
        self.last_burst_time = 0

    def _prune_old(self):
        """Keep only alerts within last 60 seconds."""
        now = time.time()
        self.alert_timestamps = [t for t in self.alert_timestamps if (now - t) < 60]

    def can_fire(self):
        now = time.time()
        self._prune_old()

        # 1) BURST CONTROL
        if len(self.alert_timestamps) < self.burst_limit:
            return True, "burst-ok"

        # Burst exceeded -> apply cooldown
        if now - self.last_burst_time < self.cooldown_seconds:
            return False, "cooldown"

        # 2) SLA / per-minute control
        if len(self.alert_timestamps) >= self.max_alerts_per_minute:
            return False, "sla-limit"

        return True, "allowed"

    def fire_alert(self, event_id: str, risk: float):
        """
        Try firing alert. Returns decision + reason.
        """
        allowed, reason = self.can_fire()

        if allowed:
            now = time.time()
            self.alert_timestamps.append(now)
            self.last_burst_time = now

            return True, f"ALERT-FIRED event={event_id} risk={risk:.2f} reason={reason}"
        else:
            return False, f"SUPPRESSED event={event_id} risk={risk:.2f} reason={reason}"


# --------------- Demo ---------------- #

def _demo():
    print("[DEMO] Task 52 – SLA-aware alert throttling")

    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    log_path = PROJECT_ROOT / "data" / "processed" / "decision_support" / "alert_throttle_demo.jsonl"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    throttle = AlertThrottle(
        max_alerts_per_minute=5,
        burst_limit=2,
        cooldown_seconds=5
    )

    # Fake incoming high-risk events
    test_events = [
        ("e1", 1.20),
        ("e2", 1.10),
        ("e3", 1.30),
        ("e4", 0.95),
        ("e5", 1.40),
        ("e6", 1.10),
    ]

    with open(log_path, "w") as f:
        for event_id, risk in test_events:
            ok, msg = throttle.fire_alert(event_id, risk)
            print(f"  {msg}")
            f.write(msg + "\n")
            time.sleep(1)    # simulate events arriving quickly

    print("[DEMO] Task 52 completed.")


if __name__ == "__main__":
    _demo()
