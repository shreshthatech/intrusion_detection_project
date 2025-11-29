"""
Task 53: Incident priority queue with aging and escalation.

Features:
  - Insert incident with base priority (risk)
  - Priority increases over time (aging)
  - If priority passes escalation threshold → escalate flag
  - Pop highest-priority incident first
"""

import heapq
import time
from dataclasses import dataclass, field
from typing import Optional
from pathlib import Path
import json


@dataclass(order=True)
class PrioritizedIncident:
    priority: float
    timestamp: float
    event_id: str = field(compare=False)
    zone: str = field(compare=False)
    base_risk: float = field(compare=False)
    escalated: bool = field(default=False, compare=False)


class IncidentPriorityQueue:
    def __init__(self, aging_rate: float = 0.02, escalation_threshold: float = 1.2):
        """
        aging_rate: priority increases by this *per second*
        escalation_threshold: once priority > threshold → escalate
        """
        self.queue = []
        self.aging_rate = aging_rate
        self.escalation_threshold = escalation_threshold

    def add_incident(self, event_id: str, risk: float, zone: str):
        now = time.time()
        initial_priority = risk  # base priority = risk score

        incident = PrioritizedIncident(
            priority=initial_priority,
            timestamp=now,
            event_id=event_id,
            zone=zone,
            base_risk=risk,
        )

        heapq.heappush(self.queue, incident)

    def _apply_aging(self, inc: PrioritizedIncident):
        """Increase priority based on time since arrival."""
        now = time.time()
        age = now - inc.timestamp
        aged_priority = inc.base_risk + age * self.aging_rate

        # Escalation rule
        if aged_priority > self.escalation_threshold:
            inc.escalated = True

        inc.priority = aged_priority

    def pop_highest(self) -> Optional[PrioritizedIncident]:
        """Remove and return incident with highest aged priority."""
        if not self.queue:
            return None

        # Update priorities of all before popping
        for inc in self.queue:
            self._apply_aging(inc)

        # Convert to max-heap pop (default is min-heap)
        self.queue.sort(reverse=True)
        top = self.queue.pop(0)
        return top


# ----------------- Demo ----------------- #

def _demo():
    print("[DEMO] Task 53 – Priority Queue with Aging & Escalation")

    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    log_path = PROJECT_ROOT / "data" / "processed" / "decision_support" / "priority_queue_demo.jsonl"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    pq = IncidentPriorityQueue(aging_rate=0.03, escalation_threshold=1.15)

    # Add some incidents
    pq.add_incident("e1", risk=0.60, zone="parking")
    pq.add_incident("e2", risk=1.10, zone="gate")
    pq.add_incident("e3", risk=0.45, zone="corridor")

    time.sleep(2)   # allow aging to change priority

    # Pop them in order of priority
    popped = []
    for i in range(3):
        inc = pq.pop_highest()
        if inc:
            popped.append(inc)
            print(f"  POP event={inc.event_id}, zone={inc.zone}, priority={inc.priority:.3f}, escalated={inc.escalated}")
            time.sleep(1)

    # Save demo log
    with open(log_path, "w") as f:
        for inc in popped:
            f.write(json.dumps({
                "event": inc.event_id,
                "zone": inc.zone,
                "priority": inc.priority,
                "escalated": inc.escalated
            }) + "\n")

    print("[DEMO] Task 53 completed.")


if __name__ == "__main__":
    _demo()
