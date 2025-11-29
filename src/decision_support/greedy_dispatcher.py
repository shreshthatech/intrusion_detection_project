"""
Task 56: Greedy dispatcher for nearest-available patrol.

We:
  - Reuse a facility ZoneGraph
  - Represent patrol agents with current zones
  - Given incidents (zone + risk), assign nearest available agent
"""

from dataclasses import dataclass
from typing import List, Dict, Tuple
from pathlib import Path
import heapq
import json


# --- ZoneGraph (same as before) --- #

class ZoneGraph:
    def __init__(self):
        self.adjacency: Dict[str, List[Tuple[str, float]]] = {}

    def add_edge(self, a: str, b: str, cost: float):
        self.adjacency.setdefault(a, [])
        self.adjacency.setdefault(b, [])
        self.adjacency[a].append((b, cost))
        self.adjacency[b].append((a, cost))

    def shortest_path(self, start: str, end: str):
        pq = [(0.0, start, [])]
        visited = set()

        while pq:
            cost, node, path = heapq.heappop(pq)
            if node in visited:
                continue
            visited.add(node)
            path = path + [node]
            if node == end:
                return cost, path
            for neighbor, w in self.adjacency.get(node, []):
                if neighbor not in visited:
                    heapq.heappush(pq, (cost + w, neighbor, path))
        return float("inf"), []


# --- Data structures --- #

@dataclass
class PatrolUnit:
    unit_id: str
    current_zone: str
    available: bool = True


@dataclass
class Incident:
    incident_id: str
    zone: str
    risk: float


class GreedyDispatcher:
    def __init__(self, graph: ZoneGraph, units: List[PatrolUnit]):
        self.graph = graph
        self.units = units

    def dispatch(self, incident: Incident):
        """
        Pick nearest available patrol unit based on travel time.

        Returns (unit, travel_time, path) or (None, inf, []) if no unit.
        """
        best_unit = None
        best_time = float("inf")
        best_path = []

        for u in self.units:
            if not u.available:
                continue
            cost, path = self.graph.shortest_path(u.current_zone, incident.zone)
            if cost < best_time:
                best_time = cost
                best_path = path
                best_unit = u

        return best_unit, best_time, best_path


# --- Demo --- #

def _demo():
    print("[DEMO] Task 56 – Greedy dispatcher")

    # Build graph
    g = ZoneGraph()
    g.add_edge("gate", "lobby", 12)
    g.add_edge("lobby", "corridor", 8)
    g.add_edge("corridor", "server_room", 15)
    g.add_edge("lobby", "parking", 10)
    g.add_edge("parking", "perimeter_fence", 20)

    # Units
    units = [
        PatrolUnit(unit_id="U1", current_zone="gate"),
        PatrolUnit(unit_id="U2", current_zone="parking"),
        PatrolUnit(unit_id="U3", current_zone="corridor"),
    ]

    dispatcher = GreedyDispatcher(g, units)

    # Some incidents
    incidents = [
        Incident(incident_id="I1", zone="server_room", risk=1.4),
        Incident(incident_id="I2", zone="perimeter_fence", risk=1.1),
        Incident(incident_id="I3", zone="lobby", risk=0.8),
    ]

    assignments = []

    for inc in incidents:
        unit, ttime, path = dispatcher.dispatch(inc)
        if unit is None:
            print(f"  Incident {inc.incident_id} at {inc.zone}: NO UNIT AVAILABLE")
            continue

        print(f"  Incident {inc.incident_id} at {inc.zone}: assign {unit.unit_id} (ETT={ttime:.1f}s, path={path})")
        assignments.append({
            "incident_id": inc.incident_id,
            "zone": inc.zone,
            "unit_id": unit.unit_id,
            "travel_time": ttime,
            "path": path,
        })

        # Optionally mark unit busy (to simulate one-at-a-time)
        unit.available = False

    # Save demo
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    out_path = PROJECT_ROOT / "data" / "processed" / "decision_support" / "greedy_dispatch_demo.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(assignments, f, indent=2)

    print("[DEMO] Task 56 completed.")


if __name__ == "__main__":
    _demo()
