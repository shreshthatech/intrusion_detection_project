"""
Task 54: Facility graph – zones as nodes, paths with travel-time costs.

Features:
  - Graph of facility zones
  - Weighted edges = travel time (seconds)
  - Dijkstra shortest path
  - Demo: compute route + travel cost
"""

import heapq
from dataclasses import dataclass
from typing import Dict, List, Tuple
from pathlib import Path
import json


@dataclass
class ZoneGraph:
    adjacency: Dict[str, List[Tuple[str, float]]]  # zone -> [(neighbor, cost), ...]

    def add_edge(self, a: str, b: str, cost: float):
        """Add bidirectional edge."""
        self.adjacency.setdefault(a, [])
        self.adjacency.setdefault(b, [])
        self.adjacency[a].append((b, cost))
        self.adjacency[b].append((a, cost))

    def shortest_path(self, start: str, end: str):
        """
        Dijkstra shortest path.
        Returns (total_cost, [path]).
        """
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


def _demo():
    print("[DEMO] Task 54 – Facility Graph")

    # Build small example layout
    graph = ZoneGraph(adjacency={})

    graph.add_edge("gate", "lobby", 12)
    graph.add_edge("lobby", "corridor", 8)
    graph.add_edge("corridor", "server_room", 15)
    graph.add_edge("lobby", "parking", 10)
    graph.add_edge("parking", "perimeter_fence", 20)

    # Demo path:
    start = "gate"
    end = "server_room"

    cost, path = graph.shortest_path(start, end)

    print(f"  Shortest path from {start} -> {end}:")
    print(f"    path = {path}")
    print(f"    travel_time = {cost} sec")

    # Save to file (optional)
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    out_path = PROJECT_ROOT / "data" / "processed" / "decision_support" / "facility_graph_demo.json"

    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w") as f:
        json.dump({"path": path, "cost": cost}, f, indent=2)

    print("[DEMO] Task 54 completed.")


if __name__ == "__main__":
    _demo()
