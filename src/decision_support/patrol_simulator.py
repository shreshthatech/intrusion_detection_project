"""
Task 55: Multi-agent patrol simulator (stochastic travel times).

We reuse the ZoneGraph concept (Task 54) and simulate:

  - Multiple patrol agents on the graph.
  - Each agent has a current zone and a route (list of zones).
  - Travel times along edges are random around a base cost.
"""

import random
from dataclasses import dataclass, field
from typing import List, Dict, Tuple
from pathlib import Path
import json
from math import inf
import heapq


# --------- Graph (copy from facility_graph with minimal code) --------- #

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
        return inf, []


# --------- Patrol agent + simulator --------- #

@dataclass
class PatrolAgent:
    agent_id: str
    current_zone: str
    route: List[str] = field(default_factory=list)
    eta_to_next: float = 0.0   # time remaining to reach next zone
    path_index: int = 0        # index in route

    def assign_route(self, route: List[str], base_edge_time: float = 0.0):
        self.route = route
        self.path_index = 0
        if len(route) > 1:
            # initial ETA for first leg
            self.eta_to_next = base_edge_time
        else:
            self.eta_to_next = 0.0


class PatrolSimulator:
    def __init__(self, graph: ZoneGraph, agents: List[PatrolAgent]):
        self.graph = graph
        self.agents = agents
        self.time = 0.0
        self.log: List[Dict] = []

    def _sample_travel_time(self, base_cost: float) -> float:
        """
        Sample stochastic travel time:
        normal around base_cost with ±20% noise.
        """
        noise = random.uniform(0.8, 1.2)
        return base_cost * noise

    def step(self, dt: float = 1.0):
        """
        Advance simulation by dt seconds.
        Move agents along their routes.
        """
        self.time += dt

        for ag in self.agents:
            # If no route or at end, skip
            if not ag.route or ag.path_index >= len(ag.route) - 1:
                continue

            ag.eta_to_next -= dt

            if ag.eta_to_next <= 0:
                # Arrived at next zone
                ag.path_index += 1
                ag.current_zone = ag.route[ag.path_index]

                # Log arrival
                self.log.append({
                    "time": self.time,
                    "agent_id": ag.agent_id,
                    "zone": ag.current_zone,
                    "event": "arrive"
                })

                # If more legs left, set new ETA with sampled travel time
                if ag.path_index < len(ag.route) - 1:
                    a = ag.route[ag.path_index]
                    b = ag.route[ag.path_index + 1]
                    # find cost
                    base_cost = None
                    for nbr, w in self.graph.adjacency.get(a, []):
                        if nbr == b:
                            base_cost = w
                            break
                    if base_cost is None:
                        base_cost = 10.0  # fallback
                    ag.eta_to_next = self._sample_travel_time(base_cost)

    def run_until_done(self, max_time: float = 300.0, dt: float = 1.0):
        """
        Run until all agents reach end of route or time exceeded.
        """
        while self.time < max_time:
            active = False
            for ag in self.agents:
                if ag.route and ag.path_index < len(ag.route) - 1:
                    active = True
                    break
            if not active:
                break
            self.step(dt=dt)


# --------- Demo for Task 55 --------- #

def _demo():
    print("[DEMO] Task 55 – Multi-agent patrol simulator")

    # Build graph (same zones as Task 54)
    g = ZoneGraph()
    g.add_edge("gate", "lobby", 12)
    g.add_edge("lobby", "corridor", 8)
    g.add_edge("corridor", "server_room", 15)
    g.add_edge("lobby", "parking", 10)
    g.add_edge("parking", "perimeter_fence", 20)

    # Two agents starting at different zones
    agents = [
        PatrolAgent(agent_id="A1", current_zone="gate"),
        PatrolAgent(agent_id="A2", current_zone="parking"),
    ]

    # Assign them routes to server_room and perimeter_fence
    # using shortest path
    cost1, path1 = g.shortest_path("gate", "server_room")
    cost2, path2 = g.shortest_path("parking", "perimeter_fence")

    agents[0].assign_route(path1, base_edge_time=5.0)
    agents[1].assign_route(path2, base_edge_time=5.0)

    sim = PatrolSimulator(g, agents)
    sim.run_until_done(max_time=200.0, dt=1.0)

    # Print summary
    for log_e in sim.log[:10]:
        print(f"  t={log_e['time']:.1f}s agent={log_e['agent_id']} arrived at {log_e['zone']}")

    # Save log
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    out_path = PROJECT_ROOT / "data" / "processed" / "decision_support" / "patrol_sim_demo.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w") as f:
        json.dump(sim.log, f, indent=2)

    print("[DEMO] Task 55 completed.")


if __name__ == "__main__":
    _demo()
