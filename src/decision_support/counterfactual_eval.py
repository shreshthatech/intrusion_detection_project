import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[2] / "src"))

"""
Task 59 – Counterfactual Evaluation
Compare 3 policies:
  - Greedy dispatcher  (Task 56)
  - Q-learning dispatcher (Task 57)
  - DQN dispatcher (Task 58)
"""

import random
import numpy as np
from dataclasses import dataclass
from typing import List
import json

from decision_support.greedy_dispatcher import PatrolUnit, Incident, GreedyDispatcher
from decision_support.q_learning_dispatcher import QLearningDispatcher
from decision_support.dqn_dispatcher import DQNDispatcher


# ------------------------------------------------------------
# Unified facility graph for all 3 policies
# ------------------------------------------------------------
class FacilityGraph:
    def __init__(self):
        self.zones = ["gate", "lobby", "corridor", "server_room", "parking", "perimeter_fence"]
        self.zone_to_idx = {z: i for i, z in enumerate(self.zones)}

        # adjacency: zone -> [(neighbor, cost)]
        self.adj = {
            "gate": [("lobby", 12)],
            "lobby": [("gate", 12), ("corridor", 8), ("parking", 10)],
            "corridor": [("lobby", 8), ("server_room", 15)],
            "server_room": [("corridor", 15)],
            "parking": [("lobby", 10), ("perimeter_fence", 20)],
            "perimeter_fence": [("parking", 20)],
        }

    def shortest_path_cost(self, start: str, end: str) -> float:
        """Dijkstra for travel time."""
        import heapq
        pq = [(0.0, start)]
        visited = set()

        while pq:
            cost, z = heapq.heappop(pq)
            if z in visited:
                continue
            visited.add(z)
            if z == end:
                return cost
            for nxt, w in self.adj.get(z, []):
                if nxt not in visited:
                    heapq.heappush(pq, (cost + w, nxt))
        return 9999.0

    def shortest_path(self, start: str, end: str):
        """Dijkstra that also returns the path (for GreedyDispatcher)."""
        import heapq
        pq = [(0.0, start, [])]
        visited = set()

        while pq:
            cost, z, path = heapq.heappop(pq)
            if z in visited:
                continue
            visited.add(z)
            path = path + [z]
            if z == end:
                return cost, path
            for nxt, w in self.adj.get(z, []):
                if nxt not in visited:
                    heapq.heappush(pq, (cost + w, nxt, path))

        return 9999.0, []


# ------------------------------------------------------------
# Utility: evaluate a policy on multiple incidents
# ------------------------------------------------------------
def evaluate_policy(policy_name, dispatcher, graph, incidents):
    results = []

    for inc in incidents:
        out = dispatcher.dispatch(inc)

        # Some dispatchers return (unit, time, path),
        # others return just the PatrolUnit.
        if isinstance(out, tuple):
            chosen = out[0]      # first element is PatrolUnit
        else:
            chosen = out

        travel = graph.shortest_path_cost(chosen.current_zone, inc.zone)
        results.append(travel)

    results = np.array(results)
    summary = {
        "policy": policy_name,
        "mean_travel_time": float(results.mean()),
        "median": float(np.median(results)),
        "min": float(results.min()),
        "max": float(results.max()),
        "scores": results.tolist(),
    }

    return summary


    return summary


# ------------------------------------------------------------
# MAIN DEMO FOR TASK 59
# ------------------------------------------------------------
def _demo():
    print("[DEMO] Task 59 – Counterfactual Evaluation")

    # Unified graph
    g = FacilityGraph()

    # create synthetic incidents
    zones = g.zones
    test_incidents = [
    Incident(
        incident_id=f"I{i}",
        zone=random.choice(zones),
        risk=random.uniform(0.5, 1.5)
    )
    for i in range(20)
]


    # --- Instantiate policies with identical starting units ---
    greedy = GreedyDispatcher(
        g,
        [PatrolUnit("U1", "gate"), PatrolUnit("U2", "parking"), PatrolUnit("U3", "corridor")]
    )

    qlearn = QLearningDispatcher(
        g,
        [PatrolUnit("U1", "gate"), PatrolUnit("U2", "parking"), PatrolUnit("U3", "corridor")]
    )
    qlearn.train(episodes=400)

    dqn = DQNDispatcher(
        g,
        [PatrolUnit("U1", "gate"), PatrolUnit("U2", "parking"), PatrolUnit("U3", "corridor")]
    )
    dqn.train(episodes=600)

    # --- Evaluate each policy on the same incidents ---
    greedy_res = evaluate_policy("Greedy", greedy, g, test_incidents)
    q_res = evaluate_policy("Q-Learning", qlearn, g, test_incidents)
    dqn_res = evaluate_policy("DQN", dqn, g, test_incidents)

    all_results = [greedy_res, q_res, dqn_res]

    print("\n=== POLICY COMPARISON ===")
    for r in all_results:
        print(f"\nPolicy = {r['policy']}")
        print(f"  mean travel time = {r['mean_travel_time']:.2f}")
        print(f"  median           = {r['median']:.2f}")
        print(f"  min              = {r['min']:.2f}")
        print(f"  max              = {r['max']:.2f}")

    # Save results
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    out = PROJECT_ROOT / "data" / "processed" / "decision_support" / "counterfactual_eval.json"
    out.parent.mkdir(parents=True, exist_ok=True)

    with open(out, "w") as f:
        json.dump(all_results, f, indent=2)

    print("\n[SUCCESS] Task 59 completed.")
    print(f"[INFO] Results saved to: {out}")


if __name__ == "__main__":
    _demo()

