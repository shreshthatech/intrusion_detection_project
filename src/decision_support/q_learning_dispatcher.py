"""
Task 57: Q-learning dispatcher
State = (incident_zone_idx, u1_zone_idx, u2_zone_idx, u3_zone_idx)
Action = pick which patrol unit to dispatch (0=U1, 1=U2, 2=U3)
Reward = -travel_time  (lower response time gives higher reward)
"""

import numpy as np
import random
from dataclasses import dataclass
from typing import List, Dict, Tuple
from pathlib import Path


# -----------------------------------------------------------
# Simple zone graph with travel times (same as before)
# -----------------------------------------------------------
class ZoneGraph:
    def __init__(self):
        self.zones = ["gate", "lobby", "corridor", "server_room", "parking", "perimeter_fence"]
        self.zone_to_idx = {z:i for i,z in enumerate(self.zones)}

        # adjacency list with travel times
        self.adj = {
            "gate": [("lobby", 12)],
            "lobby": [("gate",12), ("corridor",8), ("parking",10)],
            "corridor": [("lobby",8), ("server_room",15)],
            "server_room": [("corridor",15)],
            "parking": [("lobby",10), ("perimeter_fence",20)],
            "perimeter_fence": [("parking",20)]
        }

    def shortest_path_cost(self, start: str, end: str) -> float:
        """Dijkstra for travel time only."""
        import heapq
        pq = [(0, start)]
        visited = set()

        while pq:
            cost, zone = heapq.heappop(pq)
            if zone in visited:
                continue
            visited.add(zone)
            if zone == end:
                return cost
            for nxt, w in self.adj.get(zone, []):
                if nxt not in visited:
                    heapq.heappush(pq, (cost + w, nxt))

        return 9999.0


# -----------------------------------------------------------
# Patrol units + incidents
# -----------------------------------------------------------
@dataclass
class PatrolUnit:
    unit_id: str
    current_zone: str


@dataclass
class Incident:
    zone: str


# -----------------------------------------------------------
# Q-learning Dispatcher
# -----------------------------------------------------------
class QLearningDispatcher:
    def __init__(self, graph: ZoneGraph, units: List[PatrolUnit]):
        self.graph = graph
        self.units = units

        self.n_zones = len(graph.zones)
        self.n_units = len(units)

        # Q-table shape:
        # (incident_zone, u1_zone, u2_zone, u3_zone, action)
        self.Q = np.zeros((self.n_zones, self.n_zones, self.n_zones, self.n_zones, self.n_units))

        # hyperparameters
        self.alpha = 0.2
        self.gamma = 0.9
        self.epsilon = 0.2  # exploration

    def encode_state(self, incident: Incident, units: List[PatrolUnit]):
        return (
            self.graph.zone_to_idx[incident.zone],
            self.graph.zone_to_idx[units[0].current_zone],
            self.graph.zone_to_idx[units[1].current_zone],
            self.graph.zone_to_idx[units[2].current_zone],
        )

    def pick_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, self.n_units - 1)
        return int(np.argmax(self.Q[state]))

    def reward(self, chosen_unit_idx: int, incident: Incident):
        u = self.units[chosen_unit_idx]
        travel_time = self.graph.shortest_path_cost(u.current_zone, incident.zone)
        return -travel_time  # faster = higher reward

    def train(self, episodes=300):
        zones = self.graph.zones

        for ep in range(episodes):
            # randomize unit placements
            for u in self.units:
                u.current_zone = random.choice(zones)

            # random incident
            inc = Incident(zone=random.choice(zones))

            # encode state
            s = self.encode_state(inc, self.units)

            # pick action
            a = self.pick_action(s)

            # compute reward
            r = self.reward(a, inc)

            # next state (units don't move in this simplified version)
            s2 = s

            # Q-learning update
            old_q = self.Q[s][a]
            best_next = np.max(self.Q[s2])
            self.Q[s][a] = old_q + self.alpha * (r + self.gamma * best_next - old_q)

        print("[INFO] Q-learning training complete.")

    def dispatch(self, incident: Incident):
        state = self.encode_state(incident, self.units)
        action = int(np.argmax(self.Q[state]))
        return self.units[action]


# -----------------------------------------------------------
# Demo for Task 57
# -----------------------------------------------------------
def _demo():
    print("[DEMO] Task 57 – Q-learning dispatcher")

    g = ZoneGraph()
    units = [
        PatrolUnit("U1", "gate"),
        PatrolUnit("U2", "parking"),
        PatrolUnit("U3", "corridor"),
    ]

    agent = QLearningDispatcher(g, units)

    print("[INFO] Training Q-table…")
    agent.train(episodes=500)

    # test on 3 example incidents
    test_incidents = [
        Incident("server_room"),
        Incident("lobby"),
        Incident("perimeter_fence"),
    ]

    for inc in test_incidents:
        chosen = agent.dispatch(inc)
        ttime = g.shortest_path_cost(chosen.current_zone, inc.zone)
        print(f"  Incident at {inc.zone}: dispatch {chosen.unit_id} (ETT={ttime}s)")

    print("[DEMO] Task 57 completed.")


if __name__ == "__main__":
    _demo()
