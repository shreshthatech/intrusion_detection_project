"""
Task 58: Deep Q-Network (DQN) dispatcher
State = (incident_zone_idx, u1_zone_idx, u2_zone_idx, u3_zone_idx) → 4 integers
Action = which unit to dispatch (0,1,2)
Reward = -travel_time (fast dispatch = higher reward)
"""

import random
import numpy as np
from dataclasses import dataclass
from typing import List
from pathlib import Path
from collections import deque

import torch
import torch.nn as nn
import torch.optim as optim


# ---------------------------------------------------------
# Zone Graph (same as previous tasks)
# ---------------------------------------------------------
class ZoneGraph:
    def __init__(self):
        self.zones = ["gate", "lobby", "corridor", "server_room", "parking", "perimeter_fence"]
        self.zone_to_idx = {z: i for i, z in enumerate(self.zones)}

        self.adj = {
            "gate": [("lobby", 12)],
            "lobby": [("gate", 12), ("corridor", 8), ("parking", 10)],
            "corridor": [("lobby", 8), ("server_room", 15)],
            "server_room": [("corridor", 15)],
            "parking": [("lobby", 10), ("perimeter_fence", 20)],
            "perimeter_fence": [("parking", 20)]
        }

    def shortest_path_cost(self, start, end):
        import heapq
        pq = [(0, start)]
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


# ---------------------------------------------------------
# Data models
# ---------------------------------------------------------
@dataclass
class PatrolUnit:
    unit_id: str
    current_zone: str


@dataclass
class Incident:
    zone: str


# ---------------------------------------------------------
# DQN Network
# ---------------------------------------------------------
class DQN(nn.Module):
    def __init__(self, n_states=4, n_actions=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_states, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, n_actions),
        )

    def forward(self, x):
        return self.net(x)


# ---------------------------------------------------------
# DQN Dispatcher
# ---------------------------------------------------------
class DQNDispatcher:
    def __init__(self, graph: ZoneGraph, units: List[PatrolUnit]):
        self.graph = graph
        self.units = units

        self.state_dim = 4
        self.n_actions = len(units)

        self.policy = DQN(self.state_dim, self.n_actions)
        self.target = DQN(self.state_dim, self.n_actions)
        self.target.load_state_dict(self.policy.state_dict())

        self.optimizer = optim.Adam(self.policy.parameters(), lr=1e-3)

        self.memory = deque(maxlen=5000)
        self.gamma = 0.95
        self.epsilon = 0.2  # exploration
        self.batch_size = 32

    # ---- Encode state ----
    def encode_state(self, incident: Incident):
        return np.array([
            self.graph.zone_to_idx[incident.zone],
            self.graph.zone_to_idx[self.units[0].current_zone],
            self.graph.zone_to_idx[self.units[1].current_zone],
            self.graph.zone_to_idx[self.units[2].current_zone],
        ], dtype=np.float32)

    def reward(self, unit_idx, incident):
        u = self.units[unit_idx]
        t = self.graph.shortest_path_cost(u.current_zone, incident.zone)
        return -t

    # ---- Choose action ----
    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, self.n_actions - 1)
        s = torch.tensor(state).float().unsqueeze(0)
        qvals = self.policy(s)
        return int(torch.argmax(qvals).item())

    # ---- Replay training ----
    def replay(self):
        if len(self.memory) < self.batch_size:
            return

        batch = random.sample(self.memory, self.batch_size)

        states, actions, rewards, next_states = zip(*batch)
        states = torch.tensor(states).float()
        actions = torch.tensor(actions)
        rewards = torch.tensor(rewards).float()
        next_states = torch.tensor(next_states).float()

        qvals = self.policy(states).gather(1, actions.unsqueeze(1)).squeeze()
        next_qvals = self.target(next_states).max(1)[0]

        target_vals = rewards + self.gamma * next_qvals

        loss = nn.MSELoss()(qvals, target_vals.detach())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    # ---- Train loop ----
    def train(self, episodes=400):
        zones = self.graph.zones

        for ep in range(episodes):
            # Randomize patrol placements
            for u in self.units:
                u.current_zone = random.choice(zones)

            incident = Incident(zone=random.choice(zones))
            state = self.encode_state(incident)

            action = self.select_action(state)
            reward = self.reward(action, incident)
            next_state = self.encode_state(incident)  # simple env

            self.memory.append((state, action, reward, next_state))

            self.replay()

            # Slowly update target network
            if ep % 20 == 0:
                self.target.load_state_dict(self.policy.state_dict())

        print("[INFO] DQN training complete.")

    def dispatch(self, incident):
        s = self.encode_state(incident)
        a = self.select_action(s)
        return self.units[a]


# ---------------------------------------------------------
# Demo
# ---------------------------------------------------------
def _demo():
    print("[DEMO] Task 58 – DQN dispatcher")

    g = ZoneGraph()

    units = [
        PatrolUnit("U1", "gate"),
        PatrolUnit("U2", "parking"),
        PatrolUnit("U3", "corridor"),
    ]

    agent = DQNDispatcher(g, units)

    print("[INFO] Training DQN…")
    agent.train(episodes=600)

    test_incidents = [
        Incident("server_room"),
        Incident("lobby"),
        Incident("perimeter_fence"),
    ]

    for inc in test_incidents:
        chosen = agent.dispatch(inc)
        ttime = g.shortest_path_cost(chosen.current_zone, inc.zone)
        print(f"  Incident at {inc.zone}: dispatch {chosen.unit_id}  (ETT={ttime}s)")

    print("[DEMO] Task 58 completed.")


if __name__ == "__main__":
    _demo()

