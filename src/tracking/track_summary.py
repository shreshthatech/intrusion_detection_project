"""
Task 40: Track-level event summarizer.

Given a sequence of track points (t, x, y), we compute:
  - start_time, end_time, duration
  - start_position, end_position
  - total_distance
  - average_speed
  - simple hotspot region (most visited grid cell)
"""

from dataclasses import dataclass
from typing import List, Dict, Tuple
import numpy as np


@dataclass
class TrackPoint:
    t: float
    x: float
    y: float


class TrackSummarizer:
    def __init__(self, grid_size: int = 3):
        """
        grid_size: for simple hotspot heatmap (grid_size x grid_size).
                   coordinates assumed in [0,1] x [0,1] for demo.
        """
        self.grid_size = grid_size

    def summarize(self, track_id: int, points: List[TrackPoint]) -> Dict:
        if len(points) < 2:
            return {
                "track_id": track_id,
                "start_time": None,
                "end_time": None,
                "duration": 0.0,
                "start_pos": None,
                "end_pos": None,
                "total_distance": 0.0,
                "avg_speed": 0.0,
                "hotspot_cell": None,
            }

        ts = np.array([p.t for p in points], dtype=float)
        xs = np.array([p.x for p in points], dtype=float)
        ys = np.array([p.y for p in points], dtype=float)

        # Time info
        start_time = float(ts[0])
        end_time = float(ts[-1])
        duration = end_time - start_time

        # Positions
        start_pos = (float(xs[0]), float(ys[0]))
        end_pos = (float(xs[-1]), float(ys[-1]))

        # Distances & speed
        dx = np.diff(xs)
        dy = np.diff(ys)
        dt = np.diff(ts)
        dist_step = np.sqrt(dx**2 + dy**2)
        total_distance = float(dist_step.sum())

        # avoid division by zero
        total_time = dt.sum() if dt.sum() > 1e-6 else 1e-6
        avg_speed = total_distance / total_time

        # Hotspot: simple 2D grid over [0,1]x[0,1]
        gs = self.grid_size
        # clamp coords into [0,1]
        xs_clamped = np.clip(xs, 0.0, 1.0)
        ys_clamped = np.clip(ys, 0.0, 1.0)
        # compute cell index
        cell_x = np.floor(xs_clamped * gs).astype(int)
        cell_y = np.floor(ys_clamped * gs).astype(int)
        cell_x = np.clip(cell_x, 0, gs - 1)
        cell_y = np.clip(cell_y, 0, gs - 1)

        # count visits per cell
        heat = np.zeros((gs, gs), dtype=int)
        for cx, cy in zip(cell_x, cell_y):
            heat[cy, cx] += 1  # (row=y, col=x)

        # find hotspot cell
        idx = np.unravel_index(np.argmax(heat), heat.shape)
        hotspot = (int(idx[1]), int(idx[0]))  # (x_cell, y_cell)

        return {
            "track_id": track_id,
            "start_time": start_time,
            "end_time": end_time,
            "duration": duration,
            "start_pos": start_pos,
            "end_pos": end_pos,
            "total_distance": total_distance,
            "avg_speed": avg_speed,
            "hotspot_cell": hotspot,
            "grid_size": gs,
        }


# -------------- DEMO -------------- #

def _demo():
    print("[DEMO] TrackSummarizer (Task 40)")

    # Synthetic track: moves from left to right with some noise
    np.random.seed(0)
    N = 25
    ts = np.linspace(0, 5, N)
    xs = np.linspace(0.1, 0.9, N) + np.random.normal(scale=0.02, size=N)
    ys = 0.5 + np.random.normal(scale=0.05, size=N)

    points = [TrackPoint(t=float(t), x=float(x), y=float(y))
              for t, x, y in zip(ts, xs, ys)]

    summarizer = TrackSummarizer(grid_size=3)
    summary = summarizer.summarize(track_id=7, points=points)

    print("\nSummary for track 7:")
    for k, v in summary.items():
        print(f"  {k}: {v}")

    print("\n[DEMO] Done.")


if __name__ == "__main__":
    _demo()
