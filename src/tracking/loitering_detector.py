"""
Task 38: Loitering detector using dwell-time inside a region.

We:
 - Define a zone polygon
 - Build synthetic track points
 - Compute dwell time
 - Trigger loitering alert if > threshold
"""

from dataclasses import dataclass
from shapely.geometry import Point, Polygon
import numpy as np


@dataclass
class TrackPoint:
    t: float
    x: float
    y: float


class LoiteringDetector:
    def __init__(self, zone_polygon: Polygon, dwell_threshold_sec: float = 4.0):
        """
        zone_polygon: shapely Polygon defining monitored area
        dwell_threshold_sec: time spent inside polygon to trigger alert
        """
        self.zone = zone_polygon
        self.dwell_th = dwell_threshold_sec

    def compute_dwell(self, points):
        """
        points: list of TrackPoint
        Returns total time inside zone.
        """
        if len(points) < 2:
            return 0.0

        dwell = 0.0
        for i in range(1, len(points)):
            p_prev = points[i - 1]
            p_curr = points[i]

            inside_prev = self.zone.contains(Point(p_prev.x, p_prev.y))
            inside_curr = self.zone.contains(Point(p_curr.x, p_curr.y))

            # estimate dt between points
            dt = p_curr.t - p_prev.t

            # count dwell if both or either are inside
            if inside_prev or inside_curr:
                dwell += dt

        return dwell

    def is_loitering(self, points):
        dwell = self.compute_dwell(points)
        return dwell >= self.dwell_th, dwell


# ----------------- DEMO ----------------- #

def _demo():
    print("[DEMO] Loitering Detector (Task 38)")

    # zone in 0..1 normalized space
    zone = Polygon([(0.3, 0.3), (0.7, 0.3), (0.7, 0.7), (0.3, 0.7)])
    detector = LoiteringDetector(zone, dwell_threshold_sec=4.0)

    # Synthetic track: stays mostly inside zone for 6 seconds
    ts = np.linspace(0, 6, 30)
    xs = 0.5 + np.random.normal(scale=0.02, size=30)  # inside zone
    ys = 0.5 + np.random.normal(scale=0.02, size=30)

    points = [TrackPoint(t=float(t), x=float(x), y=float(y))
              for t, x, y in zip(ts, xs, ys)]

    is_loit, dwell = detector.is_loitering(points)

    print(f"Total dwell time = {dwell:.2f} sec")
    print("Loitering =", is_loit)

    print("\n[DEMO] Done.")


if __name__ == "__main__":
    _demo()
