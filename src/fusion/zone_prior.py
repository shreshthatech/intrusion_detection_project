"""
Task 27: Rule + ML Hybrid
-------------------------

We add geofence + restricted-zone priors to influence the Bayesian threat score.

Each zone has:
    - polygon coordinates
    - base prior multiplier

Example:
    public:        0.3x
    semi_secure:   1.0x
    restricted:    2.0x
    high_security: 5.0x

These multipliers modify the Bayesian posterior to bias toward higher vigilance in sensitive areas.
"""

from pathlib import Path
import json
from shapely.geometry import Point, Polygon


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ZONE_FILE = PROJECT_ROOT / "data" / "config" / "zones.json"

# Example zones.json structure:
# {
#   "public": {
#       "multiplier": 0.3,
#       "polygon": [[0,0],[10,0],[10,10],[0,10]]
#   },
#   "restricted": {
#       "multiplier": 2.0,
#       "polygon": [[20,20],[30,20],[30,30],[20,30]]
#   }
# }


class ZonePrior:
    def __init__(self):
        if not ZONE_FILE.exists():
            self._write_default()
        with open(ZONE_FILE, "r") as f:
            self.zones = json.load(f)
        self._compile_polygons()

    def _write_default(self):
        """
        Creates default zones if none exist.
        """
        default = {
            "public": {
                "multiplier": 0.3,
                "polygon": [[0,0],[50,0],[50,50],[0,50]]
            },
            "restricted": {
                "multiplier": 2.0,
                "polygon": [[60,60],[80,60],[80,80],[60,80]]
            },
            "high_security": {
                "multiplier": 5.0,
                "polygon": [[90,90],[110,90],[110,110],[90,110]]
            }
        }
        ZONE_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(ZONE_FILE, "w") as f:
            json.dump(default, f, indent=2)

    def _compile_polygons(self):
        self.poly = {}
        for zone_name, data in self.zones.items():
            coords = data["polygon"]
            self.poly[zone_name] = Polygon(coords)

    def get_multiplier(self, x: float, y: float) -> float:
        """
        For a given position (x,y), return the zone multiplier.
        """
        p = Point(x, y)
        for zone_name, poly in self.poly.items():
            if poly.contains(p):
                return float(self.zones[zone_name]["multiplier"])
        return 1.0  # default fallback


# ---------------- DEMO ---------------- #

def _demo():
    print("[DEMO] Zone Prior Demo")

    zp = ZonePrior()

    test_points = [
        (10, 10),   # public
        (65, 65),   # restricted
        (100, 100), # high security
        (200, 200)  # outside any zone
    ]

    for pt in test_points:
        mult = zp.get_multiplier(pt[0], pt[1])
        print(f"Point {pt} -> multiplier {mult}")


if __name__ == "__main__":
    _demo()
