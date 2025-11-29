"""
Task 20: Export per-frame detections to compact JSONL format.

Input:
  data/processed/rgb_detections/postprocessed_detections.jsonl

Output:
  data/processed/api/frame_events.jsonl

Schema per line:
  {
    "t": <time_seconds>,
    "frame": <frame_idx>,
    "type": <label>,
    "bbox": [x1, y1, x2, y2],
    "p": <probability>
  }
"""

from pathlib import Path
import json


PROJECT_ROOT = Path(__file__).resolve().parents[2]

POSTPROC_IN = PROJECT_ROOT / "data" / "processed" / "rgb_detections" / "postprocessed_detections.jsonl"
API_DIR = PROJECT_ROOT / "data" / "processed" / "api"
API_DIR.mkdir(parents=True, exist_ok=True)
FRAME_EVENTS_OUT = API_DIR / "frame_events.jsonl"


def export_events(fps: float = 23.97):
    print("[INFO] Exporting detections from:", POSTPROC_IN)

    in_f = open(POSTPROC_IN, "r")
    out_f = open(FRAME_EVENTS_OUT, "w")

    count = 0
    for line in in_f:
        line = line.strip()
        if not line:
            continue
        obj = json.loads(line)

        frame_idx = int(obj["frame_idx"])
        t = frame_idx / fps

        event = {
            "t": round(t, 3),
            "frame": frame_idx,
            "type": obj["label"],
            "bbox": obj["bbox"],
            "p": float(obj.get("conf_calibrated", obj.get("conf_raw", 0.0))),
        }

        out_f.write(json.dumps(event) + "\n")
        count += 1

    in_f.close()
    out_f.close()

    print(f"[SUCCESS] Exported {count} events to:", FRAME_EVENTS_OUT)


if __name__ == "__main__":
    export_events()
