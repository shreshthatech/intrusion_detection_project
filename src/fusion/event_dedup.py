"""
Task 28: Event deduplication across sensors (spatial + temporal).

We:
  - Define a simple Event structure.
  - Implement IoU-based spatial overlap.
  - Implement EventDeduplicator which:
      * groups events within a time window
      * merges events of same type that overlap in space
  - Demo: read frame_events.jsonl (video), add fake audio/RF events,
          deduplicate, and print stats.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from pathlib import Path
import json
import math


PROJECT_ROOT = Path(__file__).resolve().parents[2]
FRAME_EVENTS_FILE = PROJECT_ROOT / "data" / "processed" / "api" / "frame_events.jsonl"


@dataclass
class Event:
    t: float                     # timestamp in seconds
    frame: int                   # frame index (if applicable)
    sensor: str                  # "video", "audio", "rf", "thermal", ...
    type: str                    # e.g. "person", "gunshot", "jammer"
    bbox: Optional[List[int]]    # [x1,y1,x2,y2] or None for non-spatial (audio/RF)
    p: float                     # probability/confidence
    extra: Dict[str, Any] = field(default_factory=dict)


def iou(boxA: List[int], boxB: List[int]) -> float:
    """
    IoU between two [x1,y1,x2,y2] boxes.
    """
    x1A, y1A, x2A, y2A = boxA
    x1B, y1B, x2B, y2B = boxB

    inter_x1 = max(x1A, x1B)
    inter_y1 = max(y1A, y1B)
    inter_x2 = min(x2A, x2B)
    inter_y2 = min(y2A, y2B)

    if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
        return 0.0

    inter = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
    areaA = (x2A - x1A) * (y2A - y1A)
    areaB = (x2B - x1B) * (y2B - y1B)

    union = areaA + areaB - inter
    if union <= 0:
        return 0.0
    return inter / union


class EventDeduplicator:
    """
    Deduplicate events based on temporal + spatial overlap.

    Rules:
      - Same type
      - |t1 - t2| <= time_window
      - IoU(bbox) >= iou_thresh (if both have bboxes)
      - For audio/RF without bbox, only time_window is used.
    """

    def __init__(self, time_window: float = 1.0, iou_thresh: float = 0.3):
        self.time_window = time_window
        self.iou_thresh = iou_thresh

    def dedup(self, events: List[Event]) -> List[Event]:
        if not events:
            return []

        # sort by time
        events = sorted(events, key=lambda e: e.t)

        clusters: List[List[Event]] = []

        for ev in events:
            placed = False
            # Try to fit into an existing cluster
            for cluster in clusters:
                rep = cluster[0]  # representative event (first)
                if self._same_event(rep, ev):
                    cluster.append(ev)
                    placed = True
                    break
            if not placed:
                clusters.append([ev])

        # Merge clusters into single events
        merged: List[Event] = []
        for cluster in clusters:
            merged.append(self._merge_cluster(cluster))

        return merged

    def _same_event(self, a: Event, b: Event) -> bool:
        # Check time
        if abs(a.t - b.t) > self.time_window:
            return False

        # Check type
        if a.type != b.type:
            return False

        # If both have bboxes, check IoU
        if a.bbox is not None and b.bbox is not None:
            return iou(a.bbox, b.bbox) >= self.iou_thresh

        # If at least one has no bbox (audio/RF), only time+type used
        return True

    def _merge_cluster(self, cluster: List[Event]) -> Event:
        """
        Merge events for same underlying incident.
        Strategy:
          - time: average
          - frame: min
          - bbox: average corners if spatial
          - p: 1 - product(1-p_i)  (union of probabilities)
          - sensor: "multi" if >1 sensor involved, else that sensor
        """
        if not cluster:
            raise ValueError("Empty cluster")

        if len(cluster) == 1:
            return cluster[0]

        ts = [e.t for e in cluster]
        frames = [e.frame for e in cluster]
        ps = [e.p for e in cluster]
        sensors = {e.sensor for e in cluster}

        # time
        t_avg = sum(ts) / len(ts)
        frame_min = min(frames)

        # probability union
        p_union = 1.0
        for p in ps:
            p_union *= (1.0 - p)
        p_union = 1.0 - p_union

        # bbox merging (only for events that have bbox)
        bboxes = [e.bbox for e in cluster if e.bbox is not None]
        if bboxes:
            x1 = sum(b[0] for b in bboxes) / len(bboxes)
            y1 = sum(b[1] for b in bboxes) / len(bboxes)
            x2 = sum(b[2] for b in bboxes) / len(bboxes)
            y2 = sum(b[3] for b in bboxes) / len(bboxes)
            bbox_merged = [int(x1), int(y1), int(x2), int(y2)]
        else:
            bbox_merged = None

        # sensor label
        sensor_label = "multi" if len(sensors) > 1 else next(iter(sensors))

        # Take type from first
        ev_type = cluster[0].type

        return Event(
            t=t_avg,
            frame=frame_min,
            sensor=sensor_label,
            type=ev_type,
            bbox=bbox_merged,
            p=p_union,
            extra={"sensors": list(sensors), "count": len(cluster)},
        )


# ---------------- DEMO USING frame_events.jsonl ---------------- #

def _demo():
    print("[DEMO] EventDeduplicator demo")

    if not FRAME_EVENTS_FILE.exists():
        print("[WARN] frame_events.jsonl not found at", FRAME_EVENTS_FILE)
        return

    # Load some video events
    events: List[Event] = []
    with open(FRAME_EVENTS_FILE, "r") as f:
        for i, line in enumerate(f):
            if i > 50:  # just first 50 for demo
                break
            obj = json.loads(line)
            events.append(
                Event(
                    t=float(obj["t"]),
                    frame=int(obj["frame"]),
                    sensor="video",
                    type=obj["type"],
                    bbox=obj["bbox"],
                    p=float(obj["p"]),
                    extra={"source": "video"},
                )
            )

    # Add some fake audio events around same times
    fake_audio = []
    for ev in events[::10]:  # every 10th event
        fake_audio.append(
            Event(
                t=ev.t + 0.2,   # slightly shifted in time
                frame=ev.frame,
                sensor="audio",
                type=ev.type,
                bbox=None,      # no spatial info
                p=0.4,
                extra={"source": "audio_fake"},
            )
        )

    all_events = events + fake_audio

    print(f"[DEMO] Loaded {len(events)} video events + {len(fake_audio)} fake audio events")
    dedup = EventDeduplicator(time_window=0.5, iou_thresh=0.3)
    merged = dedup.dedup(all_events)
    print(f"[DEMO] After deduplication: {len(merged)} events")

    # Show first few merged events
    for ev in merged[:5]:
        print(
            f"  t={ev.t:.3f}, frame={ev.frame}, sensor={ev.sensor}, "
            f"type={ev.type}, p={ev.p:.3f}, sensors={ev.extra.get('sensors')}"
        )


if __name__ == "__main__":
    _demo()
