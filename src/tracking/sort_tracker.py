"""
Task 31 + 32: SORT-style tracker with built-in Kalman Filter (constant velocity).

- Reads detections from: data/processed/api/frame_events.jsonl
- Tracks 'person' detections across frames.
- Uses KalmanFilterCV for smooth prediction + update.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict
import json
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
EVENTS_FILE = PROJECT_ROOT / "data" / "processed" / "api" / "frame_events.jsonl"


# ---------------- IoU helper ---------------- #

def iou(boxA, boxB) -> float:
    """
    IoU between boxes: [x1,y1,x2,y2]
    """
    x1A, y1A, x2A, y2A = boxA
    x1B, y1B, x2B, y2B = boxB

    inter_x1 = max(x1A, x1B)
    inter_y1 = max(y1A, y1B)
    inter_x2 = min(x2A, x2B)
    inter_y2 = min(y1A, y2B)

    if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
        return 0.0

    inter = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
    areaA = (x2A - x1A) * (y2A - y1A)
    areaB = (x2B - x1B) * (y2B - y1B)

    union = areaA + areaB - inter
    if union <= 0:
        return 0.0
    return inter / union


# ---------------- Kalman Filter (Task 32) ---------------- #

class KalmanFilterCV:
    """
    Constant-velocity Kalman Filter for bbox:

        state x = [cx, cy, w, h, vx, vy, vw, vh]^T
        measurement z = [cx, cy, w, h]^T
    """

    def __init__(self):
        self.dim_x = 8
        self.dim_z = 4

        self.x = np.zeros((self.dim_x, 1))
        self.P = np.eye(self.dim_x) * 10.0

        dt = 1.0
        self.F = np.eye(self.dim_x)
        for i in range(4):
            self.F[i, i + 4] = dt

        self.H = np.zeros((self.dim_z, self.dim_x))
        self.H[0, 0] = 1.0
        self.H[1, 1] = 1.0
        self.H[2, 2] = 1.0
        self.H[3, 3] = 1.0

        self.Q = np.eye(self.dim_x) * 1.0
        self.R = np.eye(self.dim_z) * 1.0

    def init_state(self, bbox):
        x1, y1, x2, y2 = bbox
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        w = x2 - x1
        h = y2 - y1

        self.x = np.array([[cx], [cy], [w], [h], [0.0], [0.0], [0.0], [0.0]])
        self.P = np.eye(self.dim_x) * 10.0

    def predict(self):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x.copy()

    def update(self, bbox):
        x1, y1, x2, y2 = bbox
        z = np.array([
            [(x1 + x2) / 2.0],
            [(y1 + y2) / 2.0],
            [x2 - x1],
            [y2 - y1]
        ])

        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y

        I = np.eye(self.dim_x)
        self.P = (I - K @ self.H) @ self.P

    def get_bbox(self):
        cx, cy, w, h = self.x[0, 0], self.x[1, 0], self.x[2, 0], self.x[3, 0]
        x1 = int(cx - w / 2.0)
        y1 = int(cy - h / 2.0)
        x2 = int(cx + w / 2.0)
        y2 = int(cy + h / 2.0)
        return [x1, y1, x2, y2]


# ---------------- Track + Tracker ---------------- #

@dataclass
class Track:
    track_id: int
    kf: KalmanFilterCV
    bbox: List[int]        # last bbox (KF-predicted/updated)
    last_frame: int
    hits: int = 1
    missed: int = 0


class KalmanSORTTracker:
    """
    SORT-like tracker with built-in Kalman filter.

    detections: list of { "bbox": [x1,y1,x2,y2], "score": float }
    """

    def __init__(self, iou_thresh: float = 0.3, max_missed: int = 10):
        self.iou_thresh = iou_thresh
        self.max_missed = max_missed
        self.next_track_id = 1
        self.tracks: Dict[int, Track] = {}

    def _match(self, detections: List[Dict]):
        det_indices = list(range(len(detections)))
        track_ids = list(self.tracks.keys())

        matches = []          # (track_id, det_idx)
        unmatched_tracks = set(track_ids)
        unmatched_dets = set(det_indices)

        det_sorted = sorted(det_indices,
                            key=lambda i: detections[i]["score"],
                            reverse=True)

        for d_idx in det_sorted:
            best_iou = 0.0
            best_track_id = None

            for t_id in list(unmatched_tracks):
                t = self.tracks[t_id]
                i = iou(t.bbox, detections[d_idx]["bbox"])
                if i > best_iou:
                    best_iou = i
                    best_track_id = t_id

            if best_track_id is not None and best_iou >= self.iou_thresh:
                matches.append((best_track_id, d_idx))
                unmatched_tracks.discard(best_track_id)
                unmatched_dets.discard(d_idx)

        return matches, list(unmatched_tracks), list(unmatched_dets)

    def update(self, detections: List[Dict], frame_idx: int) -> List[Track]:
        # 0) Predict all tracks forward one step
        for t in self.tracks.values():
            t.kf.predict()
            t.bbox = t.kf.get_bbox()

        # 1) Match detections to predicted tracks
        matches, unmatched_tracks, unmatched_dets = self._match(detections)

        # 2) Update matched tracks with new detections
        for track_id, d_idx in matches:
            det = detections[d_idx]
            t = self.tracks[track_id]
            t.kf.update(det["bbox"])
            t.bbox = t.kf.get_bbox()
            t.last_frame = frame_idx
            t.hits += 1
            t.missed = 0

        # 3) Increase missed count on unmatched tracks
        for t_id in unmatched_tracks:
            t = self.tracks[t_id]
            t.missed += 1

        # 4) Create new tracks for unmatched detections
        for d_idx in unmatched_dets:
            det = detections[d_idx]
            kf = KalmanFilterCV()
            kf.init_state(det["bbox"])
            bbox_init = kf.get_bbox()
            self.tracks[self.next_track_id] = Track(
                track_id=self.next_track_id,
                kf=kf,
                bbox=bbox_init,
                last_frame=frame_idx,
                hits=1,
                missed=0
            )
            self.next_track_id += 1

        # 5) Remove dead tracks
        dead_ids = [t_id for t_id, t in self.tracks.items()
                    if t.missed > self.max_missed]
        for t_id in dead_ids:
            del self.tracks[t_id]

        return list(self.tracks.values())


# ---------------- DEMO ---------------- #

def _demo():
    print("[DEMO] KalmanSORTTracker demo using frame_events.jsonl")

    if not EVENTS_FILE.exists():
        print("[WARN] Events file not found:", EVENTS_FILE)
        return

    per_frame: Dict[int, List[Dict]] = {}
    with open(EVENTS_FILE, "r") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            if obj["type"] != "person":
                continue

            frame = int(obj["frame"])
            det = {
                "bbox": obj["bbox"],
                "score": float(obj["p"]),
            }
            per_frame.setdefault(frame, []).append(det)

    tracker = KalmanSORTTracker(iou_thresh=0.3, max_missed=5)
    frames_sorted = sorted(per_frame.keys())[:80]

    for frame_idx in frames_sorted:
        dets = per_frame[frame_idx]
        tracks = tracker.update(dets, frame_idx)

        print(f"[FRAME {frame_idx}] detections={len(dets)} active_tracks={len(tracks)}")
        for t in tracks:
            x1, y1, x2, y2 = t.bbox
            print(f"  Track {t.track_id}: bbox=({x1},{y1},{x2},{y2}), hits={t.hits}, missed={t.missed}")

    print("[DEMO] Done.")


if __name__ == "__main__":
    _demo()
