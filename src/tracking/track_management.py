"""
Task 34: Track management with birth/death + confidence decay.

This is a standalone demo that shows:
  - how to maintain a track confidence score,
  - how tracks are "born" only after enough hits,
  - how they "die" when missed or low confidence.

You can port the same logic into your main KalmanSORTTracker later.
"""

from dataclasses import dataclass
from typing import List, Dict
import random


@dataclass
class Track:
    track_id: int
    confidence: float       # running confidence
    hits: int = 1           # matched frames
    missed: int = 0         # missed frames
    is_confirmed: bool = False  # becomes True after enough hits


class TrackManager:
    def __init__(
        self,
        min_hits: int = 3,
        max_missed: int = 5,
        decay_per_miss: float = 0.8,
        alpha: float = 0.6,
        min_conf: float = 0.2,
    ):
        """
        min_hits: how many matches before track is considered 'confirmed'
        max_missed: frames allowed to be missed before forced deletion
        decay_per_miss: multiplier applied to confidence when track is missed
        alpha: smoothing factor for confidence update:
               new_conf = alpha*old + (1-alpha)*det_score
        min_conf: if confidence drops below this, track is removed
        """
        self.min_hits = min_hits
        self.max_missed = max_missed
        self.decay_per_miss = decay_per_miss
        self.alpha = alpha
        self.min_conf = min_conf

        self.tracks: Dict[int, Track] = {}
        self.next_id = 1

    def new_track(self, det_score: float) -> Track:
        t = Track(
            track_id=self.next_id,
            confidence=det_score,
            hits=1,
            missed=0,
            is_confirmed=False,
        )
        self.tracks[self.next_id] = t
        self.next_id += 1
        return t

    def update_match(self, track_id: int, det_score: float):
        t = self.tracks[track_id]
        # exponential smoothing for confidence
        t.confidence = self.alpha * t.confidence + (1.0 - self.alpha) * det_score
        t.hits += 1
        t.missed = 0
        if not t.is_confirmed and t.hits >= self.min_hits:
            t.is_confirmed = True

    def update_missed(self, track_id: int):
        t = self.tracks[track_id]
        t.missed += 1
        t.confidence *= self.decay_per_miss

    def prune(self):
        """Remove dead tracks."""
        dead_ids = []
        for tid, t in self.tracks.items():
            if t.missed > self.max_missed or t.confidence < self.min_conf:
                dead_ids.append(tid)
        for tid in dead_ids:
            del self.tracks[tid]

    def active_tracks(self) -> List[Track]:
        """Return only confirmed tracks."""
        return [t for t in self.tracks.values() if t.is_confirmed]


# -------------- DEMO -------------- #

def _demo():
    print("[DEMO] TrackManager (Task 34)")

    tm = TrackManager(
        min_hits=3,
        max_missed=4,
        decay_per_miss=0.8,
        alpha=0.6,
        min_conf=0.2,
    )

    # Simulate a single object with varying detection scores and occasional misses.
    # We'll pretend frames 0..14 exist, and we log what happens.
    scores = [0.9, 0.85, 0.8, None, 0.7, None, None, 0.75, 0.6, None, None, 0.55, 0.5, None, 0.4]

    track_id = None

    for frame_idx, s in enumerate(scores):
        print(f"\n[FRAME {frame_idx}] detection_score={s}")

        if track_id is None:
            # No track yet: if detection appears, start new
            if s is not None:
                t = tm.new_track(det_score=s)
                track_id = t.track_id
                print(f"  New track {track_id} created with conf={t.confidence:.3f}")
        else:
            # We have a track
            if s is not None:
                tm.update_match(track_id, det_score=s)
                t = tm.tracks[track_id]
                print(f"  Track {track_id} MATCHED: conf={t.confidence:.3f}, hits={t.hits}, missed={t.missed}, confirmed={t.is_confirmed}")
            else:
                tm.update_missed(track_id)
                t = tm.tracks[track_id]
                print(f"  Track {track_id} MISSED: conf={t.confidence:.3f}, hits={t.hits}, missed={t.missed}, confirmed={t.is_confirmed}")

        # Prune dead tracks
        tm.prune()

        # If our track died, clear track_id
        if track_id is not None and track_id not in tm.tracks:
            print(f"  Track {track_id} REMOVED.")
            track_id = None

        # Show active (confirmed) tracks
        active = tm.active_tracks()
        if active:
            ids = [t.track_id for t in active]
            print(f"  Active confirmed tracks: {ids}")
        else:
            print("  No confirmed tracks yet or all removed.")

    print("\n[DEMO] Done.")


if __name__ == "__main__":
    _demo()
