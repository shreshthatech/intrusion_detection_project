"""
Task 21: Time-sync buffer for late fusion using ring queues.

We implement:
  - RingBuffer: fixed-size buffer of (timestamp, data) for one sensor
  - MultiSensorBuffer: manages multiple RingBuffers and lets you query
    a time-aligned snapshot across sensors.
"""

from collections import deque
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class Sample:
    """Represents one sensor sample."""
    t: float      # timestamp in seconds
    payload: Any  # arbitrary data (features, detections, etc.)


class RingBuffer:
    """
    Fixed-size ring buffer for (timestamp, payload) samples of one sensor.
    Old entries are automatically discarded when capacity is exceeded.
    """

    def __init__(self, capacity: int = 256):
        self.capacity = capacity
        self._buf: deque[Sample] = deque(maxlen=capacity)

    def push(self, t: float, payload: Any) -> None:
        """Append a new sample."""
        self._buf.append(Sample(t=t, payload=payload))

    def __len__(self) -> int:
        return len(self._buf)

    def as_list(self) -> List[Sample]:
        """Return samples as a list (oldest -> newest)."""
        return list(self._buf)

    def nearest(self, t_query: float, max_delta: Optional[float] = None) -> Optional[Sample]:
        """
        Return the sample whose timestamp is closest to t_query.

        If max_delta is not None, return None if the closest sample is farther
        than max_delta seconds away.
        """
        if not self._buf:
            return None

        best_sample = None
        best_dist = float("inf")

        for s in self._buf:
            dist = abs(s.t - t_query)
            if dist < best_dist:
                best_dist = dist
                best_sample = s

        if max_delta is not None and best_dist > max_delta:
            return None

        return best_sample


class MultiSensorBuffer:
    """
    Maintains ring buffers per sensor and provides time-synchronized views.

    Typical usage:
      buf = MultiSensorBuffer(default_capacity=512)

      # push samples
      buf.push("video", t, video_features)
      buf.push("audio", t, audio_features)
      buf.push("rf",    t, rf_features)

      # later fusion by time
      snapshot = buf.get_aligned(t_query, max_delta=0.2)
      # snapshot is dict: sensor_name -> Sample or None
    """

    def __init__(self, default_capacity: int = 512):
        self.default_capacity = default_capacity
        self.buffers: Dict[str, RingBuffer] = {}

    def _get_or_create(self, sensor_name: str) -> RingBuffer:
        if sensor_name not in self.buffers:
            self.buffers[sensor_name] = RingBuffer(capacity=self.default_capacity)
        return self.buffers[sensor_name]

    def push(self, sensor_name: str, t: float, payload: Any) -> None:
        """Push a sample into the named sensor's buffer."""
        buf = self._get_or_create(sensor_name)
        buf.push(t, payload)

    def get_aligned(
        self,
        t_query: float,
        max_delta: Optional[float] = None,
    ) -> Dict[str, Optional[Sample]]:
        """
        For a fusion time t_query, return nearest sample per sensor.

        Args:
          t_query: target time for fusion.
          max_delta: if provided, drop sensors whose closest sample is
                     farther than this many seconds.

        Returns:
          dict: {sensor_name: Sample or None}
        """
        result: Dict[str, Optional[Sample]] = {}
        for name, buf in self.buffers.items():
            result[name] = buf.nearest(t_query, max_delta=max_delta)
        return result

    def recent_times(self) -> Dict[str, Optional[float]]:
        """
        Convenience: get timestamp of the latest sample in each sensor.
        """
        out: Dict[str, Optional[float]] = {}
        for name, buf in self.buffers.items():
            if len(buf) == 0:
                out[name] = None
            else:
                out[name] = buf.as_list()[-1].t
        return out


# --- Quick self-test / demo ---

def _demo():
    """
    Small demo to show how the buffers work.
    Run this file directly to see it in action:

        python src/fusion/time_sync_buffer.py
    """
    import random

    print("[DEMO] Building MultiSensorBuffer with sensors: video, audio, rf")
    msb = MultiSensorBuffer(default_capacity=10)

    # Simulate some samples
    for i in range(10):
        t = i * 0.1
        msb.push("video", t, {"frame_id": i})
        msb.push("audio", t + random.uniform(-0.03, 0.03), {"audio_chunk": i})
        msb.push("rf", t + random.uniform(-0.05, 0.05), {"rf_power": i})

    print("[DEMO] Recent times:", msb.recent_times())

    # Query a time for fusion
    t_query = 0.55
    snapshot = msb.get_aligned(t_query, max_delta=0.08)
    print(f"[DEMO] Aligned samples near t={t_query:.2f} (max_delta=0.08):")
    for name, sample in snapshot.items():
        if sample is None:
            print(f"  {name}: NONE")
        else:
            print(f"  {name}: t={sample.t:.3f}, payload={sample.payload}")


if __name__ == "__main__":
    _demo()
