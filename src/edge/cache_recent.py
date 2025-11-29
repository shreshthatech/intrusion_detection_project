"""
Task 66: On-device caching of embeddings & last-N frames.

We implement:
  - FrameEmbeddingCache: keeps a rolling window of recent embeddings
  - Demonstration with synthetic data

This would typically run on an edge device to avoid recomputing
embeddings or reloading frames from disk/network.
"""

from collections import deque
from dataclasses import dataclass
from typing import List, Optional
import time
import torch


@dataclass
class CacheEntry:
    frame_id: int
    timestamp: float
    embedding: torch.Tensor  # 1D vector


class FrameEmbeddingCache:
    def __init__(self, max_frames: int = 100):
        """
        max_frames: maximum number of recent frames to keep.
        Oldest entries are dropped as new ones come in.
        """
        self.max_frames = max_frames
        self._entries: deque[CacheEntry] = deque(maxlen=max_frames)

    def add(self, frame_id: int, embedding: torch.Tensor, timestamp: Optional[float] = None):
        """
        Add a new frame + embedding to cache.
        """
        if timestamp is None:
            timestamp = time.time()

        # ensure 1D tensor on CPU
        if embedding.is_cuda:
            embedding = embedding.cpu()
        embedding = embedding.detach().view(-1)

        entry = CacheEntry(frame_id=frame_id, timestamp=timestamp, embedding=embedding)
        self._entries.append(entry)

    def get_last_n(self, n: int) -> List[CacheEntry]:
        """
        Return the last n entries (most recent first).
        """
        n = min(n, len(self._entries))
        return list(self._entries)[-n:]

    def get_since(self, t_min: float) -> List[CacheEntry]:
        """
        Return all entries with timestamp >= t_min.
        """
        return [e for e in self._entries if e.timestamp >= t_min]

    def __len__(self):
        return len(self._entries)


# ----------------- DEMO ----------------- #

def _demo():
    print("[DEMO] Task 66 – FrameEmbeddingCache")

    cache = FrameEmbeddingCache(max_frames=5)

    # Simulate adding 7 frames with random embeddings (dim=64)
    for i in range(7):
        emb = torch.randn(64)
        cache.add(frame_id=i, embedding=emb)
        print(f"  Added frame_id={i}, cache_size={len(cache)}")

    print("\n[INFO] Cache contents (most recent last):")
    for entry in cache.get_last_n(5):
        print(f"  frame_id={entry.frame_id}, ts={entry.timestamp:.0f}, emb_dim={entry.embedding.shape[0]}")

    # Example: get entries from the last X seconds
    now = time.time()
    recent = cache.get_since(now - 60)  # last 60 seconds
    print(f"\n[INFO] Entries in last 60s: {len(recent)}")

    print("\n[DEMO] Task 66 completed.")


if __name__ == "__main__":
    _demo()
