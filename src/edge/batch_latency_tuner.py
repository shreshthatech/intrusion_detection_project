"""
Task 65: Batch-size / latency tuner

We:
  - Load TinyMotionCNN (or any model)
  - Try batch sizes: 1,2,4,8,16,32
  - Measure speed
  - Pick best batch size
  - Save report
"""

import time
import json
import torch
import torch.nn as nn
from pathlib import Path


# Use the same simple TinyMotionCNN from earlier
class TinyMotionCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(2, 8, 3, padding=1), nn.ReLU(),
            nn.Conv2d(8, 16, 3, padding=1), nn.ReLU(),
            nn.Conv2d(16, 1, 1), nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x)


def main():
    print("[INFO] Task 65 – Batch-size / Latency Tuner")

    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    MODEL_PATH = PROJECT_ROOT / "models" / "vision" / "tiny_motion_cnn.pt"
    REPORT_PATH = PROJECT_ROOT / "models" / "vision" / "batch_latency_report.json"

    # Load model
    model = TinyMotionCNN()
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    model.eval()

    batch_sizes = [1, 2, 4, 8, 16, 32]
    results = {}

    print("\n[TESTING LATENCY]")
    for b in batch_sizes:
        x = torch.randn(b, 2, 90, 160)
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

        # Warm-up
        for _ in range(10):
            model(x)

        # Measure time
        t0 = time.time()
        for _ in range(40):
            model(x)
        dt = time.time() - t0

        avg_time = dt / 40
        results[b] = avg_time
        print(f"  batch={b:<3}   avg_time={avg_time:.6f} sec")

    # Pick best batch size
    best_batch = min(results, key=results.get)
    print(f"\n[BEST] Optimal batch size = {best_batch} (fastest inference)")

    # Save JSON report
    with open(REPORT_PATH, "w") as f:
        json.dump({
            "latency_results": results,
            "best_batch": best_batch
        }, f, indent=2)

    print(f"[SUCCESS] Report saved to: {REPORT_PATH}")
    print("[DEMO] Task 65 completed.")


if __name__ == "__main__":
    main()
