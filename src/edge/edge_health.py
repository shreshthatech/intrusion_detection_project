"""
Task 67: Edge health metrics (FPS, thermals, memory, power).

We simulate:
  - CPU usage
  - RAM usage
  - FPS sampling
  - Device temperature
  - Power usage

And save a JSONL log file.
"""

import time
import random
import json
import psutil
from pathlib import Path


def get_fps():
    # Simulate FPS measure (real systems measure frame timings)
    return round(random.uniform(18.0, 32.0), 2)


def get_temperature():
    # Simulated temperature for CPU/GPU
    return round(random.uniform(50.0, 75.0), 2)


def get_power_usage():
    # Simulated power draw in watts
    return round(random.uniform(2.0, 8.0), 2)


def sample_edge_metrics():
    cpu = psutil.cpu_percent()
    ram = psutil.virtual_memory().percent
    fps = get_fps()
    temp = get_temperature()
    power = get_power_usage()

    return {
        "timestamp": time.time(),
        "cpu_percent": cpu,
        "ram_percent": ram,
        "fps": fps,
        "temperature_c": temp,
        "power_w": power
    }


def main():
    print("[INFO] Task 67 – Edge Health Metrics")

    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    LOG_DIR = PROJECT_ROOT / "data" / "processed" / "edge_metrics"
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    LOG_FILE = LOG_DIR / "edge_health_log.jsonl"

    print("[INFO] Sampling metrics for 5 seconds...")

    with open(LOG_FILE, "w") as f:
        for i in range(5):
            metrics = sample_edge_metrics()
            print(f"  Sample {i}: CPU={metrics['cpu_percent']}% RAM={metrics['ram_percent']}% FPS={metrics['fps']}")
            f.write(json.dumps(metrics) + "\n")
            time.sleep(1)

    print(f"[SUCCESS] Saved metrics to: {LOG_FILE}")
    print("[DEMO] Task 67 completed.")


if __name__ == "__main__":
    main()
