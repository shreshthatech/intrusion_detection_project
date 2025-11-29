"""
Task 10: Lightweight data profiler.

- Checks class balance (overall and per split)
- Plots class histogram
- Plots window start_time distribution
- Saves PNGs to logs/profiles/
"""

from pathlib import Path
import json
from collections import Counter
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[2]
WINDOWS_DIR = PROJECT_ROOT / "data" / "processed" / "windows"

PROFILE_DIR = PROJECT_ROOT / "logs" / "profiles"
PROFILE_DIR.mkdir(parents=True, exist_ok=True)


def load_jsonl(path: Path):
    items = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    return items


def profile_file(name: str, path: Path):
    print(f"[INFO] Profiling {name} -> {path}")
    items = load_jsonl(path)

    labels = [int(x["label"]) for x in items]
    starts = [float(x["start_time"]) for x in items]

    counts = Counter(labels)
    print(f"    Class counts: {dict(counts)}")

    # Class balance bar chart
    plt.figure()
    plt.bar(counts.keys(), counts.values())
    plt.xticks([0, 1], ["no_intrusion", "intrusion"])
    plt.title(f"Class balance: {name}")
    plt.xlabel("Class")
    plt.ylabel("Count")
    plt.tight_layout()
    out_bar = PROFILE_DIR / f"class_balance_{name}.png"
    plt.savefig(out_bar)
    plt.close()

    # Start_time histogram
    plt.figure()
    plt.hist(starts, bins=10)
    plt.title(f"Window start times: {name}")
    plt.xlabel("start_time (s)")
    plt.ylabel("count")
    plt.tight_layout()
    out_hist = PROFILE_DIR / f"start_time_hist_{name}.png"
    plt.savefig(out_hist)
    plt.close()

    print(f"    Saved plots: {out_bar}, {out_hist}")


def main():
    # Overall (balanced)
    overall = WINDOWS_DIR / "window_labels_balanced.jsonl"
    train = WINDOWS_DIR / "train_windows.jsonl"
    val = WINDOWS_DIR / "val_windows.jsonl"
    test = WINDOWS_DIR / "test_windows.jsonl"

    profile_file("overall_balanced", overall)
    profile_file("train", train)
    profile_file("val", val)
    profile_file("test", test)

    print("[SUCCESS] Data profiling completed. Check logs/profiles/ for plots.")


if __name__ == "__main__":
    main()
