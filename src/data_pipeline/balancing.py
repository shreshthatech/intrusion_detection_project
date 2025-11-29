"""
Task 6: Data balancing with minority oversampling.

We take window_labels.jsonl (Task 5 output),
check class counts, and oversample the minority class
so that both classes have (almost) equal number of windows.

Output: window_labels_balanced.jsonl
"""

from pathlib import Path
import json
import random
from collections import Counter

PROJECT_ROOT = Path(__file__).resolve().parents[2]
LABELS_FILE = PROJECT_ROOT / "data" / "processed" / "windows" / "window_labels.jsonl"
OUT_FILE = PROJECT_ROOT / "data" / "processed" / "windows" / "window_labels_balanced.jsonl"


def load_window_labels():
    windows = []
    with open(LABELS_FILE, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            windows.append(json.loads(line))
    return windows


def balance_windows(windows):
    # Split by class
    by_class = {0: [], 1: []}
    for w in windows:
        lab = int(w["label"])
        by_class[lab].append(w)

    counts = {k: len(v) for k, v in by_class.items()}
    print("[INFO] Original class counts:", counts)

    # If one class is empty, we can't balance properly
    if len(by_class[0]) == 0 or len(by_class[1]) == 0:
        print("[WARN] One of the classes has 0 samples, skipping balancing.")
        return windows

    # Find majority & minority
    if counts[0] > counts[1]:
        majority_class = 0
        minority_class = 1
    else:
        majority_class = 1
        minority_class = 0

    n_major = counts[majority_class]
    n_minor = counts[minority_class]

    print(f"[INFO] Majority class = {majority_class} ({n_major})")
    print(f"[INFO] Minority class = {minority_class} ({n_minor})")

    # Oversample minority by random choice *with replacement*
    extra_needed = n_major - n_minor
    print(f"[INFO] Oversampling minority by {extra_needed} samples...")

    balanced = []
    balanced.extend(by_class[majority_class])
    balanced.extend(by_class[minority_class])

    if extra_needed > 0:
        for _ in range(extra_needed):
            balanced.append(random.choice(by_class[minority_class]))

    # Shuffle to mix them
    random.shuffle(balanced)

    # Show new class counts
    new_counts = Counter(int(w["label"]) for w in balanced)
    print("[INFO] New balanced class counts:", dict(new_counts))

    return balanced


def main():
    print("[INFO] Loading window labels from:", LABELS_FILE)
    windows = load_window_labels()

    balanced = balance_windows(windows)

    print("[INFO] Saving balanced labels to:", OUT_FILE)
    with open(OUT_FILE, "w") as f:
        for w in balanced:
            f.write(json.dumps(w) + "\n")

    print("[SUCCESS] Balanced window labels file created.")


if __name__ == "__main__":
    main()
