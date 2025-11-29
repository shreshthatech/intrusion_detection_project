"""
Task 9: Train/val/test split preserving temporal order.

We take window_labels_balanced.jsonl, sort by start_time,
and then split into 60% train, 20% val, 20% test.
"""

from pathlib import Path
import json


PROJECT_ROOT = Path(__file__).resolve().parents[2]
IN_FILE = PROJECT_ROOT / "data" / "processed" / "windows" / "window_labels_balanced.jsonl"

OUT_DIR = PROJECT_ROOT / "data" / "processed" / "windows"
TRAIN_FILE = OUT_DIR / "train_windows.jsonl"
VAL_FILE = OUT_DIR / "val_windows.jsonl"
TEST_FILE = OUT_DIR / "test_windows.jsonl"


def load_windows():
    windows = []
    with open(IN_FILE, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            windows.append(json.loads(line))
    return windows


def split_temporal():
    print("[INFO] Loading balanced window labels from:", IN_FILE)
    windows = load_windows()

    # sort by start_time (temporal order)
    windows.sort(key=lambda w: w["start_time"])
    n = len(windows)
    print(f"[INFO] Total windows: {n}")

    n_train = int(0.6 * n)
    n_val = int(0.2 * n)
    n_test = n - n_train - n_val

    train = windows[:n_train]
    val = windows[n_train:n_train + n_val]
    test = windows[n_train + n_val:]

    print(f"[INFO] Split sizes -> train: {len(train)}, val: {len(val)}, test: {len(test)}")

    # Save
    def save_list(path, lst):
        with open(path, "w") as f:
            for item in lst:
                f.write(json.dumps(item) + "\n")

    save_list(TRAIN_FILE, train)
    save_list(VAL_FILE, val)
    save_list(TEST_FILE, test)

    print("[SUCCESS] Saved temporal splits to:")
    print("   Train:", TRAIN_FILE)
    print("   Val:  ", VAL_FILE)
    print("   Test: ", TEST_FILE)


if __name__ == "__main__":
    split_temporal()
