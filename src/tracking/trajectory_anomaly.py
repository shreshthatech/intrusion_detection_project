"""
Task 37: Trajectory anomaly detection using Isolation Forest.

We:
 - Build synthetic "normal" and "abnormal" trajectories
 - Extract simple statistical features per trajectory
 - Fit an IsolationForest
 - Predict which tracks are anomalies
"""

import numpy as np
from sklearn.ensemble import IsolationForest


# -----------------------------------------------------------
# Synthetic trajectory generator
# -----------------------------------------------------------

def generate_normal_trajectory(N=30):
    """
    Smooth mostly straight motion with small noise.
    """
    x, y = 0.0, 0.0
    dx = 0.03 + np.random.uniform(0.0, 0.02)
    dy = 0.02 + np.random.uniform(0.0, 0.02)

    xs, ys = [], []
    for i in range(N):
        x += dx + np.random.normal(scale=0.003)
        y += dy + np.random.normal(scale=0.003)
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)


def generate_abnormal_trajectory(N=30):
    """
    Sharp turns, jitter, irregular motion.
    """
    x, y = 0.0, 0.0
    xs, ys = [], []
    for i in range(N):
        x += np.random.normal(scale=0.07)
        y += np.random.normal(scale=0.07)
        # occasional extreme jumps
        if np.random.rand() < 0.1:
            x += np.random.choice([-0.3, 0.3])
            y += np.random.choice([-0.3, 0.3])
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)


# -----------------------------------------------------------
# Feature extraction
# -----------------------------------------------------------

def extract_features(xs, ys):
    """
    Convert trajectory into numeric features.
    """
    dx = np.diff(xs)
    dy = np.diff(ys)
    speeds = np.sqrt(dx**2 + dy**2)

    total_disp = np.sqrt((xs[-1] - xs[0])**2 + (ys[-1] - ys[0])**2)
    avg_speed = speeds.mean()
    speed_var = speeds.var()
    direction_var = np.var(np.arctan2(dy, dx))

    return np.array([
        total_disp,
        avg_speed,
        speed_var,
        direction_var,
    ])


# -----------------------------------------------------------
# DEMO
# -----------------------------------------------------------

def _demo():
    print("[DEMO] Isolation Forest Trajectory Anomaly Detection (Task 37)")

    # Generate dataset
    normals = [generate_normal_trajectory() for _ in range(40)]
    abnormals = [generate_abnormal_trajectory() for _ in range(10)]

    # Extract features
    X = []
    labels = []  # 1 = normal, -1 = abnormal (sklearn convention reversed later)
    for xs, ys in normals:
        X.append(extract_features(xs, ys))
        labels.append(1)

    for xs, ys in abnormals:
        X.append(extract_features(xs, ys))
        labels.append(-1)

    X = np.vstack(X)

    # Train Isolation Forest
    clf = IsolationForest(contamination=0.2, random_state=0)
    clf.fit(X)

    preds = clf.predict(X)  # 1 = normal, -1 = anomaly

    # Print results
    print("\n=== Results ===")
    for i, (true, pred) in enumerate(zip(labels, preds)):
        print(f"Track {i:02d}  true={'normal' if true==1 else 'abnormal'}  "
              f"pred={'normal' if pred==1 else 'abnormal'}")

    # Count accuracy
    correct = sum(1 for t, p in zip(labels, preds) if t == p)
    acc = correct / len(labels)
    print(f"\nAccuracy: {acc*100:.1f}%")
    print("\n[DEMO] Done.")


if __name__ == "__main__":
    _demo()
