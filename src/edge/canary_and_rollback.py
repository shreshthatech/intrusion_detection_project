"""
Task 68: Canary tests + rollback for on-edge model updates

We simulate:
  - Old model (v1)
  - New model (v2)
  - Canary test:
      - Output stability test
      - Speed test
      - Memory usage estimate
  - Automatic rollback if canary fails
"""

import time
import random
import psutil
import torch
import torch.nn as nn


# ------------------ Simple TinyDetector used as dummy model ------------------ #

class TinyDetector(nn.Module):
    def __init__(self, scale=1.0):
        super().__init__()
        # scale controls model size (v2 may be larger)
        hidden = int(16 * scale)

        self.features = nn.Sequential(
            nn.Conv2d(3, hidden, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden, hidden, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.fc = nn.Linear(hidden, 1)

    def forward(self, x):
        f = self.features(x)
        f = f.view(f.size(0), -1)
        return self.fc(f)


# ------------------ Canary Test Logic ------------------ #

class CanaryTester:
    def __init__(self):
        pass

    def speed_test(self, model):
        model.eval()
        x = torch.randn(1, 3, 224, 224)

        # warm-up
        for _ in range(5):
            model(x)

        t0 = time.time()
        for _ in range(20):
            model(x)
        dt = time.time() - t0

        avg = dt / 20
        return avg

    def output_stability_test(self, m1, m2):
        """
        Compare outputs of old and new model.
        They should not diverge too much.
        """
        x = torch.randn(8, 3, 224, 224)
        y1 = m1(x).detach()
        y2 = m2(x).detach()

        mse = torch.mean((y1 - y2) ** 2).item()
        return mse

    def memory_test(self):
        """
        Simulate memory pressure.
        Returns current RAM percent.
        """
        return psutil.virtual_memory().percent


# ------------------ Model Manager ------------------ #

class ModelManager:
    def __init__(self):
        self.active_version = "v1"
        self.models = {
            "v1": TinyDetector(scale=1.0),
            "v2": TinyDetector(scale=1.5),  # larger model
        }
        print(f"[INFO] Active model = {self.active_version}")

    def load_model(self, version):
        return self.models[version]

    def deploy(self, new_version):
        print(f"\n[DEPLOY] Trying to deploy model {new_version}")
        tester = CanaryTester()

        old_model = self.load_model(self.active_version)
        new_model = self.load_model(new_version)

        # ---------------- Canary Tests ---------------- #
        print("[TEST] Running canary tests...")

        speed = tester.speed_test(new_model)
        mse = tester.output_stability_test(old_model, new_model)
        mem = tester.memory_test()

        print(f"  Speed test: {speed:.6f} sec")
        print(f"  Output MSE: {mse:.6f}")
        print(f"  RAM usage: {mem:.1f}%")

        # ---------------- Conditions ---------------- #
        if speed > 0.020:
            print("[FAIL] New model too slow!")
            return False

        if mse > 1.0:  # large difference from old model
            print("[FAIL] Output stability too low!")
            return False

        if mem > 95.0:
            print("[FAIL] System low on memory!")
            return False

        # If all tests pass → promote
        self.active_version = new_version
        print(f"[SUCCESS] Promoted {new_version} to active!")
        return True

    def rollback(self):
        print("[ROLLBACK] Rolling back to v1...")
        self.active_version = "v1"
        print("[INFO] Rollback complete.")


# ------------------ DEMO ------------------ #

def _demo():
    print("[DEMO] Task 68 – Canary Tests + Rollback")

    mgr = ModelManager()

    # Try upgrade from v1 → v2
    ok = mgr.deploy("v2")

    if not ok:
        mgr.rollback()

    print(f"\n[FINAL] Active model is: {mgr.active_version}")
    print("[DEMO] Task 68 completed.")


if __name__ == "__main__":
    _demo()
