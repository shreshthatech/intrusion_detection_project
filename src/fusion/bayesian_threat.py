"""
Task 26: Bayesian updating of threat score as multimodal evidence streams in.

This class keeps a running estimate:
    P(intrusion | all evidence seen so far)

Sensors provide:
    - evidence probabilities (video/audio/rf/thermal)
    - reliability weights (0–1)
    - OOD distance penalties
    - thresholds from Task 24

Bayesian rule:
    posterior = prior * likelihood / normalization
"""

import json
from pathlib import Path
import math
from typing import Dict


PROJECT_ROOT = Path(__file__).resolve().parents[2]
THRESH_FILE = PROJECT_ROOT / "models" / "fusion" / "sensor_thresholds.json"


class BayesianThreatEstimator:
    def __init__(self, prior: float = 0.01):
        """
        prior: initial belief that any given second contains an intrusion.
        """
        self.prior = prior
        self.posterior = prior

        # Load thresholds
        with open(THRESH_FILE, "r") as f:
            self.thresholds = json.load(f)

    def _sensor_likelihood(self, p: float, thresh: float, reliability: float, ood: float) -> float:
        """
        Convert sensor raw probability to a likelihood ratio.

        Components:
            - threshold -> acts like a decision boundary
            - reliability -> (0–1) trust in sensor
            - ood -> Mahalanobis distance (higher = less trusted)

        Final likelihood is:
            L = reliability * (p / thresh) * exp(-ood)
        """
        # sensor with no calibration (None) → use passthrough
        if thresh is None:
            thresh = 0.5

        # Avoid divide by zero
        thresh = max(thresh, 1e-6)

        # Weight by OOD (high OOD => less confidence)
        ood_penalty = math.exp(-float(ood))

        L = reliability * (p / thresh) * ood_penalty

        # Clip extreme values
        return max(1e-6, min(L, 1e6))

    def update(self,
               p_video: float,
               p_audio: float,
               p_rf: float,
               p_thermal: float,
               reliability: Dict[str, float],
               ood: Dict[str, float]):
        """
        Bayesian update using incoming evidence from all sensors.

        Each sensor contributes multiplicatively to the likelihood.
        """
        # Get thresholds
        t_video = self.thresholds.get("video", 0.5)
        t_audio = self.thresholds.get("audio", 0.5)
        t_rf    = self.thresholds.get("rf", 0.5)
        t_th    = self.thresholds.get("thermal", 0.5)

        # Compute individual likelihoods
        L_video   = self._sensor_likelihood(p_video, t_video, reliability["video"], ood["video"])
        L_audio   = self._sensor_likelihood(p_audio, t_audio, reliability["audio"], ood["audio"])
        L_rf      = self._sensor_likelihood(p_rf,    t_rf,    reliability["rf"],    ood["rf"])
        L_thermal = self._sensor_likelihood(p_thermal, t_th,  reliability["thermal"], ood["thermal"])

        # Combine likelihoods
        combined_L = L_video * L_audio * L_rf * L_thermal

        # Bayesian posterior update
        prior = self.posterior
        numerator = prior * combined_L
        denominator = numerator + (1 - prior)

        self.posterior = numerator / max(denominator, 1e-9)

        return self.posterior


# ---------------- DEMO ---------------- #

def _demo():
    print("[DEMO] Bayesian Threat Estimator Demo")

    bt = BayesianThreatEstimator(prior=0.01)

    # Example sensor probabilities
    p = {
        "video": 0.7,
        "audio": 0.4,
        "rf": 0.2,
        "thermal": 0.3
    }

    # Example reliability weights
    reliability = {
        "video": 0.9,
        "audio": 0.8,
        "rf": 0.6,
        "thermal": 0.85
    }

    # Example OOD distances
    ood = {
        "video": 0.5,
        "audio": 1.0,
        "rf": 2.0,
        "thermal": 0.3
    }

    new_prob = bt.update(
        p_video=p["video"],
        p_audio=p["audio"],
        p_rf=p["rf"],
        p_thermal=p["thermal"],
        reliability=reliability,
        ood=ood
    )

    print("[DEMO] Updated threat probability:", new_prob)


if __name__ == "__main__":
    _demo()
