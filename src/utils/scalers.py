"""
Task 8: Feature standardization & min-max scaling utilities.

This module provides helper functions to:
- fit StandardScaler and MinMaxScaler on training data
- save them to disk
- load them later and apply to new data

For now we demo it using the RF CSV as "sensor features".
Later you will reuse these functions for audio/RF/vision embeddings.
"""

from pathlib import Path
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import joblib


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = PROJECT_ROOT / "models" / "scalers"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

RF_PATH = PROJECT_ROOT / "data" / "raw" / "rf" / "demo.csv"


def fit_scalers_on_rf():
    """
    Example/demo: fit scalers on RF 'power_dbm' feature.
    """
    print("[INFO] Loading RF data from:", RF_PATH)
    df = pd.read_csv(RF_PATH)

    # choose numeric feature columns (you can extend this later)
    feature_cols = ["power_dbm"]
    X = df[feature_cols].values

    std_scaler = StandardScaler()
    minmax_scaler = MinMaxScaler()

    X_std = std_scaler.fit_transform(X)
    X_mm = minmax_scaler.fit_transform(X)

    # Save scalers
    std_path = MODELS_DIR / "rf_power_std_scaler.joblib"
    mm_path = MODELS_DIR / "rf_power_minmax_scaler.joblib"

    joblib.dump(std_scaler, std_path)
    joblib.dump(minmax_scaler, mm_path)

    print("[SUCCESS] Fitted RF scalers and saved to:")
    print("   StandardScaler:", std_path)
    print("   MinMaxScaler:  ", mm_path)

    # Just to show shapes
    print("[INFO] Example transformed values:")
    print("   Standardized:", X_std[:5].ravel())
    print("   MinMax:      ", X_mm[:5].ravel())


def load_scaler(path: Path):
    """
    Generic helper to load a saved scaler.
    """
    return joblib.load(path)


if __name__ == "__main__":
    # Demo: fit scalers on RF power
    fit_scalers_on_rf()
