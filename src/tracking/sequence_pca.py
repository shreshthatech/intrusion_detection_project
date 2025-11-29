"""
Task 39: Sequence speedups via PCA on high-dimensional features.

We simulate:
  - N sequences
  - T timesteps
  - D-dimensional features (e.g., LSTM hidden states)

Then:
  - Fit PCA to reduce D -> K
  - Show explained variance and shapes.
"""

import numpy as np
from sklearn.decomposition import PCA


def _demo():
    print("[DEMO] Sequence PCA (Task 39)")

    # Simulated parameters
    N = 50        # number of tracks
    T = 20        # timesteps per track
    D = 32        # original feature dimension (e.g., LSTM hidden size)
    K = 8         # compressed dimension

    # Simulate some feature sequences
    rng = np.random.default_rng(0)
    # Shape: (N, T, D)
    seq_feats = rng.normal(size=(N, T, D))

    # Reshape to (N*T, D) for PCA fit
    X = seq_feats.reshape(-1, D)  # (N*T, D)
    print("[INFO] Original feature matrix shape:", X.shape)

    # Fit PCA
    pca = PCA(n_components=K, random_state=0)
    X_reduced = pca.fit_transform(X)
    print("[INFO] Reduced feature matrix shape:", X_reduced.shape)

    # Reshape back to sequence form
    seq_feats_reduced = X_reduced.reshape(N, T, K)
    print("[INFO] Reduced sequence shape:", seq_feats_reduced.shape)

    # Explained variance
    evr = pca.explained_variance_ratio_
    print("\n[INFO] Explained variance ratio per component:")
    for i, v in enumerate(evr):
        print(f"  PC{i+1}: {v*100:.2f}%")

    print(f"\n[INFO] Total variance retained: {evr.sum()*100:.2f}%")

    print("\n[DEMO] Example original vs reduced feature:")
    print("  Original (first timestep, first track):")
    print("   ", X[0][:5], "...")  # first 5 dims
    print("  Reduced (first timestep, first track):")
    print("   ", X_reduced[0])

    print("\n[DEMO] Done.")


if __name__ == "__main__":
    _demo()