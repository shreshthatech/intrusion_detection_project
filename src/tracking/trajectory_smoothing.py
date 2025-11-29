"""
Task 36: Trajectory smoothing via Savitzky-Golay on track history.

We:
  - define a simple Savitzky-Golay smoother for 1D sequences (no SciPy needed),
  - apply it to a synthetic noisy trajectory (x(t), y(t)),
  - show before/after values for the first few points.
"""

from dataclasses import dataclass
from typing import List, Tuple
import numpy as np


@dataclass
class TrackPoint:
    t: float
    x: float
    y: float


def savgol_1d(y: np.ndarray, window_size: int = 7, poly_order: int = 2) -> np.ndarray:
    """
    Simple Savitzky-Golay filter for 1D data using local poly fit.

    y: array of shape (N,)
    window_size: odd integer, number of points in the local window
    poly_order: polynomial order (e.g. 2 for quadratic)

    Returns smoothed y of shape (N,).
    """
    if window_size % 2 == 0 or window_size < 3:
        raise ValueError("window_size must be odd and >= 3")
    if poly_order >= window_size:
        raise ValueError("poly_order must be < window_size")

    N = len(y)
    half = window_size // 2
    y_smooth = np.zeros_like(y, dtype=float)

    # For each position, fit polynomial to its local window and evaluate at center
    for i in range(N):
        # Determine window indices
        start = max(0, i - half)
        end = min(N, i + half + 1)
        x_window = np.arange(start, end)
        y_window = y[start:end]

        # Center x for numerical stability
        x_center = x_window - x_window.mean()

        # Build Vandermonde matrix
        A = np.vander(x_center, N=poly_order + 1, increasing=True)  # (W, poly_order+1)

        # Solve least squares for polynomial coeffs
        coeffs, *_ = np.linalg.lstsq(A, y_window, rcond=None)

        # Evaluate at x=0 (center)
        y_smooth[i] = coeffs[0]

    return y_smooth


def smooth_trajectory(points: List[TrackPoint],
                      window_size: int = 7,
                      poly_order: int = 2) -> List[TrackPoint]:
    """
    Apply Savitzky-Golay smoothing separately to x(t) and y(t).
    """
    if len(points) < window_size:
        # Not enough points to smooth, just return original
        return points

    ts = np.array([p.t for p in points], dtype=float)
    xs = np.array([p.x for p in points], dtype=float)
    ys = np.array([p.y for p in points], dtype=float)

    xs_smooth = savgol_1d(xs, window_size=window_size, poly_order=poly_order)
    ys_smooth = savgol_1d(ys, window_size=window_size, poly_order=poly_order)

    smoothed = [
        TrackPoint(t=float(ts[i]), x=float(xs_smooth[i]), y=float(ys_smooth[i]))
        for i in range(len(points))
    ]
    return smoothed


# -------------- DEMO -------------- #

def _demo():
    print("[DEMO] Trajectory smoothing (Task 36)")

    # Create synthetic trajectory: a smooth curve + noise
    np.random.seed(0)
    N = 30
    ts = np.linspace(0, 3, N)
    xs = 10 + 5 * np.sin(ts) + np.random.normal(scale=0.5, size=N)
    ys = 20 + 3 * np.cos(ts) + np.random.normal(scale=0.5, size=N)

    raw_points = [TrackPoint(t=float(t), x=float(x), y=float(y))
                  for t, x, y in zip(ts, xs, ys)]

    smoothed_points = smooth_trajectory(raw_points, window_size=7, poly_order=2)

    print("\nFirst 10 points (raw vs smoothed):")
    for i in range(10):
        rp = raw_points[i]
        sp = smoothed_points[i]
        print(f"  t={rp.t:.2f}  raw_x={rp.x:.2f}, raw_y={rp.y:.2f}  "
              f"--> smooth_x={sp.x:.2f}, smooth_y={sp.y:.2f}")

    print("\n[DEMO] Done.")


if __name__ == "__main__":
    _demo()
