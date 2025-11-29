"""
Task 30: Export fused event timeline with uncertainty bands.

We will:
  - Load deduplicated events (from Task 28 demo logic)
  - Load FusionNet + MC Dropout wrapper
  - For each event, produce:
       * mean fused prob
       * variance
       * confidence bands
       * zone multiplier
  - Save everything to fused_timeline.jsonl
"""

import sys, os, json, math
from pathlib import Path

# Correct Path setup (do NOT use os.path here)
CURRENT = Path(__file__).resolve()          # .../src/fusion/export_timeline.py
FUSION_DIR = CURRENT.parent                 # .../src/fusion
SRC_DIR = FUSION_DIR.parent                 # .../src
PROJECT_ROOT = SRC_DIR.parent               # .../intrusion_detection_project

# Add src to PYTHONPATH
sys.path.append(str(SRC_DIR))


from fusion.event_dedup import Event, EventDeduplicator
from fusion.zone_prior import ZonePrior
from fusion.uncertainty_mc_dropout import MCDropoutWrapper
from fusion.learnable_fusion import FusionNet

import torch


FRAME_EVENTS = PROJECT_ROOT / "data" / "processed" / "api" / "frame_events.jsonl"
OUTPUT_FILE = PROJECT_ROOT / "data" / "processed" / "api" / "fused_timeline.jsonl"


def build_feature_vector(ev: Event):
    """
    For now, create a simple fake feature vector based on bbox & frame index.
    (In real systems, you'd use actual fused sensor embeddings.)
    """
    if ev.bbox is None:
        return torch.randn(1, 128)      # fallback
    x1, y1, x2, y2 = ev.bbox
    feats = torch.tensor([
        x1, y1, x2, y2,
        ev.frame,
        ev.p,
        len(ev.extra.get("sensors", []))
    ], dtype=torch.float32)
    feats = feats.unsqueeze(0)

    # Project to 128 dims for FusionNet input
    return torch.randn(1, 128) + 0.01 * feats.mean()


def run_export():
    print("[INFO] Loading video events...")

    # ---------------- Load events ---------------- #
    events = []
    with open(FRAME_EVENTS, "r") as f:
        for i, line in enumerate(f):
            if i > 100:
                break
            obj = json.loads(line)
            events.append(
                Event(
                    t=float(obj["t"]),
                    frame=int(obj["frame"]),
                    sensor="video",
                    type=obj["type"],
                    bbox=obj["bbox"],
                    p=float(obj["p"]),
                    extra={"source": "video"},
                )
            )

    # Also add synthetic audio events like Task 28 did
    fake_audio = []
    for ev in events[::20]:
        fake_audio.append(
            Event(
                t=ev.t + 0.2,
                frame=ev.frame,
                sensor="audio",
                type=ev.type,
                bbox=None,
                p=0.4,
                extra={"source": "audio"},
            )
        )

    all_events = events + fake_audio

    # ---------------- Deduplicate ---------------- #
    print("[INFO] Deduplicating...")
    dedup = EventDeduplicator(time_window=0.5, iou_thresh=0.3)
    merged = dedup.dedup(all_events)
    print(f"[INFO] Dedup → {len(merged)} final events.")

    # ---------------- Fusion Model ---------------- #
    dims = {
        "video": 128,
        "audio": 128,
        "rf": 128,
        "thermal": 128
    }
    fusion = FusionNet(dims=dims, proj_dim=64, fused_dim=128)
    mc = MCDropoutWrapper(fusion, dropout_p=0.3)

    # ---------------- Zone Prior ---------------- #
    zp = ZonePrior()

    # ---------------- Output File ---------------- #
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_FILE, "w") as out:
        for ev in merged:
            feats = {
                "video": build_feature_vector(ev),
                "audio": build_feature_vector(ev),
                "rf": build_feature_vector(ev),
                "thermal": build_feature_vector(ev),
            }

            mean, var = mc.mc_forward(feats, n_samples=20)
            var_scalar = float(var.mean())
            std = math.sqrt(var_scalar)

            lower = float(mean.mean() - 1.96 * std)
            upper = float(mean.mean() + 1.96 * std)

            zone_mult = zp.get_multiplier(ev.frame, ev.frame)

            result = {
                "t": ev.t,
                "frame": ev.frame,
                "sensor": ev.sensor,
                "type": ev.type,
                "bbox": ev.bbox,
                "p_mean": float(mean.mean()),
                "p_var": var_scalar,
                "p_lower": lower,
                "p_upper": upper,
                "zone_multiplier": zone_mult,
                "sensors": ev.extra.get("sensors"),
            }

            out.write(json.dumps(result) + "\n")

    print("[SUCCESS] Timeline saved to:", OUTPUT_FILE)


# ---------------- DEMO ---------------- #
if __name__ == "__main__":
    run_export()
