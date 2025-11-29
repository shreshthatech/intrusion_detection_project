"""
Task 62: Structured pruning (channels) with accuracy guardrails.

We:
  - Load TinyMotionCNN with trained weights
  - Apply channel-wise (structured) pruning to conv layers
  - Remove pruning reparam to get a smaller dense model
  - Compare:
      - number of parameters
      - MSE between original and pruned outputs on random data
"""

from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import torch.nn.functional as F


class TinyMotionCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(2, 8, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(8, 16, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 1, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x)


def count_nonzero_params(model: nn.Module):
    total = 0
    nonzero = 0
    for p in model.parameters():
        numel = p.numel()
        total += numel
        nonzero += (p != 0).sum().item()
    return total, nonzero


def main():
    print("[INFO] Task 62 – Structured pruning for TinyMotionCNN")

    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    MODEL_PATH = PROJECT_ROOT / "models" / "vision" / "tiny_motion_cnn.pt"
    PRUNED_PATH = PROJECT_ROOT / "models" / "vision" / "tiny_motion_cnn_pruned.pt"

    # ----- Load original model -----
    model_orig = TinyMotionCNN()
    model_orig.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    model_orig.eval()

    # Clone for pruning
    model_pruned = TinyMotionCNN()
    model_pruned.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    model_pruned.eval()

    # ----- Count params before pruning -----
    total, nonzero = count_nonzero_params(model_pruned)
    print(f"[BEFORE PRUNING] total_params={total}, nonzero={nonzero}")

    # ----- Apply structured pruning (channel-wise) -----
    # Prune 30% of output channels in conv layers based on L2 norm
    conv1 = model_pruned.net[0]  # Conv2d(2, 8, ...)
    conv2 = model_pruned.net[2]  # Conv2d(8, 16, ...)
    conv3 = model_pruned.net[4]  # Conv2d(16, 1, ...), skip pruning here (only 1 out channel)

    amount = 0.3  # 30%

    print(f"[INFO] Applying ln_structured pruning with amount={amount} on conv1, conv2")
    prune.ln_structured(conv1, name="weight", amount=amount, n=2, dim=0)
    prune.ln_structured(conv2, name="weight", amount=amount, n=2, dim=0)

    # After pruning, remove the reparam (make zeros permanent)
    prune.remove(conv1, "weight")
    prune.remove(conv2, "weight")

    # ----- Count params after pruning -----
    total_p, nonzero_p = count_nonzero_params(model_pruned)
    print(f"[AFTER PRUNING] total_params={total_p}, nonzero={nonzero_p}")

    # ----- Accuracy guardrail: compare outputs on random data -----
    with torch.inference_mode():
        x = torch.randn(16, 2, 90, 160)  # batch of 16 random samples
        y_orig = model_orig(x)
        y_pruned = model_pruned(x)
        mse = F.mse_loss(y_pruned, y_orig).item()

    print(f"[GUARDRAIL] MSE between original and pruned outputs: {mse:.6f}")

    # You can define a simple guardrail threshold, e.g. 0.01
    threshold = 0.01
    if mse > threshold:
        print(f"[WARN] MSE={mse:.6f} > {threshold}, pruning may be too aggressive.")
    else:
        print(f"[OK] MSE={mse:.6f} within acceptable range (threshold={threshold}).")

    # ----- Save pruned model -----
    torch.save(model_pruned.state_dict(), PRUNED_PATH)
    print(f"[SUCCESS] Saved pruned model to: {PRUNED_PATH}")

    print("\n[DEMO] Task 62 structured pruning completed.")


if __name__ == "__main__":
    main()
