import torch
import torch.nn as nn
import time
from pathlib import Path

# Try to set quantization backend (for x86 this is usually 'fbgemm')
try:
    torch.backends.quantized.engine = "fbgemm"
except Exception:
    pass


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


def quantize_model():
    print("[INFO] Starting INT8 Static Quantization (Task 61)")

    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    MODEL_PATH = PROJECT_ROOT / "models" / "vision" / "tiny_motion_cnn.pt"
    OUT_PATH = PROJECT_ROOT / "models" / "vision" / "tiny_motion_cnn_int8.pt"

    print(f"[INFO] Loading model from: {MODEL_PATH}")
    model = TinyMotionCNN()
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    model.eval()

    # Example input for calibration
    example = torch.randn(1, 2, 90, 160)

    # ---- Step 1: Fuse conv+relu blocks ----
    fused = torch.quantization.fuse_modules(
        model,
        [["net.0", "net.1"], ["net.2", "net.3"]],
        inplace=False,
    )
    print("[INFO] Layers fused.")

    # ---- Step 2: Prepare + convert to quantized ----
    fused.qconfig = torch.quantization.get_default_qconfig("fbgemm")
    torch.quantization.prepare(fused, inplace=True)
    # calibration
    with torch.inference_mode():
        fused(example)

    quantized = torch.quantization.convert(fused, inplace=False)
    print("[INFO] Model converted to quantized version.")

    # ---- Step 3: Save quantized weights ----
    torch.save(quantized.state_dict(), OUT_PATH)
    print(f"[SUCCESS] Saved INT8 quantized model to: {OUT_PATH}")

    # ---- Step 4: Compare model sizes ----
    fp32_size = MODEL_PATH.stat().st_size / 1024
    int8_size = OUT_PATH.stat().st_size / 1024

    print("\n[MODEL SIZE]")
    print(f"  FP32: {fp32_size:.2f} KB")
    print(f"  INT8: {int8_size:.2f} KB")

    # ---- Step 5: Speed test (FP32 always works) ----
    inputs = torch.randn(1, 2, 90, 160)
    with torch.inference_mode():
        t0 = time.time()
        for _ in range(200):
            model(inputs)
        fp32_time = time.time() - t0

    print("\n[SPEED TEST]")
    print(f"  FP32 time (200 runs): {fp32_time:.4f} sec")

    # Try speed-test on quantized model, but don't crash if backend unsupported
    try:
        with torch.inference_mode():
            t0 = time.time()
            for _ in range(200):
                quantized(inputs)
            int8_time = time.time() - t0
        print(f"  INT8 time (200 runs): {int8_time:.4f} sec")
    except Exception as e:
        print("  [WARN] Could not run quantized model on this PyTorch build.")
        print(f"         Backend/ops not available. Details: {e}")

    print("\n[DEMO] Task 61 quantization flow completed.")


if __name__ == "__main__":
    quantize_model()
