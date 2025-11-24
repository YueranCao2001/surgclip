import os
import time
import json
import torch
import open_clip
import numpy as np
from PIL import Image
import torchvision.transforms as T
import matplotlib.pyplot as plt
import torch.nn.functional as F

# ==============================
#  Paths & basic config
# ==============================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Any existing frame from your dataset
IMAGE_PATH = "frames/video01/frame_000002.jpg"

# Trained FP32 adapter checkpoint (your best adapter)
ADAPTER_FP32_PATH = "results/clip_adapter_best.pth"

RESULT_DIR = "results"
FIG_DIR = os.path.join(RESULT_DIR, "figures_int8")
os.makedirs(FIG_DIR, exist_ok=True)

NUM_ITERS_SPEED = 500   # iterations for latency benchmark


# ==============================
#  Utils
# ==============================
def save_csv(path, header, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(",".join(header) + "\n")
        for r in rows:
            f.write(",".join(str(x) for x in r) + "\n")
    print(f"[INFO] Saved CSV to {path}")


def build_adapter_from_state(state_dict, device):
    """
    Rebuild the 2-layer MLP adapter from a state_dict.

    We expect keys like:
      '0.weight', '0.bias', '2.weight', '2.bias'
    corresponding to:
      Linear(in_dim -> hidden_dim), ReLU, Linear(hidden_dim -> out_dim)
    """
    # Handle nested dicts if necessary (e.g., {'state_dict': {...}})
    if isinstance(state_dict, dict) and "state_dict" in state_dict \
            and isinstance(state_dict["state_dict"], dict):
        state_dict = state_dict["state_dict"]

    w0 = state_dict["0.weight"]  # [hidden_dim, in_dim]
    w2 = state_dict["2.weight"]  # [out_dim, hidden_dim]

    in_dim = w0.shape[1]
    hidden_dim = w0.shape[0]
    out_dim = w2.shape[0]

    adapter = torch.nn.Sequential(
        torch.nn.Linear(in_dim, hidden_dim),
        torch.nn.ReLU(),
        torch.nn.Linear(hidden_dim, out_dim),
    )
    adapter.load_state_dict(state_dict)
    adapter.to(device)
    adapter.eval()
    return adapter


# ==============================
#  Load CLIP + adapters
# ==============================
def load_clip_and_adapters():
    print(f"[INFO] Loading CLIP backbone on {DEVICE} ...")

    # CLIP backbone still runs on GPU (if available)
    clip_model, _, _ = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="openai", device=DEVICE
    )
    clip_model.eval()

    # Image preprocessing
    transform = T.Compose(
        [
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(
                (0.48145466, 0.4578275, 0.40821073),
                (0.26862954, 0.26130258, 0.27577711),
            ),
        ]
    )
    img = transform(Image.open(IMAGE_PATH)).unsqueeze(0).to(DEVICE)

    # ----- load FP32 adapter on CPU -----
    state_fp32 = torch.load(ADAPTER_FP32_PATH, map_location="cpu")
    adapter_fp32_cpu = build_adapter_from_state(state_fp32, device="cpu")

    # ----- build REAL INT8 adapter via dynamic quantization (CPU only) -----
    adapter_int8_cpu = torch.quantization.quantize_dynamic(
        adapter_fp32_cpu, {torch.nn.Linear}, dtype=torch.qint8
    )
    adapter_int8_cpu.eval()

    # Save INT8 adapter state_dict for size comparison
    int8_path = os.path.join(RESULT_DIR, "clip_adapter_int8_cpu.pth")
    torch.save(adapter_int8_cpu.state_dict(), int8_path)
    print(f"[INFO] Saved INT8 adapter (CPU) state_dict to {int8_path}")

    return clip_model, adapter_fp32_cpu, adapter_int8_cpu, img


# ==============================
#  Experiment 1: latency (CPU-only adapters)
# ==============================
def benchmark_speed(clip_model, adapter_fp32_cpu, adapter_int8_cpu, img):
    print("\n[EXP1] Benchmarking FP32 vs INT8 adapter latency (CPU) ...")

    clip_model.eval()

    # 1) Encode image with CLIP on DEVICE (GPU if available)
    with torch.no_grad():
        feat_gpu = clip_model.encode_image(img.to(DEVICE))
    # 2) Move features to CPU; adapters will run on CPU
    feat_cpu = feat_gpu.cpu()

    @torch.no_grad()
    def run_adapter(adapter, n_iter, desc):
        # Warm-up
        for _ in range(10):
            _ = adapter(feat_cpu)
        t0 = time.time()
        for _ in range(n_iter):
            _ = adapter(feat_cpu)
        t1 = time.time()
        latency = (t1 - t0) / n_iter
        print(f"  {desc} avg latency: {latency * 1000:.3f} ms/frame")
        return latency

    lat_fp32 = run_adapter(adapter_fp32_cpu, NUM_ITERS_SPEED, "FP32 adapter (CPU)")
    lat_int8 = run_adapter(adapter_int8_cpu, NUM_ITERS_SPEED, "INT8 adapter (CPU)")

    # CSV
    csv_path = os.path.join(RESULT_DIR, "int8_speed_benchmark.csv")
    save_csv(
        csv_path,
        header=["model", "avg_latency_ms"],
        rows=[
            ["adapter_fp32_cpu", f"{lat_fp32 * 1000:.4f}"],
            ["adapter_int8_cpu", f"{lat_int8 * 1000:.4f}"],
        ],
    )

    # Bar plot
    labels = ["adapter_fp32_cpu", "adapter_int8_cpu"]
    latencies_ms = [lat_fp32 * 1000, lat_int8 * 1000]

    plt.figure(figsize=(6, 4))
    bars = plt.bar(labels, latencies_ms)
    for bar, v in zip(bars, latencies_ms):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.5,
            f"{v:.2f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )
    plt.ylabel("Latency (ms / frame)")
    plt.title("Inference Latency (Adapter only, CPU): FP32 vs INT8")
    plt.tight_layout()

    fig_path = os.path.join(FIG_DIR, "int8_speed_benchmark.png")
    plt.savefig(fig_path, dpi=300)
    plt.close()
    print(f"[INFO] Saved speed figure to {fig_path}")

    return lat_fp32, lat_int8, feat_cpu


# ==============================
#  Experiment 2: file size & params
# ==============================
def benchmark_model_size_and_params(adapter_fp32_cpu, adapter_int8_cpu):
    print("\n[EXP2] Comparing adapter file size & parameter count ...")

    def get_file_mb(path):
        return os.path.getsize(path) / 1024 / 1024

    def count_params(m):
        return sum(p.numel() for p in m.parameters())

    size_fp32 = get_file_mb(ADAPTER_FP32_PATH)

    int8_path = os.path.join(RESULT_DIR, "clip_adapter_int8_cpu.pth")
    size_int8 = get_file_mb(int8_path)

    params_fp32 = count_params(adapter_fp32_cpu)
    params_int8 = count_params(adapter_int8_cpu)

    print(f"  FP32 adapter size : {size_fp32:.3f} MB, params: {params_fp32}")
    print(f"  INT8 adapter size : {size_int8:.3f} MB, params: {params_int8}")
    if size_int8 > 0:
        print(f"  Compression ratio : {size_fp32 / size_int8:.2f}x")

    # CSV
    csv_path = os.path.join(RESULT_DIR, "int8_model_size_comparison.csv")
    save_csv(
        csv_path,
        header=["model", "file_size_MB", "num_params"],
        rows=[
            ["adapter_fp32", f"{size_fp32:.6f}", params_fp32],
            ["adapter_int8_cpu", f"{size_int8:.6f}", params_int8],
        ],
    )

    # Bar plot (file size)
    labels = ["adapter_fp32", "adapter_int8_cpu"]
    sizes = [size_fp32, size_int8]

    plt.figure(figsize=(6, 4))
    bars = plt.bar(labels, sizes)
    for bar, v in zip(bars, sizes):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{v:.2f} MB",
            ha="center",
            va="bottom",
            fontsize=10,
        )
    plt.ylabel("File size (MB)")
    plt.title("Adapter File Size: FP32 vs INT8 (CPU)")
    plt.tight_layout()
    fig_path = os.path.join(FIG_DIR, "int8_model_size_comparison.png")
    plt.savefig(fig_path, dpi=300)
    plt.close()
    print(f"[INFO] Saved size figure to {fig_path}")

    return size_fp32, size_int8, params_fp32, params_int8


# ==============================
#  Experiment 3: representation error
# ==============================
def benchmark_representation_error(adapter_fp32_cpu, adapter_int8_cpu, feat_cpu):
    """
    Compare FP32 vs INT8 adapter outputs on the same CLIP feature.
    We report L2 error, mean abs error, max abs error and cosine similarity,
    and also save a histogram of per-dimension differences.
    """
    print("\n[EXP3] Measuring representation error (FP32 vs INT8) ...")

    adapter_fp32_cpu.eval()
    adapter_int8_cpu.eval()

    with torch.no_grad():
        out_fp32 = adapter_fp32_cpu(feat_cpu)
        out_int8 = adapter_int8_cpu(feat_cpu)

    diff = (out_fp32 - out_int8).view(-1).cpu().numpy()
    l2_err = float(np.linalg.norm(diff))
    mean_abs = float(np.mean(np.abs(diff)))
    max_abs = float(np.max(np.abs(diff)))

    v1 = out_fp32.view(-1)
    v2 = out_int8.view(-1)
    cos_sim = float(F.cosine_similarity(v1, v2, dim=0).item())

    print(f"  L2 error            : {l2_err:.6f}")
    print(f"  Mean |diff|         : {mean_abs:.6f}")
    print(f"  Max  |diff|         : {max_abs:.6f}")
    print(f"  Cosine similarity   : {cos_sim:.6f}")

    # CSV with one row
    csv_path = os.path.join(RESULT_DIR, "int8_representation_error.csv")
    save_csv(
        csv_path,
        header=["l2_error", "mean_abs_error", "max_abs_error", "cosine_similarity"],
        rows=[[f"{l2_err:.8f}", f"{mean_abs:.8f}", f"{max_abs:.8f}", f"{cos_sim:.8f}"]],
    )

    # Histogram figure
    plt.figure(figsize=(6, 4))
    plt.hist(diff, bins=50)
    plt.xlabel("FP32 - INT8 adapter output difference")
    plt.ylabel("Frequency")
    plt.title("Per-dimension Output Difference: FP32 vs INT8")
    plt.tight_layout()
    fig_path = os.path.join(FIG_DIR, "int8_representation_error_hist.png")
    plt.savefig(fig_path, dpi=300)
    plt.close()
    print(f"[INFO] Saved representation error histogram to {fig_path}")

    return l2_err, mean_abs, max_abs, cos_sim


# ==============================
#  Main
# ==============================
def main():
    clip_model, adapter_fp32_cpu, adapter_int8_cpu, img = load_clip_and_adapters()

    # Experiment 1: latency (CPU)
    lat_fp32, lat_int8, feat_cpu = benchmark_speed(
        clip_model, adapter_fp32_cpu, adapter_int8_cpu, img
    )

    # Experiment 2: file size & params
    size_fp32, size_int8, params_fp32, params_int8 = benchmark_model_size_and_params(
        adapter_fp32_cpu, adapter_int8_cpu
    )

    # Experiment 3: representation error
    l2_err, mean_abs, max_abs, cos_sim = benchmark_representation_error(
        adapter_fp32_cpu, adapter_int8_cpu, feat_cpu
    )

    # Summary JSON
    summary = {
        "device": DEVICE,
        "notes": (
            "INT8 adapter is built via torch.quantization.quantize_dynamic "
            "and runs on CPU only. CLIP backbone still runs on GPU if available."
        ),
        "latency_ms_cpu": {
            "adapter_fp32": lat_fp32 * 1000,
            "adapter_int8": lat_int8 * 1000,
        },
        "file_size_MB": {
            "adapter_fp32": size_fp32,
            "adapter_int8": size_int8,
        },
        "num_params": {
            "adapter_fp32": params_fp32,
            "adapter_int8": params_int8,
        },
        "representation_error": {
            "l2_error": l2_err,
            "mean_abs_error": mean_abs,
            "max_abs_error": max_abs,
            "cosine_similarity": cos_sim,
        },
    }

    json_path = os.path.join(RESULT_DIR, "int8_benchmark_summary.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[INFO] Saved summary JSON to {json_path}")
    print("[DONE] INT8 benchmark (CPU-only dynamic quantization) finished.")


if __name__ == "__main__":
    main()
