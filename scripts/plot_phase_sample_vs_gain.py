import os
import pandas as pd
import matplotlib.pyplot as plt

RESULTS_DIR = "results"
CSV_PATH = os.path.join(RESULTS_DIR, "phase_wise_accuracy.csv")
FIG_DIR = os.path.join(RESULTS_DIR, "figures_phase")
os.makedirs(FIG_DIR, exist_ok=True)

PHASE_NAMES = [
    "Preparation",
    "Calot Triangle Dissection",
    "Clipping & Cutting",
    "Gallbladder Dissection",
    "Gallbladder Packaging",
    "Cleaning & Coagulation",
    "Gallbladder Retraction",
]

def main():
    df = pd.read_csv(CSV_PATH)
    df = df.sort_values("phase_id")

    # Compute Top-1 gain
    df["delta_top1"] = df["adapted_top1"] - df["baseline_top1"]

    num_frames = df["num_frames"].values
    delta_top1 = df["delta_top1"].values

    # Normalize bubble sizes for visualization
    size = (num_frames / num_frames.max()) * 1000  # scale factor

    plt.figure(figsize=(6, 4))
    plt.scatter(num_frames, delta_top1, s=size, alpha=0.7)

    for i, name in enumerate(PHASE_NAMES):
        plt.text(num_frames[i], delta_top1[i] + 0.01, name,
                 ha="center", va="bottom", fontsize=8)

    plt.xlabel("Number of Frames per Phase")
    plt.ylabel("Δ Top-1 Accuracy (Adapted - Baseline)")
    plt.title("Phase Sample Size vs. Accuracy Gain")
    plt.axhline(0.0, color="gray", linestyle="--", linewidth=1)

    plt.tight_layout()
    out_path = os.path.join(FIG_DIR, "phase_sample_vs_gain.png")
    plt.savefig(out_path, dpi=300)
    print(f"[INFO] Saved figure to {out_path}")

if __name__ == "__main__":
    main()
