import os
import numpy as np
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

    # Compute deltas: adapted - baseline
    df["delta_top1"] = df["adapted_top1"] - df["baseline_top1"]
    df["delta_top5"] = df["adapted_top5"] - df["baseline_top5"]

    # Order by phase_id if needed
    df = df.sort_values("phase_id")

    # Build 2D matrix: phases x {delta_top1, delta_top5}
    data = np.vstack([df["delta_top1"].values, df["delta_top5"].values]).T

    fig, ax = plt.subplots(figsize=(5, 4))

    im = ax.imshow(data, cmap="bwr", aspect="auto", vmin=-1.0, vmax=1.0)

    # Axis ticks and labels
    ax.set_yticks(range(len(PHASE_NAMES)))
    ax.set_yticklabels(PHASE_NAMES)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Δ Top-1", "Δ Top-5"])

    # Rotate y labels for better spacing (optional)
    plt.setp(ax.get_yticklabels(), rotation=0, ha="right", rotation_mode="anchor")

    # Add numbers inside each cell
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            val = data[i, j]
            ax.text(
                j, i,
                f"{val:.2f}",
                ha="center", va="center",
                color="black" if abs(val) < 0.5 else "white",
                fontsize=8
            )

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Accuracy Gain (Adapted - Baseline)")

    ax.set_title("Phase-wise Accuracy Gain (Adapted vs Baseline)")

    plt.tight_layout()
    out_path = os.path.join(FIG_DIR, "phase_delta_heatmap.png")
    plt.savefig(out_path, dpi=300)
    print(f"[INFO] Saved figure to {out_path}")

if __name__ == "__main__":
    main()
