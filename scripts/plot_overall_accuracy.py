import os
import pandas as pd
import matplotlib.pyplot as plt

# Paths
RESULTS_DIR = "results"
CSV_PATH = os.path.join(RESULTS_DIR, "overall_accuracy_summary.csv")
FIG_DIR = os.path.join(RESULTS_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

def main():
    # Load summary CSV
    df = pd.read_csv(CSV_PATH)

    # We expect columns: run_name, top1_acc, top5_acc
    runs = df["run_name"].tolist()
    top1 = df["top1_acc"].tolist()
    top5 = df["top5_acc"].tolist()

    x = range(len(runs))
    width = 0.35

    plt.figure(figsize=(6, 4))
    # Bars for Top-1 and Top-5
    plt.bar([i - width/2 for i in x], top1, width=width, label="Top-1")
    plt.bar([i + width/2 for i in x], top5, width=width, label="Top-5")

    plt.xticks(x, runs)
    plt.ylim(0.0, 1.05)
    plt.ylabel("Accuracy")
    plt.title("Overall Top-1 / Top-5 Accuracy")
    plt.legend()

    # Add value labels on top of bars
    for i, v in enumerate(top1):
        plt.text(i - width/2, v + 0.01, f"{v:.2f}", ha="center", va="bottom", fontsize=8)
    for i, v in enumerate(top5):
        plt.text(i + width/2, v + 0.01, f"{v:.2f}", ha="center", va="bottom", fontsize=8)

    out_path = os.path.join(FIG_DIR, "overall_accuracy_bar.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    print(f"[INFO] Saved figure to {out_path}")

if __name__ == "__main__":
    main()
