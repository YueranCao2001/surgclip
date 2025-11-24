import csv
import matplotlib.pyplot as plt
import numpy as np
import os

csv_path = "results/phase_wise_accuracy.csv"

phase_names = []
baseline_top1 = []
baseline_top5 = []
adapted_top1 = []
adapted_top5 = []

with open(csv_path, "r") as f:
    reader = csv.DictReader(f)
    for row in reader:
        phase_names.append(row["phase_name"])
        baseline_top1.append(float(row["baseline_top1"]))
        baseline_top5.append(float(row["baseline_top5"]))
        adapted_top1.append(float(row["adapted_top1"]))
        adapted_top5.append(float(row["adapted_top5"]))

x = np.arange(len(phase_names))
width = 0.2

plt.figure(figsize=(14,6))
plt.bar(x - 1.5*width, baseline_top1, width, label="Baseline Top-1", color="#4C72B0")
plt.bar(x - 0.5*width, adapted_top1, width, label="Adapted Top-1", color="#55A868")
plt.bar(x + 0.5*width, baseline_top5, width, label="Baseline Top-5", color="#C44E52")
plt.bar(x + 1.5*width, adapted_top5, width, label="Adapted Top-5", color="#8172B3")

plt.xticks(x, phase_names, rotation=30, ha="right")
plt.ylabel("Accuracy")
plt.title("Phase-wise Accuracy (Baseline vs Adapted)")
plt.ylim(0, 1.05)
plt.grid(axis="y", linestyle="--", alpha=0.5)
plt.legend()

os.makedirs("results/figures_phase", exist_ok=True)
plt.tight_layout()
plt.savefig("results/figures_phase/phase_accuracy_grouped_bar.png", dpi=300)
plt.show()

print("[INFO] Saved to results/figures_phase/")
