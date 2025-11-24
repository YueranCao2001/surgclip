import json
import os
import matplotlib.pyplot as plt
import numpy as np

# Root paths (assume you run this from the project root)
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(ROOT_DIR, "results")
FIG_DIR = os.path.join(RESULTS_DIR, "figures")

os.makedirs(FIG_DIR, exist_ok=True)

def load_scores(json_path):
    """Load top-1 and top-5 accuracy from a retrieval_scores JSON file."""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    # Keys are exactly the ones printed in your logs
    top1 = data.get("top1_accuracy", 0.0)
    top5 = data.get("top5_accuracy", 0.0)
    return top1, top5

def main():
    # ---- 1. Define file paths ----
    baseline_json = os.path.join(RESULTS_DIR, "retrieval_scores.json")
    adapted_fp32_json = os.path.join(RESULTS_DIR, "retrieval_scores_adapted.json")
    adapted_int8_json = os.path.join(RESULTS_DIR, "retrieval_scores_adapted_int8.json")

    # ---- 2. Load metrics ----
    top1_baseline, top5_baseline = load_scores(baseline_json)
    top1_fp32, top5_fp32 = load_scores(adapted_fp32_json)
    top1_int8, top5_int8 = load_scores(adapted_int8_json)

    print("[INFO] Baseline     - top1: %.4f, top5: %.4f" % (top1_baseline, top5_baseline))
    print("[INFO] Adapted FP32 - top1: %.4f, top5: %.4f" % (top1_fp32, top5_fp32))
    print("[INFO] Adapted INT8 - top1: %.4f, top5: %.4f" % (top1_int8, top5_int8))

    # ---- 3. Prepare data for plotting ----
    models = ["baseline", "adapter_fp32", "adapter_int8"]
    top1_values = [top1_baseline, top1_fp32, top1_int8]
    top5_values = [top5_baseline, top5_fp32, top5_int8]

    x = np.arange(len(models))
    width = 0.35  # bar width

    # ---- 4. Plot grouped bar chart ----
    fig, ax = plt.subplots(figsize=(8, 5))

    rects1 = ax.bar(x - width / 2, top1_values, width, label="Top-1")
    rects2 = ax.bar(x + width / 2, top5_values, width, label="Top-5")

    ax.set_ylabel("Accuracy")
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_ylim(0.0, 1.05)
    ax.set_title("Overall Top-1 / Top-5 Accuracy (FP32 vs INT8)")
    ax.legend()

    # Add value labels on top of bars
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f"{height:.2f}",
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha="center", va="bottom", fontsize=9)

    autolabel(rects1)
    autolabel(rects2)

    fig.tight_layout()

    out_path = os.path.join(FIG_DIR, "overall_accuracy_bar_int8.png")
    plt.savefig(out_path, dpi=300)
    print(f"[INFO] Saved figure to {out_path}")

if __name__ == "__main__":
    main()
