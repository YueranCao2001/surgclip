# scripts/reliability_analysis.py
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# --------------------------------------------------
# Basic paths
# --------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
RESULT_DIR = ROOT / "results"
FIG_DIR = RESULT_DIR / "figures_reliability"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# We fix one shift name here (matching generate_shift_scores.py)
SHIFT_NAME = "brightness_blur"

METHOD_FILES_NORMAL = {
    "baseline_fp32": RESULT_DIR / "retrieval_scores.json",
    "adapter_fp32": RESULT_DIR / "retrieval_scores_adapted.json",
    "adapter_int8": RESULT_DIR / "retrieval_scores_adapted_int8.json",
}

METHOD_FILES_SHIFT = {
    "baseline_fp32": RESULT_DIR / f"retrieval_scores_shift_{SHIFT_NAME}.json",
    "adapter_fp32": RESULT_DIR / f"retrieval_scores_adapted_shift_{SHIFT_NAME}.json",
    "adapter_int8": RESULT_DIR / f"retrieval_scores_adapted_int8_shift_{SHIFT_NAME}.json",
}


# --------------------------------------------------
# Loading & parsing JSON
# --------------------------------------------------
def load_scores_json(path: Path, method_name: str, condition: str) -> pd.DataFrame:
    """
    Load a retrieval_scores*.json file and convert to a DataFrame.

    Expected structure (your current files):
      {
        "num_frames": ...,
        "top1_accuracy": ...,
        "top5_accuracy": ...,
        "examples": [
          {
            "frame_path": "...",
            "true_phase_id": int,
            "pred_phase_ids": [int, ...],
            ...
          }, ...
        ]
      }
    """
    if not path.exists():
        raise FileNotFoundError(f"{path} does not exist.")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict) and "examples" in data:
        examples = data["examples"]
    elif isinstance(data, list):
        examples = data
    else:
        raise ValueError(
            f"Unexpected JSON structure in {path}. "
            "Expected a dict with key 'examples' or a list."
        )

    rows = []
    for ex in examples:
        true_id = int(ex["true_phase_id"])
        pred_ids = ex.get("pred_phase_ids", [])
        if not isinstance(pred_ids, list) or len(pred_ids) == 0:
            continue

        top1 = int(pred_ids[0])
        top5 = [int(pid) for pid in pred_ids[:5]]

        rows.append(
            {
                "frame_path": ex.get("frame_path", ""),
                "true_phase_id": true_id,
                "top1_pred": top1,
                "top5_pred_list": top5,
                "method": method_name,
                "condition": condition,
            }
        )

    df = pd.DataFrame(rows)
    return df


def compute_accuracies(df: pd.DataFrame) -> dict:
    """
    Compute overall top-1, top-5 accuracy and per-phase accuracy.
    """
    # Overall
    top1_acc = (df["true_phase_id"] == df["top1_pred"]).mean()
    top5_acc = df.apply(
        lambda r: int(r["true_phase_id"] in r["top5_pred_list"]), axis=1
    ).mean()

    # Per-phase
    phase_ids = sorted(df["true_phase_id"].unique())
    phase_top1 = {}
    for pid in phase_ids:
        sub = df[df["true_phase_id"] == pid]
        if len(sub) == 0:
            phase_top1[pid] = np.nan
        else:
            phase_top1[pid] = (sub["true_phase_id"] == sub["top1_pred"]).mean()

    return {
        "top1": float(top1_acc),
        "top5": float(top5_acc),
        "phase_top1": phase_top1,
    }


# --------------------------------------------------
# Plotting helpers
# --------------------------------------------------
def plot_overall_accuracy_bar(results_normal, results_shift, fig_path: Path):
    """
    results_normal / results_shift:
        dict[method_name] -> {"top1": ..., "top5": ...}
    """
    methods = list(results_normal.keys())
    x = np.arange(len(methods))
    width = 0.25

    top1_normal = [results_normal[m]["top1"] for m in methods]
    top1_shift = [results_shift.get(m, {"top1": np.nan})["top1"] for m in methods]
    top5_normal = [results_normal[m]["top5"] for m in methods]
    top5_shift = [results_shift.get(m, {"top5": np.nan})["top5"] for m in methods]

    plt.figure(figsize=(8, 5))
    b1 = plt.bar(x - width, top1_normal, width, label="Top-1 (normal)")
    b2 = plt.bar(x, top1_shift, width, label=f"Top-1 (shift={SHIFT_NAME})")
    b3 = plt.bar(x + width, top5_normal, width, label="Top-5 (normal)", alpha=0.7)
    # We plot top-5 shift as line markers on top of bars to avoid clutter.
    plt.plot(
        x + width,
        top5_shift,
        marker="o",
        linestyle="--",
        label=f"Top-5 (shift={SHIFT_NAME})",
        color="black",
    )

    for bar in b1:
        h = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            h + 0.001,
            f"{h:.3f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    for bar in b2:
        h = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            h + 0.001,
            f"{h:.3f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    plt.xticks(x, methods, rotation=15)
    plt.ylabel("Accuracy")
    plt.title("Top-1 / Top-5 accuracy under normal vs shifted data")
    plt.ylim(0.0, max(top1_normal + top1_shift + top5_normal + top5_shift) * 1.25 + 1e-4)
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_path, dpi=300)
    plt.close()
    print(f"[INFO] Saved overall reliability figure to {fig_path}")


def plot_phase_accuracy(results_normal, results_shift, fig_path: Path):
    """
    Plot per-phase top-1 accuracy drop between normal and shifted data.
    results_normal / results_shift:
        dict[method_name] -> {"phase_top1": {phase_id: acc}}
    """
    methods = list(results_normal.keys())
    # Collect all phase ids
    all_phase_ids = sorted(
        {
            pid
            for m in methods
            for pid in results_normal[m]["phase_top1"].keys()
        }
    )

    plt.figure(figsize=(10, 5))
    for m in methods:
        normal_phase = results_normal[m]["phase_top1"]
        shift_phase = results_shift.get(m, {"phase_top1": {}})["phase_top1"]

        drops = []
        for pid in all_phase_ids:
            a_norm = normal_phase.get(pid, np.nan)
            a_shift = shift_phase.get(pid, np.nan)
            if np.isnan(a_norm) or np.isnan(a_shift):
                drops.append(np.nan)
            else:
                drops.append(a_shift - a_norm)  # negative means worse under shift

        plt.plot(
            all_phase_ids,
            drops,
            marker="o",
            label=f"{m}: shift - normal",
        )

    plt.axhline(0.0, color="gray", linestyle="--", linewidth=1)
    plt.xticks(all_phase_ids, [f"P{pid}" for pid in all_phase_ids])
    plt.xlabel("Phase ID")
    plt.ylabel("Accuracy difference (shift - normal)")
    plt.title(f"Per-phase robustness under shift = {SHIFT_NAME}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_path, dpi=300)
    plt.close()
    print(f"[INFO] Saved per-phase reliability figure to {fig_path}")


# --------------------------------------------------
# Main
# --------------------------------------------------
def main():
    results_normal = {}
    results_shift = {}

    # 1) Load normal JSONs
    for method, path in METHOD_FILES_NORMAL.items():
        if not path.exists():
            print(f"[WARN] Normal JSON for {method} not found at {path}, skip.")
            continue
        print(f"[INFO] Loading NORMAL scores for {method} from {path}")
        df = load_scores_json(path, method, condition="normal")
        accs = compute_accuracies(df)
        results_normal[method] = accs

    # 2) Load shift JSONs (if available)
    for method, path in METHOD_FILES_SHIFT.items():
        if not path.exists():
            print(f"[WARN] Shift JSON for {method} not found at {path}, skip.")
            continue
        print(f"[INFO] Loading SHIFTED scores for {method} from {path}")
        df = load_scores_json(path, method, condition=f"shift_{SHIFT_NAME}")
        accs = compute_accuracies(df)
        results_shift[method] = accs

    if not results_normal:
        print("[ERROR] No normal JSON loaded. Check paths in METHOD_FILES_NORMAL.")
        return

    # 3) Save a small CSV summary for the paper
    summary_rows = []
    for method in results_normal.keys():
        normal = results_normal[method]
        shift = results_shift.get(method, {"top1": np.nan, "top5": np.nan})
        summary_rows.append(
            {
                "method": method,
                "condition": "normal",
                "top1": normal["top1"],
                "top5": normal["top5"],
            }
        )
        summary_rows.append(
            {
                "method": method,
                "condition": f"shift_{SHIFT_NAME}",
                "top1": shift["top1"],
                "top5": shift["top5"],
            }
        )
    summary_df = pd.DataFrame(summary_rows)
    csv_path = RESULT_DIR / "reliability_summary.csv"
    summary_df.to_csv(csv_path, index=False)
    print(f"[INFO] Saved numeric reliability summary to {csv_path}")

    # 4) Plots
    overall_fig = FIG_DIR / "top1_top5_accuracy_bar.png"
    plot_overall_accuracy_bar(results_normal, results_shift, overall_fig)

    phase_fig = FIG_DIR / "phase_accuracy_diff.png"
    plot_phase_accuracy(results_normal, results_shift, phase_fig)

    print("[DONE] Reliability analysis finished.")


if __name__ == "__main__":
    main()
