import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


ROOT = Path(r"D:\GU\paper\surgclip")
EXAMPLES_JSON = ROOT / "results" / "retrieval_examples_adapted.json"
FIG_DIR = ROOT / "results" / "figures"

MAX_PER_FIG = 12  # you asked for 12 examples per figure

PHASE_SHORT = {
    1: "Preparation",
    2: "Calot Triangle Dissection",
    3: "Clipping & Cutting",
    4: "Gallbladder Dissection",
    5: "Gallbladder Packaging",
    6: "Cleaning & Coagulation",
    7: "Gallbladder Retraction",
}


def load_examples():
    if not EXAMPLES_JSON.exists():
        raise FileNotFoundError(EXAMPLES_JSON)

    with open(EXAMPLES_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)

    top1_correct = data.get("top1_correct", [])
    top5_correct_only = data.get("top5_correct_only", [])
    failures = data.get("failures", [])

    print(f"[INFO] Loaded examples: "
          f"top1_correct={len(top1_correct)}, "
          f"top5_correct_only={len(top5_correct_only)}, "
          f"failures={len(failures)}")

    return top1_correct, top5_correct_only, failures


def sample_diverse(ex_list, max_total=12, max_per_phase=3):
    """
    Sample up to max_total examples, with at most max_per_phase per GT phase.
    """
    selected = []
    per_phase = defaultdict(int)

    for ex in ex_list:
        if len(selected) >= max_total:
            break
        ph = int(ex["true_phase_id"])
        if per_phase[ph] >= max_per_phase:
            continue
        selected.append(ex)
        per_phase[ph] += 1

    return selected


def draw_grid(examples, title, outfile, color_mode="green", cols=4):
    n = len(examples)
    if n == 0:
        print(f"[WARN] No examples for {title}")
        return

    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(
        rows,
        cols,
        figsize=(4 * cols, 4 * rows),
        constrained_layout=True,
    )

    if rows == 1:
        axes = [axes] if isinstance(axes, plt.Axes) else axes
    else:
        axes = axes.flatten()

    if color_mode == "green":
        title_color = "green"
    elif color_mode == "orange":
        title_color = "orange"
    else:
        title_color = "red"

    for i, ax in enumerate(axes):
        if i >= n:
            ax.axis("off")
            continue

        ex = examples[i]
        img_path = ROOT / ex["frame_path"]
        true_id = int(ex["true_phase_id"])
        pred_top1 = int(ex["pred_phase_ids"][0])

        if img_path.exists():
            img = Image.open(img_path).convert("RGB")
            ax.imshow(np.asarray(img))
        else:
            ax.text(0.5, 0.5, "Missing frame", ha="center", va="center")
        ax.axis("off")

        gt_name = PHASE_SHORT.get(true_id, f"Phase {true_id}")
        pred_name = PHASE_SHORT.get(pred_top1, f"Phase {pred_top1}")

        ax.set_title(
            f"GT: {gt_name}\nPred@1: {pred_name}",
            color=title_color,
            fontsize=9,
        )

    fig.suptitle(title, fontsize=16)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(outfile, dpi=220)
    plt.close(fig)
    print(f"[INFO] Saved figure: {outfile}")


def main():
    top1_correct, top5_correct_only, failures = load_examples()

    top1_sel = sample_diverse(top1_correct, max_total=MAX_PER_FIG, max_per_phase=3)
    top5_sel = sample_diverse(top5_correct_only, max_total=MAX_PER_FIG, max_per_phase=3)
    fail_sel = sample_diverse(failures, max_total=MAX_PER_FIG, max_per_phase=3)

    draw_grid(
        top1_sel,
        title="Adapted CLIP – Top-1 Correct Cases",
        outfile=FIG_DIR / "adapted_top1_correct.png",
        color_mode="green",
    )

    draw_grid(
        top5_sel,
        title="Adapted CLIP – Top-5 Correct but Top-1 Wrong",
        outfile=FIG_DIR / "adapted_top5_correct_only.png",
        color_mode="orange",
    )

    draw_grid(
        fail_sel,
        title="Adapted CLIP – Failure Cases (Not in Top-5)",
        outfile=FIG_DIR / "adapted_failures.png",
        color_mode="red",
    )


if __name__ == "__main__":
    main()
