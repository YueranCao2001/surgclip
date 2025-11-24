import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Root project directory
ROOT = Path(r"D:\GU\paper\surgclip")

SAMPLES_JSON = ROOT / "results" / "retrieval_examples.json"
FIG_DIR = ROOT / "results" / "figures"

PHASE_SHORT = {
    1: "Preparation",
    2: "Calot Triangle Dissection",
    3: "Clipping & Cutting",
    4: "Gallbladder Dissection",
    5: "Gallbladder Packaging",
    6: "Cleaning & Coagulation",
    7: "Gallbladder Retraction",
}


def load_samples():
    """
    Load sampled examples (top-1 correct and top-1 wrong)
    from retrieval_examples.json.
    """
    if not SAMPLES_JSON.exists():
        raise FileNotFoundError(SAMPLES_JSON)

    with open(SAMPLES_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)

    ex_correct = data.get("examples_correct", [])
    ex_wrong = data.get("examples_wrong", [])

    print(f"[INFO] Loaded {len(ex_correct)} top-1 correct examples "
          f"and {len(ex_wrong)} top-1 wrong examples.")
    return ex_correct, ex_wrong


def categorize_examples(ex_correct, ex_wrong):
    """
    Build three sets:
    1) top1_correct     : GT == Pred@1
    2) top5_correct_only: GT in top-5 but GT != Pred@1
    3) failures         : GT not in top-5
    """
    top1_correct = list(ex_correct)
    top5_correct_only = []
    failures = []

    for ex in ex_wrong:
        true_id = int(ex["true_phase_id"])
        pred_ids = [int(pid) for pid in ex["pred_phase_ids"]]

        if true_id in pred_ids:
            # GT appears somewhere in top-5, but we know top-1 is wrong
            top5_correct_only.append(ex)
        else:
            failures.append(ex)

    print(f"[INFO] top1_correct     : {len(top1_correct)}")
    print(f"[INFO] top5_correct_only: {len(top5_correct_only)}")
    print(f"[INFO] failures         : {len(failures)}")
    return top1_correct, top5_correct_only, failures


def sample_diverse(ex_list, max_total=12, max_per_phase=3):
    """
    Sample up to max_total examples, with at most max_per_phase
    examples per ground-truth phase to increase diversity.
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
    """
    Draw a grid of examples.
    color_mode:
        "green" : title in green  (top-1 correct)
        "orange": title in orange (top-5 correct but top-1 wrong)
        "red"   : title in red    (failures)
    """
    n = len(examples)
    if n == 0:
        print(f"[WARN] No examples to draw for {title}")
        return

    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(
        rows, cols,
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

        # Load image
        if img_path.exists():
            img = Image.open(img_path).convert("RGB")
            ax.imshow(np.asarray(img))
        else:
            ax.text(0.5, 0.5, "Missing frame", ha="center", va="center")
        ax.axis("off")

        gt_name = PHASE_SHORT.get(true_id, f"Phase {true_id}")
        pred_name = PHASE_SHORT.get(pred_top1, f"Phase {pred_top1}")

        # Compose short title
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
    ex_correct, ex_wrong = load_samples()
    top1_correct, top5_correct_only, failures = categorize_examples(
        ex_correct, ex_wrong
    )

    # Sample subsets with diversity
    top1_sel = sample_diverse(top1_correct, max_total=12, max_per_phase=3)
    top5_sel = sample_diverse(top5_correct_only, max_total=12, max_per_phase=3)
    fail_sel = sample_diverse(failures, max_total=12, max_per_phase=3)

    # 1) Top-1 correct cases
    draw_grid(
        top1_sel,
        title="CLIP Retrieval – Top-1 Correct Cases",
        outfile=FIG_DIR / "triplet_top1_correct.png",
        color_mode="green",
    )

    # 2) Top-5 correct but Top-1 wrong
    draw_grid(
        top5_sel,
        title="CLIP Retrieval – Top-5 Correct but Top-1 Wrong",
        outfile=FIG_DIR / "triplet_top5_correct_only.png",
        color_mode="orange",
    )

    # 3) Top-5 failure cases
    draw_grid(
        fail_sel,
        title="CLIP Retrieval – Failure Cases (Not in Top-5)",
        outfile=FIG_DIR / "triplet_failures.png",
        color_mode="red",
    )


if __name__ == "__main__":
    main()
