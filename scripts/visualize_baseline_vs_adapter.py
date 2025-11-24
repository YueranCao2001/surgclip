import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


ROOT = Path(r"D:\GU\paper\surgclip")
EX_JSON = ROOT / "results" / "baseline_vs_adapter_examples.json"
FIG_DIR = ROOT / "results" / "figures"

MAX_PER_FIG = 12  # 每类 12 个示例

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
    with open(EX_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)
    return (
        data.get("improved", []),
        data.get("unchanged_good", []),
        data.get("unchanged_bad", []),
        data.get("degraded", []),
    )


def sample_diverse(ex_list, max_total=12, max_per_phase=3):
    selected = []
    per_phase = defaultdict(int)
    for ex in ex_list:
        if len(selected) >= max_total:
            break
        ph = int(ex["gt_id"])
        if per_phase[ph] >= max_per_phase:
            continue
        selected.append(ex)
        per_phase[ph] += 1
    return selected


def draw_pairs(examples, title, outfile, title_color="green"):
    n = len(examples)
    if n == 0:
        print(f"[WARN] No examples for {title}")
        return

    rows = n
    cols = 2  # 左 baseline, 右 adapted

    fig, axes = plt.subplots(
        rows,
        cols,
        figsize=(6 * cols, 3 * rows),
        constrained_layout=True,
    )

    for row_idx, ex in enumerate(examples):
        img_path = ROOT / ex["frame_path"]
        gt = int(ex["gt_id"])
        pb = int(ex["baseline_pred_id"])
        pa = int(ex["adapted_pred_id"])

        if img_path.exists():
            img = Image.open(img_path).convert("RGB")
            img_arr = np.asarray(img)
        else:
            img_arr = None

        # Baseline (left)
        ax_l = axes[row_idx, 0] if rows > 1 else axes[0]
        if img_arr is not None:
            ax_l.imshow(img_arr)
        ax_l.axis("off")
        gt_name = PHASE_SHORT.get(gt, f"P{gt}")
        pb_name = PHASE_SHORT.get(pb, f"P{pb}")
        ax_l.set_title(
            f"Baseline\nGT: {gt_name}\nPred: {pb_name}",
            color="red" if pb != gt else "green",
            fontsize=9,
        )

        # Adapted (right)
        ax_r = axes[row_idx, 1] if rows > 1 else axes[1]
        if img_arr is not None:
            ax_r.imshow(img_arr)
        ax_r.axis("off")
        pa_name = PHASE_SHORT.get(pa, f"P{pa}")
        ax_r.set_title(
            f"Adapted\nGT: {gt_name}\nPred: {pa_name}",
            color="red" if pa != gt else "green",
            fontsize=9,
        )

    fig.suptitle(title, fontsize=16)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(outfile, dpi=220)
    plt.close(fig)
    print(f"[INFO] Saved figure: {outfile}")


def main():
    improved, unchanged_good, unchanged_bad, degraded = load_examples()

    imp_sel = sample_diverse(improved, MAX_PER_FIG, max_per_phase=3)
    deg_sel = sample_diverse(degraded, MAX_PER_FIG, max_per_phase=3)

    draw_pairs(
        imp_sel,
        title="Baseline vs Adapted – Improved Cases\n(Baseline wrong, Adapted correct)",
        outfile=FIG_DIR / "baseline_vs_adapted_improved.png",
    )

    draw_pairs(
        deg_sel,
        title="Baseline vs Adapted – Degraded Cases\n(Baseline correct, Adapted wrong)",
        outfile=FIG_DIR / "baseline_vs_adapted_degraded.png",
    )


if __name__ == "__main__":
    main()
