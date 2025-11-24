import argparse
import os
from typing import List

import matplotlib.pyplot as plt
import pandas as pd


def setup_matplotlib():
    """Basic matplotlib style."""
    plt.rcParams.update({
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "legend.fontsize": 9,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "figure.dpi": 200,
    })


def read_phase_csv(csv_path: str) -> pd.DataFrame:
    """
    Load the phase-wise accuracy CSV.

    Expected columns:
      - phase_name
      - baseline_top1
      - baseline_top5
      - adapted_top1
      - adapted_top5
    """
    df = pd.read_csv(csv_path)

    required_cols = [
        "phase_name",
        "baseline_top1",
        "baseline_top5",
        "adapted_top1",
        "adapted_top5",
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in CSV: {missing}")

    return df


def plot_grouped_bar(df: pd.DataFrame, out_path: str):
    """Grouped bar: phase vs accuracy (baseline/adapted, top1/top5)."""
    phases: List[str] = df["phase_name"].tolist()
    x = range(len(phases))

    width = 0.18
    fig, ax = plt.subplots(figsize=(10, 4))

    ax.bar(
        [i - 1.5 * width for i in x],
        df["baseline_top1"],
        width,
        label="Baseline Top-1",
    )
    ax.bar(
        [i - 0.5 * width for i in x],
        df["adapted_top1"],
        width,
        label="Adapted Top-1",
    )
    ax.bar(
        [i + 0.5 * width for i in x],
        df["baseline_top5"],
        width,
        label="Baseline Top-5",
    )
    ax.bar(
        [i + 1.5 * width for i in x],
        df["adapted_top5"],
        width,
        label="Adapted Top-5",
    )

    ax.set_xticks(list(x))
    ax.set_xticklabels(phases, rotation=30, ha="right")
    ax.set_ylabel("Accuracy")
    ax.set_title("Phase-wise Accuracy (Baseline vs Adapted)")
    ax.set_ylim(0.0, 1.0)
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_delta_bar(df: pd.DataFrame, out_path: str):
    """Per-phase improvement: ΔTop-1 / ΔTop-5."""
    phases = df["phase_name"].tolist()
    x = range(len(phases))

    delta_top1 = df["adapted_top1"] - df["baseline_top1"]
    delta_top5 = df["adapted_top5"] - df["baseline_top5"]

    width = 0.3
    fig, ax = plt.subplots(figsize=(10, 4))

    ax.bar(
        [i - width / 2 for i in x],
        delta_top1,
        width,
        label="Δ Top-1 (Adapted - Baseline)",
    )
    ax.bar(
        [i + width / 2 for i in x],
        delta_top5,
        width,
        label="Δ Top-5 (Adapted - Baseline)",
    )

    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_xticks(list(x))
    ax.set_xticklabels(phases, rotation=30, ha="right")
    ax.set_ylabel("Accuracy Difference")
    ax.set_title("Per-phase Accuracy Gain (Adapted vs Baseline)")
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_scatter_baseline_vs_adapt(df: pd.DataFrame, out_path: str):
    """
    Scatter: x=baseline, y=adapted, one point per phase.
    Points above diagonal => improved.
    """
    fig, ax = plt.subplots(figsize=(4.5, 4.5))

    # Top-1
    ax.scatter(
        df["baseline_top1"],
        df["adapted_top1"],
        marker="o",
        label="Top-1",
    )
    # Top-5
    ax.scatter(
        df["baseline_top5"],
        df["adapted_top5"],
        marker="^",
        label="Top-5",
    )

    min_val, max_val = 0.0, 1.0
    ax.plot([min_val, max_val], [min_val, max_val], "k--", linewidth=0.8)

    for _, row in df.iterrows():
        phase = str(row["phase_name"])
        x1, y1 = row["baseline_top1"], row["adapted_top1"]
        ax.text(x1, y1, phase, fontsize=7, alpha=0.7)

    ax.set_xlabel("Baseline Accuracy")
    ax.set_ylabel("Adapted Accuracy")
    ax.set_title("Baseline vs Adapted Accuracy (per-phase)")
    ax.set_xlim(min_val, max_val)
    ax.set_ylim(min_val, max_val)
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Visualize phase-wise accuracy from CSV."
    )
    parser.add_argument(
        "--csv",
        type=str,
        default="results/phase_wise_accuracy.csv",
        help="Path to phase-wise accuracy CSV file.",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="results/figures_phase",
        help="Output directory for generated figures.",
    )
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    setup_matplotlib()

    df = read_phase_csv(args.csv)

    grouped_path = os.path.join(args.out_dir, "phase_accuracy_grouped_bar.png")
    plot_grouped_bar(df, grouped_path)

    delta_path = os.path.join(args.out_dir, "phase_accuracy_delta_bar.png")
    plot_delta_bar(df, delta_path)

    scatter_path = os.path.join(args.out_dir, "phase_accuracy_baseline_vs_adapt.png")
    plot_scatter_baseline_vs_adapt(df, scatter_path)

    print(f"[INFO] Saved figures to: {args.out_dir}")


if __name__ == "__main__":
    main()
