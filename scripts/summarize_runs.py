import argparse
import json
import os
from typing import Dict, List

import pandas as pd


def load_metrics(path: str) -> Dict:
    """Load a JSON metrics file and extract top-1 / top-5 / num_frames."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 尝试几种可能的 key，防止命名略有不同
    def pick(d: Dict, candidates):
        for k in candidates:
            if k in d:
                return d[k]
        # 如果都没有，就抛个错误，让你看一下 json 里有哪些 key
        raise KeyError(f"None of {candidates} found in {path}. Available keys: {list(d.keys())}")

    top1 = pick(data, ["top1_acc", "top1_accuracy", "top1"])
    top5 = pick(data, ["top5_acc", "top5_accuracy", "top5"])
    num_frames = pick(data, ["num_frames", "n_frames", "total_frames"])

    return {
        "top1_acc": top1,
        "top5_acc": top5,
        "num_frames": num_frames,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Summarize baseline and adapted CLIP runs into one table."
    )
    parser.add_argument(
        "--baseline",
        type=str,
        default="results/retrieval_scores.json",
        help="JSON file with baseline metrics.",
    )
    parser.add_argument(
        "--adapted",
        type=str,
        default="results/retrieval_scores_adapted.json",
        help="JSON file with adapted CLIP metrics.",
    )
    parser.add_argument(
        "--adapter_partial",
        type=str,
        default=None,
        help="(Optional) JSON file with partially trained adapter metrics.",
    )
    parser.add_argument(
        "--out_csv",
        type=str,
        default="results/overall_accuracy_summary.csv",
        help="Path to write the summary CSV.",
    )
    args = parser.parse_args()

    rows: List[Dict] = []

    # Baseline
    if args.baseline and os.path.isfile(args.baseline):
        m = load_metrics(args.baseline)
        rows.append({
            "run_name": "baseline",
            "top1_acc": m["top1_acc"],
            "top5_acc": m["top5_acc"],
            "num_frames": m["num_frames"],
        })
    else:
        print("[WARN] Baseline metrics file not found, skipping.")

    # Partial adapter (如果你以后有部分训练的结果，可以传进来；现在可以不填)
    if args.adapter_partial:
        if os.path.isfile(args.adapter_partial):
            m = load_metrics(args.adapter_partial)
            rows.append({
                "run_name": "adapter_partial",
                "top1_acc": m["top1_acc"],
                "top5_acc": m["top5_acc"],
                "num_frames": m["num_frames"],
            })
        else:
            print("[WARN] adapter_partial metrics file not found, skipping.")

    # Full adapted
    if args.adapted and os.path.isfile(args.adapted):
        m = load_metrics(args.adapted)
        rows.append({
            "run_name": "adapter_full",
            "top1_acc": m["top1_acc"],
            "top5_acc": m["top5_acc"],
            "num_frames": m["num_frames"],
        })
    else:
        print("[WARN] Adapted metrics file not found, skipping.")

    if not rows:
        raise RuntimeError("No valid metrics files were loaded.")

    df = pd.DataFrame(rows)

    # 以 baseline 为参照，算增益
    base_row = df[df["run_name"] == "baseline"].iloc[0]
    df["delta_top1_vs_baseline"] = df["top1_acc"] - base_row["top1_acc"]
    df["delta_top5_vs_baseline"] = df["top5_acc"] - base_row["top5_acc"]

    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    df.to_csv(args.out_csv, index=False)

    print("[INFO] Summary table:")
    print(df.to_string(index=False))
    print(f"[INFO] Written summary CSV to: {args.out_csv}")


if __name__ == "__main__":
    main()
