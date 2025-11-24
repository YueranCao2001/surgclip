import json
import pandas as pd
import argparse

def evaluate(path):
    with open(path,"r") as f:
        data = json.load(f)

    correct_top1 = sum([1 if x["correct_top1"] else 0 for x in data])
    correct_top5 = sum([1 if x["correct_top5"] else 0 for x in data])
    total = len(data)

    return correct_top1/total, correct_top5/total, total

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", required=True)
    args = parser.parse_args()

    top1, top5, total = evaluate(args.json)
    print(f"[RESULT] Top-1: {top1:.4f}, Top-5: {top5:.4f}, N={total}")
