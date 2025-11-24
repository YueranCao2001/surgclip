# scripts/generate_shift_scores.py
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageEnhance, ImageFilter, ImageDraw
from tqdm import tqdm

import torch
import open_clip


# --------------------------------------------------
# Basic paths
# --------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]        # surgclip root: .../surgclip
INDEX_CSV = ROOT / "index" / "keyframe_index.csv"

RESULT_DIR = ROOT / "results"
RESULT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Paths for adapters (same as other scripts)
ADAPTER_FP32_PATH = RESULT_DIR / "clip_adapter_best.pth"

# --------------------------------------------------
# Phase texts (same as PQ / retrieval)
# --------------------------------------------------
PHASE_TEXT = {
    1: "The surgical tools are being inserted and the field is prepared for the procedure.",
    2: "The surgeon is dissecting tissue in Calot's triangle to expose the cystic duct and artery.",
    3: "The cystic duct and artery are being clipped and cut to separate them safely.",
    4: "The gallbladder is being dissected away from the liver bed.",
    5: "The gallbladder is being packed and prepared for removal.",
    6: "Residual bleeding or bile leakage is being controlled by cleaning and coagulation.",
    7: "The gallbladder is being extracted from the abdominal cavity.",
}


# --------------------------------------------------
# Image shift / corruption
# --------------------------------------------------
def apply_shift(img: Image.Image, shift_type: str) -> Image.Image:
    """
    Apply a synthetic distribution shift / corruption to the input image.

    You can add more cases here (noise, occlusion, color shift, etc.).
    """
    if shift_type is None or shift_type == "none":
        return img

    if shift_type == "brightness_blur":
        # Darker + slightly blurred
        img = ImageEnhance.Brightness(img).enhance(0.4)
        img = ImageEnhance.Contrast(img).enhance(0.8)
        img = img.filter(ImageFilter.GaussianBlur(radius=1.5))
        return img

    if shift_type == "instrument_occlusion":
        # Simulate occlusion by tools: draw a gray rectangle
        img = img.copy()
        draw = ImageDraw.Draw(img)
        w, h = img.size
        box_w, box_h = int(0.45 * w), int(0.25 * h)
        x0 = int(0.3 * w)
        y0 = int(0.3 * h)
        draw.rectangle([x0, y0, x0 + box_w, y0 + box_h], fill=(90, 90, 90))
        return img

    if shift_type == "gaussian_noise":
        # Simple additive Gaussian noise in numpy space
        arr = np.array(img).astype(np.float32) / 255.0
        noise = np.random.normal(0.0, 0.10, size=arr.shape).astype(np.float32)
        arr = np.clip(arr + noise, 0.0, 1.0)
        arr = (arr * 255.0).astype(np.uint8)
        return Image.fromarray(arr)

    # Default: no change if unknown name
    return img


# --------------------------------------------------
# CLIP backbone + adapters
# --------------------------------------------------
def build_adapter_from_state(state_dict: dict, device: torch.device):
    """
    Rebuild the 2-layer MLP adapter from a state_dict.

    Expected keys: '0.weight', '0.bias', '2.weight', '2.bias'
    corresponding to:
        Linear(in_dim -> hidden_dim), ReLU, Linear(hidden_dim -> out_dim)
    """
    if "state_dict" in state_dict and isinstance(state_dict["state_dict"], dict):
        state_dict = state_dict["state_dict"]

    w0 = state_dict["0.weight"]  # [hidden_dim, in_dim]
    w2 = state_dict["2.weight"]  # [out_dim, hidden_dim]

    in_dim = w0.shape[1]
    hidden_dim = w0.shape[0]
    out_dim = w2.shape[0]

    adapter = torch.nn.Sequential(
        torch.nn.Linear(in_dim, hidden_dim),
        torch.nn.ReLU(),
        torch.nn.Linear(hidden_dim, out_dim),
    )
    adapter.load_state_dict(state_dict)
    adapter.to(device)
    adapter.eval()
    return adapter


def load_model_and_text(method: str):
    """
    Load CLIP backbone + (optionally) adapters, and text embeddings.
    method in {"baseline", "adapted", "adapted_int8"}.
    """
    model_name = "ViT-B-32"
    pretrained = "openai"
    clip_model, _, preprocess = open_clip.create_model_and_transforms(
        model_name, pretrained=pretrained, device=DEVICE
    )
    tokenizer = open_clip.get_tokenizer(model_name)
    clip_model.eval()

    # Encode phase texts
    phase_ids = sorted(PHASE_TEXT.keys())
    texts = [PHASE_TEXT[i] for i in phase_ids]
    with torch.no_grad():
        text_tokens = tokenizer(texts).to(DEVICE)
        text_feats = clip_model.encode_text(text_tokens)
        text_feats = text_feats / text_feats.norm(dim=-1, keepdim=True)

    adapter = None
    adapter_device = DEVICE

    if method == "adapted":
        # FP32 adapter on DEVICE
        state = torch.load(ADAPTER_FP32_PATH, map_location=DEVICE)
        adapter = build_adapter_from_state(state, DEVICE)
        adapter_device = DEVICE

    elif method == "adapted_int8":
        # INT8 adapter via dynamic quantization (CPU-only)
        state = torch.load(ADAPTER_FP32_PATH, map_location="cpu")
        adapter_fp32 = build_adapter_from_state(state, torch.device("cpu"))
        adapter_int8 = torch.quantization.quantize_dynamic(
            adapter_fp32, {torch.nn.Linear}, dtype=torch.qint8
        )
        adapter_int8.eval()
        adapter = adapter_int8
        adapter_device = torch.device("cpu")

        # For INT8 we will move features + text to CPU
        text_feats = text_feats.cpu()

    return clip_model, preprocess, text_feats, phase_ids, adapter, adapter_device


# --------------------------------------------------
# Main retrieval loop
# --------------------------------------------------
def run_retrieval(method: str, shift_type: str, out_path: Path):
    """
    Compute top-k phase predictions for all frames under a given shift.
    Save to JSON compatible with existing retrieval_scores*.json.
    """
    print(f"[INFO] Method = {method}, shift = {shift_type}")
    print(f"[INFO] Loading index from {INDEX_CSV}")

    df = pd.read_csv(INDEX_CSV)
    # We only care about frames with valid phase_id in PHASE_TEXT
    df = df[df["phase_id"].isin(PHASE_TEXT.keys())].reset_index(drop=True)
    num_frames = len(df)
    print(f"[INFO] Number of frames: {num_frames}")

    clip_model, preprocess, text_feats, phase_ids, adapter, adapter_device = load_model_and_text(method)

    results = []
    correct_top1 = 0
    correct_top5 = 0

    start_time = time.time()

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Running retrieval"):
        frame_rel = row["frame_path"]
        true_phase = int(row["phase_id"])

        img_path = ROOT / frame_rel
        img = Image.open(img_path).convert("RGB")
        img_shifted = apply_shift(img, shift_type)

        image_tensor = preprocess(img_shifted).unsqueeze(0)

        with torch.no_grad():
            image_tensor = image_tensor.to(DEVICE)
            img_feat = clip_model.encode_image(image_tensor)

            if adapter is not None:
                # Move to adapter device if needed (INT8 on CPU)
                img_feat = img_feat.to(adapter_device)
                img_feat = adapter(img_feat)

            img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)

            # Move both to same device for similarity computation
            if adapter is not None and adapter_device.type == "cpu":
                feat = img_feat.cpu()
                text = text_feats.cpu()
            else:
                feat = img_feat
                text = text_feats

            # Similarity with all 7 phases => ranking
            sims = (feat @ text.T).squeeze(0)  # (7,)
            order = torch.argsort(sims, descending=True).cpu().numpy()
            pred_phase_ids = [int(phase_ids[i]) for i in order[:7]]

        # Top-1 / Top-5 accuracy
        if true_phase == pred_phase_ids[0]:
            correct_top1 += 1
        if true_phase in pred_phase_ids[:5]:
            correct_top5 += 1

        pred_phase_texts = [PHASE_TEXT[pid] for pid in pred_phase_ids]

        results.append(
            {
                "frame_path": frame_rel,
                "true_phase_id": true_phase,
                "pred_phase_ids": pred_phase_ids,
                "pred_phase_texts": pred_phase_texts,
            }
        )

    elapsed = time.time() - start_time
    top1_acc = correct_top1 / num_frames
    top5_acc = correct_top5 / num_frames

    print(f"[RESULT] top-1 accuracy: {top1_acc:.6f}")
    print(f"[RESULT] top-5 accuracy: {top5_acc:.6f}")
    print(f"[INFO] Elapsed time: {elapsed/60:.2f} min")

    out_obj = {
        "num_frames": num_frames,
        "top1_accuracy": top1_acc,
        "top5_accuracy": top5_acc,
        "method": method,
        "shift_type": shift_type,
        "examples": results,
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out_obj, f, indent=2)
    print(f"[INFO] Saved shifted retrieval scores to {out_path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate retrieval_scores JSON under synthetic distribution shift."
    )
    parser.add_argument(
        "--method",
        type=str,
        default="baseline",
        choices=["baseline", "adapted", "adapted_int8"],
        help="Which model variant to use.",
    )
    parser.add_argument(
        "--shift",
        type=str,
        default="brightness_blur",
        help="Shift type name (see apply_shift).",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.method == "baseline":
        suffix = f"shift_{args.shift}"
        out_name = f"retrieval_scores_{suffix}.json"
    elif args.method == "adapted":
        suffix = f"adapted_shift_{args.shift}"
        out_name = f"retrieval_scores_{suffix}.json"
    else:  # adapted_int8
        suffix = f"adapted_int8_shift_{args.shift}"
        out_name = f"retrieval_scores_{suffix}.json"

    out_path = RESULT_DIR / out_name
    run_retrieval(args.method, args.shift, out_path)


if __name__ == "__main__":
    main()
