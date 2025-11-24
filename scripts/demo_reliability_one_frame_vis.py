# scripts/demo_reliability_one_frame_vis.py
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageEnhance, ImageFilter
import matplotlib.pyplot as plt
import torch
import open_clip


# --------------------------------------------------
# Basic paths and config
# --------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]   # project root: .../surgclip
INDEX_CSV = ROOT / "index" / "keyframe_index.csv"
RESULT_DIR = ROOT / "results"
RESULT_DIR.mkdir(parents=True, exist_ok=True)

ADAPTER_FP32_PATH = RESULT_DIR / "clip_adapter_best.pth"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
# Synthetic shift: brightness + blur
# --------------------------------------------------
def apply_brightness_blur(img: Image.Image) -> Image.Image:
    """
    Apply the same synthetic distribution shift used in reliability experiments:
    darker scene + slightly lower contrast + mild Gaussian blur.
    """
    img = ImageEnhance.Brightness(img).enhance(0.4)
    img = ImageEnhance.Contrast(img).enhance(0.8)
    img = img.filter(ImageFilter.GaussianBlur(radius=1.5))
    return img


# --------------------------------------------------
# Adapter helpers
# --------------------------------------------------
def build_adapter_from_state(state_dict: dict, device: torch.device) -> torch.nn.Module:
    """
    Rebuild the 2-layer MLP adapter from the saved state_dict.

    Expected keys:
        "0.weight", "0.bias", "2.weight", "2.bias"
    corresponding to:
        Linear(in_dim -> hidden_dim), ReLU, Linear(hidden_dim -> out_dim)
    """
    if "state_dict" in state_dict and isinstance(state_dict["state_dict"], dict):
        state_dict = state_dict["state_dict"]

    w0 = state_dict["0.weight"]
    w2 = state_dict["2.weight"]

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


def load_clip_and_text():
    """
    Load CLIP ViT-B/32 backbone and encode the 7 phase text prompts.
    """
    model_name = "ViT-B-32"
    pretrained = "openai"

    clip_model, _, preprocess = open_clip.create_model_and_transforms(
        model_name, pretrained=pretrained, device=DEVICE
    )
    tokenizer = open_clip.get_tokenizer(model_name)
    clip_model.eval()

    phase_ids = sorted(PHASE_TEXT.keys())
    texts = [PHASE_TEXT[i] for i in phase_ids]
    with torch.no_grad():
        tokens = tokenizer(texts).to(DEVICE)
        text_feats = clip_model.encode_text(tokens)
        text_feats = text_feats / text_feats.norm(dim=-1, keepdim=True)

    return clip_model, preprocess, text_feats, phase_ids


# --------------------------------------------------
# Core prediction for one image
# --------------------------------------------------
def predict_one_image(
    clip_model,
    preprocess,
    text_feats: torch.Tensor,
    phase_ids,
    img: Image.Image,
    adapter: torch.nn.Module | None,
    adapter_device: torch.device,
    method_name: str,
    condition: str,
):
    """
    Run text-to-image phase retrieval for a single image and return top-5 predictions.

    Returns:
        {
          "method": method_name,
          "condition": condition,
          "top5_phase_ids": [int, ...],
          "top5_scores": [float, ...]
        }
    """
    img_tensor = preprocess(img).unsqueeze(0)

    with torch.no_grad():
        # Always encode images on DEVICE
        img_tensor = img_tensor.to(DEVICE)
        feats = clip_model.encode_image(img_tensor)

        # Optional adapter (FP32 on DEVICE or INT8 on CPU)
        if adapter is not None:
            feats = feats.to(adapter_device)
            feats = adapter(feats)

        feats = feats / feats.norm(dim=-1, keepdim=True)

        # Select device for similarity computation
        if adapter is not None and adapter_device.type == "cpu":
            feats_sim = feats.cpu()
            text_sim = text_feats.cpu()
        else:
            feats_sim = feats
            text_sim = text_feats

        sims = (feats_sim @ text_sim.T).squeeze(0)  # (7,)
        order = torch.argsort(sims, descending=True).cpu().numpy()

    top5_ids = [int(phase_ids[i]) for i in order[:5]]
    top5_scores = [float(sims[i].item()) for i in order[:5]]

    # Pretty print to terminal
    print(f"\n=== {method_name} | condition = {condition} ===")
    for rank, (pid, score) in enumerate(zip(top5_ids, top5_scores), start=1):
        print(f"Top-{rank}: Phase {pid} | score={score:.4f}")
        print(f"        {PHASE_TEXT[pid]}")

    return {
        "method": method_name,
        "condition": condition,
        "top5_phase_ids": top5_ids,
        "top5_scores": top5_scores,
    }


# --------------------------------------------------
# Visualization
# --------------------------------------------------
def visualize_one_frame(
    img_normal: Image.Image,
    img_shift: Image.Image,
    predictions: dict,
    true_phase: int,
    fig_path: Path,
):
    """
    Create a 2-column figure (normal vs shifted) and one row per method.
    Each subplot shows the image and the Top-1 predicted phase.

    predictions:
        {
          "baseline_fp32": {
              "normal": {...},
              "brightness_blur": {...}
          },
          "adapter_fp32": {...},
          "adapter_int8": {...}
        }
    """
    methods = list(predictions.keys())
    n_methods = len(methods)
    if n_methods == 0:
        print("[WARN] No methods in predictions dict, skip visualization.")
        return

    fig, axes = plt.subplots(
        n_methods, 2, figsize=(8, 3 * n_methods)
    )

    # Ensure axes is 2D array even if n_methods == 1
    if n_methods == 1:
        axes = np.expand_dims(axes, axis=0)

    img_normal_np = np.array(img_normal)
    img_shift_np = np.array(img_shift)

    for row_idx, method in enumerate(methods):
        # Normal
        ax0 = axes[row_idx, 0]
        ax0.imshow(img_normal_np)
        ax0.axis("off")
        pred_norm = predictions[method].get("normal", None)
        if pred_norm is not None and pred_norm["top5_phase_ids"]:
            top1_pid = pred_norm["top5_phase_ids"][0]
            ax0.set_title(
                f"{method}\n"
                f"normal: Top-1 = P{top1_pid}",
                fontsize=10,
            )
        else:
            ax0.set_title(f"{method}\nnormal: no prediction", fontsize=10)

        # Shifted
        ax1 = axes[row_idx, 1]
        ax1.imshow(img_shift_np)
        ax1.axis("off")
        pred_shift = predictions[method].get("brightness_blur", None)
        if pred_shift is not None and pred_shift["top5_phase_ids"]:
            top1_pid_s = pred_shift["top5_phase_ids"][0]
            ax1.set_title(
                f"{method}\n"
                f"brightness_blur: Top-1 = P{top1_pid_s}",
                fontsize=10,
            )
        else:
            ax1.set_title(
                f"{method}\nbrightness_blur: no prediction", fontsize=10
            )

    plt.suptitle(
        f"Demo reliability on one keyframe (true phase = P{true_phase})",
        fontsize=12,
    )
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(fig_path, dpi=300)
    plt.close()
    print(f"[INFO] Saved demo visualization to {fig_path}")


# --------------------------------------------------
# Main
# --------------------------------------------------
def main():
    # 1) Pick one keyframe from keyframe_index.csv
    df = pd.read_csv(INDEX_CSV)
    df = df[df["phase_id"].isin(PHASE_TEXT.keys())].reset_index(drop=True)
    if len(df) == 0:
        print("[ERROR] No valid frames in index for phases 1–7.")
        return

    row = df.iloc[0]  # demo: first valid frame
    true_phase = int(row["phase_id"])
    frame_rel_path = row["frame_path"]
    img_path = ROOT / frame_rel_path

    print(f"[INFO] Demo frame: {img_path}")
    print(f"[INFO] True phase ID: {true_phase}")

    img_normal = Image.open(img_path).convert("RGB")
    img_shift = apply_brightness_blur(img_normal.copy())

    # 2) Load CLIP + text embeddings
    clip_model, preprocess, text_feats, phase_ids = load_clip_and_text()

    # 3) Build adapters
    methods_predictions = {}
    method_names = []

    # Baseline: no adapter
    method_names.append("baseline_fp32")

    # FP32 adapter if exists
    adapter_fp32 = None
    if ADAPTER_FP32_PATH.exists():
        print(f"[INFO] Loading FP32 adapter from {ADAPTER_FP32_PATH}")
        state_fp32 = torch.load(ADAPTER_FP32_PATH, map_location=DEVICE)
        adapter_fp32 = build_adapter_from_state(state_fp32, DEVICE)
        method_names.append("adapter_fp32")
    else:
        print("[WARN] FP32 adapter checkpoint not found, skip adapter methods.")

    # INT8 adapter (CPU) if FP32 is available
    adapter_int8 = None
    if adapter_fp32 is not None:
        print("[INFO] Building INT8 adapter (dynamic quantization on CPU)")
        state_cpu = torch.load(ADAPTER_FP32_PATH, map_location="cpu")
        adapter_cpu = build_adapter_from_state(state_cpu, torch.device("cpu"))
        adapter_int8 = torch.quantization.quantize_dynamic(
            adapter_cpu, {torch.nn.Linear}, dtype=torch.qint8
        )
        adapter_int8.eval()
        method_names.append("adapter_int8")

    # 4) Run predictions under both conditions (normal / brightness_blur)
    methods_predictions = {}

    # Baseline
    print("\n[INFO] Running baseline (no adapter) ...")
    baseline_normal = predict_one_image(
        clip_model,
        preprocess,
        text_feats,
        phase_ids,
        img_normal,
        adapter=None,
        adapter_device=DEVICE,
        method_name="baseline_fp32",
        condition="normal",
    )
    baseline_shift = predict_one_image(
        clip_model,
        preprocess,
        text_feats,
        phase_ids,
        img_shift,
        adapter=None,
        adapter_device=DEVICE,
        method_name="baseline_fp32",
        condition="brightness_blur",
    )
    methods_predictions["baseline_fp32"] = {
        "normal": baseline_normal,
        "brightness_blur": baseline_shift,
    }

    # Adapter FP32
    if adapter_fp32 is not None:
        print("\n[INFO] Running adapter FP32 ...")
        adapter_fp32_normal = predict_one_image(
            clip_model,
            preprocess,
            text_feats,
            phase_ids,
            img_normal,
            adapter=adapter_fp32,
            adapter_device=DEVICE,
            method_name="adapter_fp32",
            condition="normal",
        )
        adapter_fp32_shift = predict_one_image(
            clip_model,
            preprocess,
            text_feats,
            phase_ids,
            img_shift,
            adapter=adapter_fp32,
            adapter_device=DEVICE,
            method_name="adapter_fp32",
            condition="brightness_blur",
        )
        methods_predictions["adapter_fp32"] = {
            "normal": adapter_fp32_normal,
            "brightness_blur": adapter_fp32_shift,
        }

    # Adapter INT8
    if adapter_int8 is not None:
        print("\n[INFO] Running adapter INT8 ...")
        adapter_int8_normal = predict_one_image(
            clip_model,
            preprocess,
            text_feats,
            phase_ids,
            img_normal,
            adapter=adapter_int8,
            adapter_device=torch.device("cpu"),
            method_name="adapter_int8",
            condition="normal",
        )
        adapter_int8_shift = predict_one_image(
            clip_model,
            preprocess,
            text_feats,
            phase_ids,
            img_shift,
            adapter=adapter_int8,
            adapter_device=torch.device("cpu"),
            method_name="adapter_int8",
            condition="brightness_blur",
        )
        methods_predictions["adapter_int8"] = {
            "normal": adapter_int8_normal,
            "brightness_blur": adapter_int8_shift,
        }

    # 5) Save per-frame data to JSON
    demo_result = {
        "frame_path": frame_rel_path,
        "true_phase_id": true_phase,
        "methods": methods_predictions,
    }

    json_path = RESULT_DIR / "demo_reliability_one_frame_results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(demo_result, f, indent=2)
    print(f"\n[INFO] Saved demo per-frame predictions to {json_path}")

    # 6) Save visualization figure
    fig_path = RESULT_DIR / "demo_reliability_one_frame_vis.png"
    visualize_one_frame(
        img_normal=img_normal,
        img_shift=img_shift,
        predictions=methods_predictions,
        true_phase=true_phase,
        fig_path=fig_path,
    )

    print("\n[DONE] Demo reliability (one frame) finished.")


if __name__ == "__main__":
    main()
