import json
from pathlib import Path

import torch
import pandas as pd
from PIL import Image
from tqdm import tqdm
import open_clip
from torch import nn


# ==========================
# Paths & config
# ==========================

ROOT = Path(r"D:\GU\paper\surgclip")
INDEX_CSV = ROOT / "index" / "keyframe_index.csv"
ADAPTER_PATH = ROOT / "results" / "clip_adapter_best.pth"
OUT_JSON = ROOT / "results" / "retrieval_scores_adapted.json"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EVAL_SUBSAMPLE_FRACTION = 1.0   # 可以先用 0.3 调试，再改 1.0 跑完整


PHASE_TEXT = {
    1: "The surgical tools are being inserted and the field is prepared for the procedure.",
    2: "The surgeon is dissecting tissue in Calot's triangle to expose the cystic duct and artery.",
    3: "The cystic duct and artery are being clipped and cut to separate them safely.",
    4: "The gallbladder is being dissected away from the liver bed.",
    5: "The gallbladder is being packed and prepared for removal.",
    6: "Residual bleeding or bile leakage is being controlled by cleaning and coagulation.",
    7: "The gallbladder is being extracted from the abdominal cavity.",
}


# ==========================
# CLIP + Adapter (same as train)
# ==========================

class ClipPhaseAdapter(nn.Module):
    def __init__(self, device):
        super().__init__()

        model_name = "ViT-B-32"
        pretrained = "openai"

        clip_model, _, preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained, device=device
        )
        tokenizer = open_clip.get_tokenizer(model_name)

        self.clip = clip_model
        self.preprocess = preprocess
        self.tokenizer = tokenizer

        # freeze clip
        for p in self.clip.parameters():
            p.requires_grad = False

        emb_dim = clip_model.text_projection.shape[1]
        self.adapter = nn.Sequential(
            nn.Linear(emb_dim, emb_dim),
            nn.ReLU(),
            nn.Linear(emb_dim, emb_dim),
        )

        self.adapter.to(device)

    def encode_image(self, imgs):
        with torch.no_grad():
            feats = self.clip.encode_image(imgs)
        return self.adapter(feats)

    def encode_text(self, texts):
        tokens = self.tokenizer(texts).to(DEVICE)
        with torch.no_grad():
            feats = self.clip.encode_text(tokens)
        return self.adapter(feats)


# ==========================
# Main eval
# ==========================

def main():
    print(f"[INFO] Using device: {DEVICE}")

    # 1) Load model + adapter weights
    model = ClipPhaseAdapter(DEVICE)
    if not ADAPTER_PATH.exists():
        raise FileNotFoundError(ADAPTER_PATH)
    state = torch.load(ADAPTER_PATH, map_location=DEVICE)
    model.adapter.load_state_dict(state)
    model.eval()
    print(f"[INFO] Loaded adapter from {ADAPTER_PATH}")

    # 2) Precompute text embeddings for 7 phases
    phase_ids = sorted(PHASE_TEXT.keys())
    phase_texts = [PHASE_TEXT[i] for i in phase_ids]

    with torch.no_grad():
        text_feats = model.encode_text(phase_texts)
        text_feats = text_feats / text_feats.norm(dim=-1, keepdim=True)

    # 3) Load index
    df = pd.read_csv(INDEX_CSV)
    df = df[df["phase_id"].isin(phase_ids)].reset_index(drop=True)

    if EVAL_SUBSAMPLE_FRACTION < 1.0:
        df = df.sample(frac=EVAL_SUBSAMPLE_FRACTION, random_state=42).reset_index(drop=True)

    print(f"[INFO] Evaluating on {len(df)} frames (subsample={EVAL_SUBSAMPLE_FRACTION}).")

    top1_correct = 0
    top5_correct = 0
    total = 0

    examples = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Evaluating"):
        frame_rel = row["frame_path"]
        true_phase = int(row["phase_id"])

        img_path = ROOT / frame_rel
        image = Image.open(img_path).convert("RGB")
        img_tensor = model.preprocess(image).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            img_feat = model.encode_image(img_tensor)
            img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
            sim = img_feat @ text_feats.T    # 1 x 7

            values, indices = sim.topk(k=5, dim=-1)
            pred_ids = [phase_ids[i] for i in indices[0].tolist()]

        total += 1
        if pred_ids[0] == true_phase:
            top1_correct += 1
        if true_phase in pred_ids:
            top5_correct += 1

        # Save a few examples for qualitative use
        if len(examples) < 50:
            examples.append(
                {
                    "frame_path": frame_rel,
                    "true_phase_id": true_phase,
                    "true_phase_text": PHASE_TEXT[true_phase],
                    "pred_phase_ids": pred_ids,
                    "pred_phase_texts": [PHASE_TEXT[i] for i in pred_ids],
                }
            )

    top1_acc = top1_correct / total
    top5_acc = top5_correct / total

    print(f"[RESULT] Adapted CLIP – Top-1 accuracy: {top1_acc:.4f}")
    print(f"[RESULT] Adapted CLIP – Top-5 accuracy: {top5_acc:.4f}")
    print(f"[INFO] Evaluated {total} frames.")

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(
            {
                "num_frames": total,
                "top1_accuracy": top1_acc,
                "top5_accuracy": top5_acc,
                "examples": examples,
            },
            f,
            indent=2,
        )

    print(f"[INFO] Saved adapted retrieval scores to: {OUT_JSON}")


if __name__ == "__main__":
    main()
