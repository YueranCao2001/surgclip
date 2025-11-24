import json
from pathlib import Path

import torch
from torch import nn
from torch.quantization import quantize_dynamic
import pandas as pd
from PIL import Image
from tqdm import tqdm
import open_clip


ROOT = Path(r"D:\GU\paper\surgclip")
INDEX_CSV = ROOT / "index" / "keyframe_index.csv"
ADAPTER_PATH = ROOT / "results" / "clip_adapter_best.pth"

OUT_JSON = ROOT / "results" / "retrieval_scores_adapted_int8.json"

DEVICE = torch.device("cpu")


PHASE_TEXT = {
    1: "The surgical tools are being inserted and the field is prepared for the procedure.",
    2: "The surgeon is dissecting tissue in Calot's triangle to expose the cystic duct and artery.",
    3: "The cystic duct and artery are being clipped and cut to separate them safely.",
    4: "The gallbladder is being dissected away from the liver bed.",
    5: "The gallbladder is being packed and prepared for removal.",
    6: "Residual bleeding or bile leakage is being controlled by cleaning and coagulation.",
    7: "The gallbladder is being extracted from the abdominal cavity.",
}


# =====================
# Model definition
# =====================
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

        for p in self.clip.parameters():
            p.requires_grad = False

        emb_dim = clip_model.text_projection.shape[1]

        # adapter MLP
        self.adapter = nn.Sequential(
            nn.Linear(emb_dim, emb_dim),
            nn.ReLU(),
            nn.Linear(emb_dim, emb_dim),
        )

        self.to(device)

    def encode_image(self, imgs):
        with torch.no_grad():
            feats = self.clip.encode_image(imgs)
        return self.adapter(feats)

    def encode_text(self, texts):
        tokens = self.tokenizer(texts).to(DEVICE)
        with torch.no_grad():
            feats = self.clip.encode_text(tokens)
        return self.adapter(feats)


# =====================
# Main INT8 eval
# =====================
def main():
    print("[INFO] Loading FP32 CLIP + Adapter")

    model_fp32 = ClipPhaseAdapter(DEVICE)

    # load adapter weights
    state = torch.load(ADAPTER_PATH, map_location=DEVICE)
    model_fp32.adapter.load_state_dict(state)

    print("[INFO] Quantizing ONLY the adapter (INT8 linear)")
    model_fp32.adapter = quantize_dynamic(
        model_fp32.adapter,
        {nn.Linear},
        dtype=torch.qint8,
    )

    model_fp32.eval()

    # precompute text embeddings
    phase_ids = sorted(PHASE_TEXT.keys())
    texts = [PHASE_TEXT[i] for i in phase_ids]

    with torch.no_grad():
        text_feat = model_fp32.encode_text(texts)
        text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)

    df = pd.read_csv(INDEX_CSV)

    total = 0
    top1 = 0
    top5 = 0

    examples = []

    for _, row in tqdm(df.iterrows(), total=len(df)):
        img_path = ROOT / row["frame_path"]
        true_pid = int(row["phase_id"])

        image = Image.open(img_path).convert("RGB")
        img_tensor = model_fp32.preprocess(image).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            img_feat = model_fp32.encode_image(img_tensor)
            img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
            sim = img_feat @ text_feat.T

        values, indices = sim.topk(k=5, dim=-1)
        pred_ids = [phase_ids[i] for i in indices[0].tolist()]

        total += 1
        if pred_ids[0] == true_pid:
            top1 += 1
        if true_pid in pred_ids:
            top5 += 1

        if len(examples) < 50:
            examples.append({
                "frame_path": row["frame_path"],
                "true_phase_id": true_pid,
                "pred_phase_ids": pred_ids
            })

    result = {
        "num_frames": total,
        "top1_accuracy": top1 / total,
        "top5_accuracy": top5 / total,
        "examples": examples,
        "quantization": "INT8 adapter only",
    }

    with open(OUT_JSON, "w", encoding="utf8") as f:
        json.dump(result, f, indent=2)

    print(result)
    print(f"[INFO] Saved to {OUT_JSON}")


if __name__ == "__main__":
    main()

