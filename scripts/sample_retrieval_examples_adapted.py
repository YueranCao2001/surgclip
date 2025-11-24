import json
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pandas as pd
from tqdm import tqdm
import open_clip


# ==========================
# Paths & config
# ==========================

ROOT = Path(r"D:\GU\paper\surgclip")
INDEX_CSV = ROOT / "index" / "keyframe_index.csv"
ADAPTER_PATH = ROOT / "results" / "clip_adapter_best.pth"
OUT_JSON = ROOT / "results" / "retrieval_examples_adapted.json"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SAMPLE_FRACTION = 0.3          # only sample 30% of frames for speed
BATCH_SIZE = 64
NUM_WORKERS = 4

MAX_SAVED_PER_CLASS = 80       # up to 80 examples for each category (later取12张就够)


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
# Dataset
# ==========================

class FrameDataset(Dataset):
    def __init__(self, csv_path, preprocess, sample_fraction=1.0):
        df = pd.read_csv(csv_path)
        df = df[df["phase_id"].isin(PHASE_TEXT.keys())].reset_index(drop=True)

        if sample_fraction < 1.0:
            df = df.sample(frac=sample_fraction, random_state=42).reset_index(
                drop=True
            )

        self.df = df
        self.root = ROOT
        self.preprocess = preprocess
        print(f"[INFO] Sampling dataset with {len(self.df)} frames.")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        frame_rel = row["frame_path"]
        phase_id = int(row["phase_id"])
        img_path = self.root / frame_rel

        image = Image.open(img_path).convert("RGB")
        image_tensor = self.preprocess(image)

        return image_tensor, phase_id, frame_rel


# ==========================
# CLIP + Adapter
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

        # freeze clip backbone
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
# Main sampling logic
# ==========================

def main():
    print(f"[INFO] Using device: {DEVICE}")

    # 1) load adapter
    model = ClipPhaseAdapter(DEVICE)
    if not ADAPTER_PATH.exists():
        raise FileNotFoundError(ADAPTER_PATH)
    state = torch.load(ADAPTER_PATH, map_location=DEVICE)
    model.adapter.load_state_dict(state)
    model.eval()
    print(f"[INFO] Loaded adapter from {ADAPTER_PATH}")

    # 2) precompute text embeddings
    phase_ids = sorted(PHASE_TEXT.keys())
    phase_texts = [PHASE_TEXT[i] for i in phase_ids]

    with torch.no_grad():
        text_feats = model.encode_text(phase_texts)
        text_feats = text_feats / text_feats.norm(dim=-1, keepdim=True)

    # 3) dataset & loader
    dataset = FrameDataset(INDEX_CSV, model.preprocess, SAMPLE_FRACTION)
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    top1_correct = []
    top5_correct_only = []
    failures = []

    for images, phase_ids_batch, frame_paths in tqdm(loader, desc="Sampling"):
        images = images.to(DEVICE)

        with torch.no_grad():
            img_feats = model.encode_image(images)
            img_feats = img_feats / img_feats.norm(dim=-1, keepdim=True)
            sim = img_feats @ text_feats.T         # (B, 7)
            values, indices = sim.topk(k=5, dim=-1)

        for i in range(images.size(0)):
            true_phase = int(phase_ids_batch[i])
            pred_phase_ids = [phase_ids[j] for j in indices[i].tolist()]

            record = {
                "frame_path": frame_paths[i],
                "true_phase_id": true_phase,
                "true_phase_text": PHASE_TEXT[true_phase],
                "pred_phase_ids": pred_phase_ids,
                "pred_phase_texts": [PHASE_TEXT[j] for j in pred_phase_ids],
            }

            if pred_phase_ids[0] == true_phase:
                if len(top1_correct) < MAX_SAVED_PER_CLASS:
                    top1_correct.append(record)
            elif true_phase in pred_phase_ids:
                if len(top5_correct_only) < MAX_SAVED_PER_CLASS:
                    top5_correct_only.append(record)
            else:
                if len(failures) < MAX_SAVED_PER_CLASS:
                    failures.append(record)

        # stop early when enough samples for all three types
        if (
            len(top1_correct) >= MAX_SAVED_PER_CLASS
            and len(top5_correct_only) >= MAX_SAVED_PER_CLASS
            and len(failures) >= MAX_SAVED_PER_CLASS
        ):
            break

    print(f"[INFO] Collected: "
          f"top1_correct={len(top1_correct)}, "
          f"top5_correct_only={len(top5_correct_only)}, "
          f"failures={len(failures)}")

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(
            {
                "top1_correct": top1_correct,
                "top5_correct_only": top5_correct_only,
                "failures": failures,
            },
            f,
            indent=2,
        )

    print(f"[INFO] Saved adapted examples to: {OUT_JSON}")


if __name__ == "__main__":
    main()
