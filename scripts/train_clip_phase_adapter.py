import math
from pathlib import Path

import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pandas as pd
from tqdm import tqdm
import open_clip


# ==========================
# Global config
# ==========================

ROOT = Path(r"D:\GU\paper\surgclip")
INDEX_CSV = ROOT / "index" / "keyframe_index.csv"

SUBSAMPLE_FRACTION = 0.2      # For fast training debug; set 1.0 later
BATCH_SIZE = 48               # 3070 laptop can handle 48 easily
NUM_WORKERS = 4
LR = 1e-4
EPOCHS = 3                    # You can increase later (5–10)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


PHASE_PROMPTS = {
    1: "Preparation phase of laparoscopic cholecystectomy.",
    2: "Calot triangle dissection during laparoscopic cholecystectomy.",
    3: "Clipping and cutting of cystic duct and cystic artery.",
    4: "Gallbladder dissection from the liver bed.",
    5: "Gallbladder packaging and placement into the retrieval bag.",
    6: "Cleaning and coagulation of the surgical field.",
    7: "Gallbladder retraction and final inspection of the cavity.",
}


# ==========================
# Dataset
# ==========================

class Cholec80PhaseDataset(Dataset):
    """
    CLIP fine-tuning dataset for Cholec80 keyframes.
    """

    def __init__(self, csv_path, preprocess, subsample_fraction=1.0):
        self.root = ROOT
        self.preprocess = preprocess

        df = pd.read_csv(csv_path)
        df = df[df["phase_id"].isin(PHASE_PROMPTS.keys())].reset_index(drop=True)

        if subsample_fraction < 1.0:
            df = df.sample(frac=subsample_fraction, random_state=123).reset_index(drop=True)

        self.df = df
        print(f"[INFO] Dataset initialized with {len(self.df)} frames.")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        img_path = ROOT / row["frame_path"]
        phase_id = int(row["phase_id"])

        image = Image.open(img_path).convert("RGB")
        image_tensor = self.preprocess(image)

        text = PHASE_PROMPTS[phase_id]

        return image_tensor, text, phase_id


# ==========================
# CLIP + Adapter
# ==========================

class ClipPhaseAdapter(nn.Module):
    """
    Fine-tuning module:
      - Freeze CLIP base
      - Add a small learnable adapter after embeddings
    """

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

        # Freeze CLIP backbone
        for p in self.clip.parameters():
            p.requires_grad = False

        emb_dim = clip_model.text_projection.shape[1]   # typically 512

        # Tiny adapter MLP
        self.adapter = nn.Sequential(
            nn.Linear(emb_dim, emb_dim),
            nn.ReLU(),
            nn.Linear(emb_dim, emb_dim),
        )

        self.adapter.to(device)

    def encode_image(self, img):
        with torch.no_grad():
            feats = self.clip.encode_image(img)
        return self.adapter(feats)

    def encode_text(self, texts):
        tokens = self.tokenizer(texts).to(DEVICE)
        with torch.no_grad():
            feats = self.clip.encode_text(tokens)
        return self.adapter(feats)


# ==========================
# CLIP Loss (InfoNCE)
# ==========================

def clip_loss(image_feats, text_feats):
    image_feats = image_feats / image_feats.norm(dim=-1, keepdim=True)
    text_feats = text_feats / text_feats.norm(dim=-1, keepdim=True)

    logits = image_feats @ text_feats.T * 100.0  # temperature scaling

    labels = torch.arange(len(logits), device=logits.device)

    loss_img = nn.CrossEntropyLoss()(logits, labels)
    loss_txt = nn.CrossEntropyLoss()(logits.T, labels)

    return (loss_img + loss_txt) / 2


# ==========================
# Train loop
# ==========================

def main():
    print(f"[INFO] Using device: {DEVICE}")

    model = ClipPhaseAdapter(DEVICE)
    optimizer = optim.Adam(model.adapter.parameters(), lr=LR)

    dataset = Cholec80PhaseDataset(INDEX_CSV, model.preprocess, SUBSAMPLE_FRACTION)

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )

    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0

        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{EPOCHS}")

        for images, texts, _ in pbar:
            images = images.to(DEVICE)

            img_feats = model.encode_image(images)
            txt_feats = model.encode_text(list(texts))

            loss = clip_loss(img_feats, txt_feats)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            pbar.set_postfix({"loss": loss.item()})

        print(f"[INFO] Epoch {epoch+1} average loss = {epoch_loss / len(loader):.4f}")

    # Save adapter weights
    save_path = ROOT / "results" / "clip_adapter.pth"
    torch.save(model.adapter.state_dict(), save_path)
    print(f"[INFO] Saved adapter to: {save_path}")


if __name__ == "__main__":
    main()
