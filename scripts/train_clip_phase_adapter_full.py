import math
from pathlib import Path

import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image
import pandas as pd
from tqdm import tqdm
import open_clip


# ==========================
# Global config
# ==========================

ROOT = Path(r"D:\GU\paper\surgclip")
INDEX_CSV = ROOT / "index" / "keyframe_index.csv"

SUBSAMPLE_FRACTION = 1.0      # use full dataset for final training
BATCH_SIZE = 48
NUM_WORKERS = 4
LR = 1e-4
EPOCHS = 8                    # final training epochs

VAL_FRACTION = 0.05           # 5% for validation

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
            df = df.sample(frac=subsample_fraction, random_state=123).reset_index(
                drop=True
            )

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
      - Freeze CLIP base model
      - Train a small adapter on top of embeddings
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

        # freeze clip backbone
        for p in self.clip.parameters():
            p.requires_grad = False

        emb_dim = clip_model.text_projection.shape[1]

        # tiny adapter MLP
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
# CLIP loss (InfoNCE)
# ==========================

def clip_loss(image_feats, text_feats):
    image_feats = image_feats / image_feats.norm(dim=-1, keepdim=True)
    text_feats = text_feats / text_feats.norm(dim=-1, keepdim=True)

    logits = image_feats @ text_feats.T * 100.0

    labels = torch.arange(len(logits), device=logits.device)

    loss_img = nn.CrossEntropyLoss()(logits, labels)
    loss_txt = nn.CrossEntropyLoss()(logits.T, labels)

    return (loss_img + loss_txt) / 2


# ==========================
# Train + validate
# ==========================

def main():
    print(f"[INFO] Using device: {DEVICE}")

    model = ClipPhaseAdapter(DEVICE)
    optimizer = optim.Adam(model.adapter.parameters(), lr=LR)

    full_dataset = Cholec80PhaseDataset(
        INDEX_CSV, model.preprocess, SUBSAMPLE_FRACTION
    )

    # Train/val split
    val_size = max(1, int(len(full_dataset) * VAL_FRACTION))
    train_size = len(full_dataset) - val_size

    train_dataset, val_dataset = random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )

    print(f"[INFO] Train size: {train_size}, Val size: {val_size}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    best_val_loss = float("inf")
    results_dir = ROOT / "results"
    ckpt_dir = results_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(EPOCHS):
        # ---------- train ----------
        model.train()
        train_loss_sum = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [train]")
        for images, texts, _ in pbar:
            images = images.to(DEVICE)

            img_feats = model.encode_image(images)
            txt_feats = model.encode_text(list(texts))

            loss = clip_loss(img_feats, txt_feats)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss_sum += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.3f}"})

        avg_train_loss = train_loss_sum / len(train_loader)

        # ---------- validate ----------
        model.eval()
        val_loss_sum = 0.0

        with torch.no_grad():
            for images, texts, _ in val_loader:
                images = images.to(DEVICE)

                img_feats = model.encode_image(images)
                txt_feats = model.encode_text(list(texts))

                loss = clip_loss(img_feats, txt_feats)
                val_loss_sum += loss.item()

        avg_val_loss = val_loss_sum / len(val_loader)

        print(
            f"[INFO] Epoch {epoch+1}/{EPOCHS}  "
            f"train_loss={avg_train_loss:.4f}  val_loss={avg_val_loss:.4f}"
        )

        # save last epoch
        last_path = results_dir / "clip_adapter_last.pth"
        torch.save(model.adapter.state_dict(), last_path)

        # save best on validation
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_path = results_dir / "clip_adapter_best.pth"
            torch.save(model.adapter.state_dict(), best_path)
            print(f"[INFO] New best model saved to: {best_path}")

        # optional: also keep per-epoch checkpoints
        epoch_path = ckpt_dir / f"clip_adapter_epoch{epoch+1}.pth"
        torch.save(model.adapter.state_dict(), epoch_path)

    print("[INFO] Training finished.")


if __name__ == "__main__":
    main()
