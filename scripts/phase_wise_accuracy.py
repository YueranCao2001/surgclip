from pathlib import Path

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pandas as pd
from tqdm import tqdm
import open_clip


ROOT = Path(r"D:\GU\paper\surgclip")
INDEX_CSV = ROOT / "index" / "keyframe_index.csv"
ADAPTER_PATH = ROOT / "results" / "clip_adapter_best.pth"
OUT_CSV = ROOT / "results" / "phase_wise_accuracy.csv"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BATCH_SIZE = 64
NUM_WORKERS = 4

PHASE_NAMES = {
    1: "Preparation",
    2: "Calot Triangle Dissection",
    3: "Clipping & Cutting",
    4: "Gallbladder Dissection",
    5: "Gallbladder Packaging",
    6: "Cleaning & Coagulation",
    7: "Gallbladder Retraction",
}


class FrameDataset(Dataset):
    def __init__(self, csv_path, preprocess):
        df = pd.read_csv(csv_path)
        df = df[df["phase_id"].isin(PHASE_NAMES.keys())].reset_index(drop=True)

        self.df = df
        self.root = ROOT
        self.preprocess = preprocess
        print(f"[INFO] Eval dataset size: {len(self.df)}")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        frame_rel = row["frame_path"]
        phase_id = int(row["phase_id"])
        img_path = self.root / frame_rel

        image = Image.open(img_path).convert("RGB")
        img_tensor = self.preprocess(image)

        return img_tensor, phase_id


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
        self.adapter = nn.Sequential(
            nn.Linear(emb_dim, emb_dim),
            nn.ReLU(),
            nn.Linear(emb_dim, emb_dim),
        )
        self.adapter.to(device)

    def encode_image_base(self, imgs):
        with torch.no_grad():
            feats = self.clip.encode_image(imgs)
        return feats

    def encode_image_adapted(self, imgs):
        with torch.no_grad():
            feats = self.clip.encode_image(imgs)
        return self.adapter(feats)

    def encode_text_base(self, texts):
        tokens = self.tokenizer(texts).to(DEVICE)
        with torch.no_grad():
            feats = self.clip.encode_text(tokens)
        return feats

    def encode_text_adapted(self, texts):
        tokens = self.tokenizer(texts).to(DEVICE)
        with torch.no_grad():
            feats = self.clip.encode_text(tokens)
        return self.adapter(feats)


def eval_model(model, dataset, mode="baseline"):
    phase_ids = sorted(PHASE_NAMES.keys())
    texts = [PHASE_NAMES[i] for i in phase_ids]

    if mode == "baseline":
        txt_feats = model.encode_text_base(texts)
    else:
        txt_feats = model.encode_text_adapted(texts)

    txt_feats = txt_feats / txt_feats.norm(dim=-1, keepdim=True)

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    # per-phase counters
    counts = {p: 0 for p in phase_ids}
    top1_correct = {p: 0 for p in phase_ids}
    top5_correct = {p: 0 for p in phase_ids}

    for imgs, gt_ids in tqdm(loader, desc=f"Eval {mode}"):
        imgs = imgs.to(DEVICE)
        gt_ids = [int(x) for x in gt_ids.tolist()]

        if mode == "baseline":
            im_feats = model.encode_image_base(imgs)
        else:
            im_feats = model.encode_image_adapted(imgs)

        im_feats = im_feats / im_feats.norm(dim=-1, keepdim=True)
        sim = im_feats @ txt_feats.T          # (B,7)
        values, indices = sim.topk(k=5, dim=-1)

        for i, gt in enumerate(gt_ids):
            counts[gt] += 1
            pred_ids = [phase_ids[j] for j in indices[i].tolist()]

            if pred_ids[0] == gt:
                top1_correct[gt] += 1
            if gt in pred_ids:
                top5_correct[gt] += 1

    return counts, top1_correct, top5_correct


def main():
    print(f"[INFO] Using device: {DEVICE}")
    model = ClipPhaseAdapter(DEVICE)

    # load adapter
    state = torch.load(ADAPTER_PATH, map_location=DEVICE)
    model.adapter.load_state_dict(state)
    model.eval()
    print(f"[INFO] Loaded adapter from {ADAPTER_PATH}")

    dataset = FrameDataset(INDEX_CSV, model.preprocess)

    # baseline
    base_counts, base_t1, base_t5 = eval_model(model, dataset, mode="baseline")
    # adapted
    ad_counts, ad_t1, ad_t5 = eval_model(model, dataset, mode="adapted")

    rows = []
    for p in sorted(PHASE_NAMES.keys()):
        n = base_counts[p]
        row = {
            "phase_id": p,
            "phase_name": PHASE_NAMES[p],
            "num_frames": n,
            "baseline_top1": base_t1[p] / n,
            "baseline_top5": base_t5[p] / n,
            "adapted_top1": ad_t1[p] / n,
            "adapted_top5": ad_t5[p] / n,
        }
        rows.append(row)

    df_out = pd.DataFrame(rows)
    df_out.to_csv(OUT_CSV, index=False)
    print(f"[INFO] Saved phase-wise accuracy to {OUT_CSV}")
    print(df_out.round(3))


if __name__ == "__main__":
    main()
