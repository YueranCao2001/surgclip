import json
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
OUT_JSON = ROOT / "results" / "baseline_vs_adapter_examples.json"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SAMPLE_FRACTION = 0.3       # 只采样 30% 帧就够
BATCH_SIZE = 64
NUM_WORKERS = 4
MAX_PER_CATEGORY = 80       # 每种情况最多存 80 条，后面再挑 12 条画图


PHASE_TEXT = {
    1: "The surgical tools are being inserted and the field is prepared for the procedure.",
    2: "The surgeon is dissecting tissue in Calot's triangle to expose the cystic duct and artery.",
    3: "The cystic duct and artery are being clipped and cut to separate them safely.",
    4: "The gallbladder is being dissected away from the liver bed.",
    5: "The gallbladder is being packed and prepared for removal.",
    6: "Residual bleeding or bile leakage is being controlled by cleaning and coagulation.",
    7: "The gallbladder is being extracted from the abdominal cavity.",
}


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
        print(f"[INFO] Sampling {len(self.df)} frames for comparison.")

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
        # baseline: 只用 CLIP
        with torch.no_grad():
            feats = self.clip.encode_image(imgs)
        return feats

    def encode_image_adapted(self, imgs):
        # adapted: 过 adapter
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


def main():
    print(f"[INFO] Using device: {DEVICE}")
    model = ClipPhaseAdapter(DEVICE)

    # load adapter
    state = torch.load(ADAPTER_PATH, map_location=DEVICE)
    model.adapter.load_state_dict(state)
    model.eval()
    print(f"[INFO] Loaded adapter from {ADAPTER_PATH}")

    # text embeddings
    phase_ids = sorted(PHASE_TEXT.keys())
    texts = [PHASE_TEXT[i] for i in phase_ids]

    with torch.no_grad():
        txt_base = model.encode_text_base(texts)
        txt_base = txt_base / txt_base.norm(dim=-1, keepdim=True)

        txt_adapt = model.encode_text_adapted(texts)
        txt_adapt = txt_adapt / txt_adapt.norm(dim=-1, keepdim=True)

    dataset = FrameDataset(INDEX_CSV, model.preprocess, SAMPLE_FRACTION)
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    improved = []   # baseline wrong, adapter correct
    unchanged_good = []   # both correct
    unchanged_bad = []    # both wrong
    degraded = []  # baseline correct, adapter wrong

    for imgs, gt_ids, frame_paths in tqdm(loader, desc="Comparing"):
        imgs = imgs.to(DEVICE)
        gt_ids = [int(x) for x in gt_ids.tolist()]

        with torch.no_grad():
            im_base = model.encode_image_base(imgs)
            im_base = im_base / im_base.norm(dim=-1, keepdim=True)
            logits_base = im_base @ txt_base.T
            _, idx_base = logits_base.topk(k=1, dim=-1)
            preds_base = [phase_ids[i] for i in idx_base.squeeze(-1).tolist()]

            im_ad = model.encode_image_adapted(imgs)
            im_ad = im_ad / im_ad.norm(dim=-1, keepdim=True)
            logits_ad = im_ad @ txt_adapt.T
            _, idx_ad = logits_ad.topk(k=1, dim=-1)
            preds_ad = [phase_ids[i] for i in idx_ad.squeeze(-1).tolist()]

        for gt, pb, pa, fp in zip(gt_ids, preds_base, preds_ad, frame_paths):
            rec = {
                "frame_path": fp,
                "gt_id": gt,
                "gt_text": PHASE_TEXT[gt],
                "baseline_pred_id": pb,
                "baseline_pred_text": PHASE_TEXT[pb],
                "adapted_pred_id": pa,
                "adapted_pred_text": PHASE_TEXT[pa],
            }

            if pb != gt and pa == gt:
                if len(improved) < MAX_PER_CATEGORY:
                    improved.append(rec)
            elif pb == gt and pa == gt:
                if len(unchanged_good) < MAX_PER_CATEGORY:
                    unchanged_good.append(rec)
            elif pb != gt and pa != gt:
                if len(unchanged_bad) < MAX_PER_CATEGORY:
                    unchanged_bad.append(rec)
            elif pb == gt and pa != gt:
                if len(degraded) < MAX_PER_CATEGORY:
                    degraded.append(rec)

        if (
            len(improved) >= MAX_PER_CATEGORY
            and len(unchanged_good) >= MAX_PER_CATEGORY
            and len(unchanged_bad) >= MAX_PER_CATEGORY
            and len(degraded) >= MAX_PER_CATEGORY
        ):
            break

    print(
        f"[INFO] Collected examples: "
        f"improved={len(improved)}, "
        f"unchanged_good={len(unchanged_good)}, "
        f"unchanged_bad={len(unchanged_bad)}, "
        f"degraded={len(degraded)}"
    )

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(
            {
                "improved": improved,
                "unchanged_good": unchanged_good,
                "unchanged_bad": unchanged_bad,
                "degraded": degraded,
            },
            f,
            indent=2,
        )

    print(f"[INFO] Saved comparison examples to: {OUT_JSON}")


if __name__ == "__main__":
    main()
