import json
from pathlib import Path

import torch
import pandas as pd
from PIL import Image
from tqdm import tqdm
import open_clip

# Root project directory
ROOT = Path(r"D:\GU\paper\surgclip")

INDEX_CSV = ROOT / "index" / "keyframe_index.csv"
OUT_JSON = ROOT / "results" / "retrieval_examples.json"

PHASE_TEXT = {
    1: "The surgical tools are being inserted and the field is prepared for the procedure.",
    2: "The surgeon is dissecting tissue in Calot's triangle to expose the cystic duct and artery.",
    3: "The cystic duct and artery are being clipped and cut to separate them safely.",
    4: "The gallbladder is being dissected away from the liver bed.",
    5: "The gallbladder is being packed and prepared for removal.",
    6: "Residual bleeding or bile leakage is being controlled by cleaning and coagulation.",
    7: "The gallbladder is being extracted from the abdominal cavity."
}


def load_model(device: torch.device):
    """Load CLIP model and tokenizer from open_clip."""
    model_name = "ViT-B-32"
    pretrained = "openai"

    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name,
        pretrained=pretrained,
        device=device
    )
    tokenizer = open_clip.get_tokenizer(model_name)
    model.eval()
    return model, preprocess, tokenizer


def compute_text_embeddings(model, tokenizer, device):
    """Encode 7 phase texts into CLIP text embeddings."""
    phase_ids = sorted(PHASE_TEXT.keys())
    texts = [PHASE_TEXT[i] for i in phase_ids]
    with torch.no_grad():
        tokens = tokenizer(texts).to(device)
        feats = model.encode_text(tokens)
        feats = feats / feats.norm(dim=-1, keepdim=True)
    return phase_ids, feats


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    if not INDEX_CSV.exists():
        raise FileNotFoundError(INDEX_CSV)

    df = pd.read_csv(INDEX_CSV)
    print(f"[INFO] Loaded index with {len(df)} frames.")

    # We will scan a larger subset to collect more correct examples.
    # Here we simply use the whole dataset and shuffle it.
    df_sample = df.sample(frac=1.0, random_state=123).reset_index(drop=True)
    print(f"[INFO] Shuffled all {len(df_sample)} frames for sampling.")

    model, preprocess, tokenizer = load_model(device)
    phase_ids, text_feats = compute_text_embeddings(model, tokenizer, device)

    # Target numbers of examples for qualitative visualization
    target_correct = 40
    target_wrong = 60

    examples_correct = []
    examples_wrong = []

    for _, row in tqdm(df_sample.iterrows(), total=len(df_sample), desc="Sampling"):
        # Stop early if we already have enough samples
        if len(examples_correct) >= target_correct and len(examples_wrong) >= target_wrong:
            break

        frame_rel = row["frame_path"]
        true_phase_id = int(row["phase_id"])
        if true_phase_id not in PHASE_TEXT:
            continue

        frame_path = ROOT / frame_rel
        if not frame_path.exists():
            continue

        image = Image.open(frame_path).convert("RGB")
        img_tensor = preprocess(image).unsqueeze(0).to(device)

        with torch.no_grad():
            img_feat = model.encode_image(img_tensor)
            img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)

        sim = img_feat @ text_feats.T
        _, idxs = sim.topk(k=len(phase_ids), dim=-1)
        ranked_ids = [phase_ids[i] for i in idxs[0].tolist()]

        example = {
            "frame_path": frame_rel,
            "true_phase_id": true_phase_id,
            "true_phase_text": PHASE_TEXT[true_phase_id],
            "pred_phase_ids": ranked_ids[:5],
            "pred_phase_texts": [PHASE_TEXT[i] for i in ranked_ids[:5]],
        }

        if ranked_ids[0] == true_phase_id:
            if len(examples_correct) < target_correct:
                examples_correct.append(example)
        else:
            if len(examples_wrong) < target_wrong:
                examples_wrong.append(example)

    print(f"[INFO] Collected {len(examples_correct)} correct examples "
          f"and {len(examples_wrong)} wrong examples.")

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(
            {
                "examples_correct": examples_correct,
                "examples_wrong": examples_wrong,
            },
            f,
            indent=2,
        )

    print(f"[INFO] Saved sampled examples to: {OUT_JSON}")


if __name__ == "__main__":
    main()
