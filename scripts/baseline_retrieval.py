import json
from pathlib import Path

import torch
import pandas as pd
from PIL import Image
from tqdm import tqdm
import open_clip


# Root project directory (change if your path is different)
ROOT = Path(r"D:\GU\paper\surgclip")

INDEX_CSV = ROOT / "index" / "keyframe_index.csv"
RESULTS_JSON = ROOT / "results" / "retrieval_scores.json"

# Canonical natural language text for each phase id (1..7).
# We will use these as the candidate texts in the retrieval task.
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
    """
    Load a CLIP model from open_clip-torch.
    We use ViT-B/32 as a simple and widely used baseline.
    """
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
    """
    Compute text embeddings for the 7 candidate phase descriptions.
    Shape: (7, d)
    """
    phase_ids = sorted(PHASE_TEXT.keys())
    texts = [PHASE_TEXT[i] for i in phase_ids]

    with torch.no_grad():
        text_tokens = tokenizer(texts).to(device)
        text_features = model.encode_text(text_tokens)
        # Normalize to unit length for cosine similarity
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    return phase_ids, text_features


def main():
    # ---------------------------------------------------------------------
    # 1) Device selection: use GPU if available, otherwise fall back to CPU
    # ---------------------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    # ---------------------------------------------------------------------
    # 2) Load CLIP model and preprocessing
    # ---------------------------------------------------------------------
    model, preprocess, tokenizer = load_model(device)
    print("[INFO] CLIP model loaded.")

    # ---------------------------------------------------------------------
    # 3) Load index CSV (frame -> phase_id mapping)
    # ---------------------------------------------------------------------
    if not INDEX_CSV.exists():
        raise FileNotFoundError(f"Index CSV not found at {INDEX_CSV}")

    df = pd.read_csv(INDEX_CSV)

    # Optional: you can subsample for quick debugging, e.g. first 5000 frames
    # df = df.sample(n=min(5000, len(df)), random_state=42).reset_index(drop=True)

    print(f"[INFO] Loaded index with {len(df)} frames.")

    # ---------------------------------------------------------------------
    # 4) Prepare text embeddings (7 candidate phase descriptions)
    # ---------------------------------------------------------------------
    phase_ids, text_features = compute_text_embeddings(model, tokenizer, device)
    num_phases = len(phase_ids)
    print(f"[INFO] Prepared text embeddings for {num_phases} phases.")

    # ---------------------------------------------------------------------
    # 5) Loop over frames and perform retrieval
    # ---------------------------------------------------------------------
    top1_correct = 0
    top5_correct = 0
    total = 0

    # We will also store a few example predictions for later qualitative analysis
    examples = []

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Retrieving"):
        frame_rel_path = row["frame_path"]        # e.g. "frames/video01/frame_000000.jpg"
        true_phase_id = int(row["phase_id"])

        # Skip rows with invalid phase id (0 means unknown)
        if true_phase_id not in PHASE_TEXT:
            continue

        frame_path = ROOT / frame_rel_path
        if not frame_path.exists():
            # If a frame is missing for some reason, skip it
            continue

        # ---- Encode image ----
        image = Image.open(frame_path).convert("RGB")
        image_tensor = preprocess(image).unsqueeze(0).to(device)

        with torch.no_grad():
            image_features = model.encode_image(image_tensor)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        # ---- Compute similarity with all text features ----
        # similarity shape: (1, 7)
        similarity = image_features @ text_features.T
        # Get ranking of candidate texts (descending similarity)
        # indices: tensor of shape (1, 7)
        _, indices = similarity.topk(k=num_phases, dim=-1)
        # Convert to list of predicted phase_ids in ranked order
        ranked_phase_ids = [phase_ids[i] for i in indices[0].tolist()]

        # Top-1 accuracy
        if ranked_phase_ids[0] == true_phase_id:
            top1_correct += 1

        # Top-5 accuracy
        if true_phase_id in ranked_phase_ids[:5]:
            top5_correct += 1

        total += 1

        # Store a few examples for later inspection (limit to e.g. 50)
        if len(examples) < 50:
            examples.append({
                "frame_path": frame_rel_path,
                "true_phase_id": true_phase_id,
                "true_phase_text": PHASE_TEXT[true_phase_id],
                "pred_phase_ids": ranked_phase_ids[:5],
                "pred_phase_texts": [PHASE_TEXT[i] for i in ranked_phase_ids[:5]]
            })

    # ---------------------------------------------------------------------
    # 6) Compute final metrics
    # ---------------------------------------------------------------------
    top1_acc = top1_correct / total if total > 0 else 0.0
    top5_acc = top5_correct / total if total > 0 else 0.0

    print(f"[RESULT] Evaluated {total} frames.")
    print(f"[RESULT] Top-1 accuracy: {top1_acc:.4f}")
    print(f"[RESULT] Top-5 accuracy: {top5_acc:.4f}")

    # ---------------------------------------------------------------------
    # 7) Save metrics and examples to JSON for later use
    # ---------------------------------------------------------------------
    RESULTS_JSON.parent.mkdir(parents=True, exist_ok=True)
    result_dict = {
        "num_frames": int(total),
        "top1_accuracy": float(top1_acc),
        "top5_accuracy": float(top5_acc),
        "examples": examples,
    }

    with open(RESULTS_JSON, "w", encoding="utf-8") as f:
        json.dump(result_dict, f, indent=2)

    print(f"[INFO] Results saved to: {RESULTS_JSON}")


if __name__ == "__main__":
    main()
