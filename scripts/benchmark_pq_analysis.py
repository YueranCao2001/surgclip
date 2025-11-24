# scripts/benchmark_pq_analysis.py
import os
import time
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

import torch
from torch.utils.data import Dataset, DataLoader
import open_clip
import matplotlib.pyplot as plt


# ==========================
# Global config
# ==========================

# Change this to your local repo root if needed
ROOT = Path(r"D:\GU\paper\surgclip")
INDEX_CSV = ROOT / "index" / "keyframe_index.csv"

RESULT_DIR = ROOT / "results"
FIG_DIR = RESULT_DIR / "figures_pq"
RESULT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Phase descriptions used as text queries for text-to-image retrieval
PHASE_TEXT = {
    1: "The surgical tools are being inserted and the field is prepared for the procedure.",
    2: "The surgeon is dissecting tissue in Calot's triangle to expose the cystic duct and artery.",
    3: "The cystic duct and artery are being clipped and cut to separate them safely.",
    4: "The gallbladder is being dissected away from the liver bed.",
    5: "The gallbladder is being packed and prepared for removal.",
    6: "Residual bleeding or bile leakage is being controlled by cleaning and coagulation.",
    7: "The gallbladder is being extracted from the abdominal cavity.",
}

# PQ configuration (you can later change M/Ks to run additional experiments)
PQ_M = 8                # Number of subspaces
PQ_KS = 256             # Number of centroids per subspace (2^8 = 256)
PQ_TRAIN_SAMPLES = 20000  # Max number of samples used to train PQ codebooks

# For faster debugging, we only use a fraction of frames.
# For final results you can set this to 1.0 (use all).
EVAL_SUBSAMPLE_FRACTION = 0.5


# ==========================
# Dataset: read keyframe_index.csv
# ==========================

class FrameDataset(Dataset):
    """
    Simple dataset that reads keyframe paths and phase labels from keyframe_index.csv
    and applies CLIP's image preprocessing.
    """
    def __init__(self, csv_path, preprocess, sample_fraction=1.0):
        df = pd.read_csv(csv_path)
        # Keep only phases that are in PHASE_TEXT
        df = df[df["phase_id"].isin(PHASE_TEXT.keys())].reset_index(drop=True)

        if sample_fraction < 1.0:
            df = df.sample(frac=sample_fraction, random_state=42).reset_index(drop=True)

        self.df = df
        self.root = ROOT
        self.preprocess = preprocess
        print(f"[INFO] PQ dataset with {len(self.df)} frames (fraction={sample_fraction}).")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_rel = row["frame_path"]
        phase_id = int(row["phase_id"])
        img_path = self.root / img_rel

        image = Image.open(img_path).convert("RGB")
        image_tensor = self.preprocess(image)

        return image_tensor, phase_id, img_rel


# ==========================
# CLIP backbone (baseline)
# ==========================

def load_clip_model():
    """
    Load CLIP model (ViT-B-32, openai) on DEVICE, together with transforms and tokenizer.
    """
    model_name = "ViT-B-32"
    pretrained = "openai"
    clip_model, _, preprocess = open_clip.create_model_and_transforms(
        model_name, pretrained=pretrained, device=DEVICE
    )
    tokenizer = open_clip.get_tokenizer(model_name)
    clip_model.eval()
    print(f"[INFO] Loaded CLIP backbone ({model_name}, {pretrained}) on {DEVICE}.")
    return clip_model, preprocess, tokenizer


@torch.no_grad()
def compute_image_embeddings(clip_model, dataloader):
    """
    Compute FP32 image embeddings (N, D) and phase labels (N,) for all keyframes.
    Embeddings are L2-normalized so that dot product is cosine similarity.
    """
    feats_list = []
    labels_list = []
    paths_list = []

    for imgs, phase_ids, rel_paths in tqdm(dataloader, desc="Encoding images"):
        imgs = imgs.to(DEVICE)
        feats = clip_model.encode_image(imgs)
        # L2 normalization to use dot-product as cosine similarity
        feats = feats / feats.norm(dim=-1, keepdim=True)

        feats_list.append(feats.cpu().numpy())
        labels_list.append(phase_ids.numpy())
        paths_list.extend(rel_paths)

    feats = np.concatenate(feats_list, axis=0)
    labels = np.concatenate(labels_list, axis=0)
    print(f"[INFO] Image embeddings shape: {feats.shape}")
    return feats, labels, paths_list


@torch.no_grad()
def compute_text_embeddings(clip_model, tokenizer):
    """
    Compute FP32 text embeddings (7, D) for the 7 phase descriptions.
    Embeddings are L2-normalized.
    """
    phase_ids = sorted(PHASE_TEXT.keys())
    texts = [PHASE_TEXT[i] for i in phase_ids]
    tokens = tokenizer(texts).to(DEVICE)
    text_feats = clip_model.encode_text(tokens)
    text_feats = text_feats / text_feats.norm(dim=-1, keepdim=True)
    text_feats = text_feats.cpu().numpy()
    print(f"[INFO] Text embeddings shape: {text_feats.shape}")
    return phase_ids, text_feats


# ==========================
# Simple Product Quantizer implementation
# ==========================

class ProductQuantizer:
    """
    Simple (educational) Product Quantization implementation:

    - Pure NumPy, no faiss dependency.
    - Uses L2 distance in each subspace.
    - Assumes dense float32 vectors and dim % M == 0.
    """

    def __init__(self, dim, M=8, Ks=256, n_iter=15, seed=123):
        assert dim % M == 0, "dim must be divisible by M"
        self.dim = dim
        self.M = M
        self.Ks = Ks
        self.subdim = dim // M
        self.n_iter = n_iter
        self.rng = np.random.RandomState(seed)

        # Codebooks: (M, Ks, subdim)
        self.codebooks = None

    def _kmeans(self, x):
        """
        Very simple k-means implementation (no optimizations).
        x: (N, subdim)
        """
        N = x.shape[0]
        assert N >= self.Ks, "Number of training samples must be >= Ks"

        # Randomly initialize centroids
        idx = self.rng.choice(N, self.Ks, replace=False)
        centroids = x[idx].copy()  # (Ks, subdim)

        for it in range(self.n_iter):
            # Distances from points to centroids: (N, Ks)
            dists = ((x[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2)
            labels = dists.argmin(axis=1)

            # Update centroids
            for k in range(self.Ks):
                mask = labels == k
                if np.any(mask):
                    centroids[k] = x[mask].mean(axis=0)
        return centroids

    def fit(self, X, max_train_samples=20000):
        """
        Train PQ codebooks on X (N, D) float32.
        """
        N, D = X.shape
        assert D == self.dim

        # Subsample if too many vectors for k-means
        if N > max_train_samples:
            idx = self.rng.choice(N, max_train_samples, replace=False)
            X_train = X[idx]
            print(f"[PQ] Train on subset {max_train_samples}/{N}")
        else:
            X_train = X
            print(f"[PQ] Train on all {N} samples")

        self.codebooks = np.zeros((self.M, self.Ks, self.subdim), dtype=np.float32)

        for m in range(self.M):
            start = m * self.subdim
            end = (m + 1) * self.subdim
            sub_x = X_train[:, start:end]
            print(f"[PQ] Training subspace {m+1}/{self.M} with dim={self.subdim}")
            self.codebooks[m] = self._kmeans(sub_x)

        print("[PQ] Training done.")

    def encode(self, X):
        """
        Encode X (N, D) into PQ codes (N, M), each code is uint8 index in [0, Ks).
        """
        assert self.codebooks is not None
        N, D = X.shape
        assert D == self.dim

        codes = np.empty((N, self.M), dtype=np.uint8)

        for m in range(self.M):
            start = m * self.subdim
            end = (m + 1) * self.subdim
            sub_x = X[:, start:end]  # (N, subdim)
            cb = self.codebooks[m]   # (Ks, subdim)

            # Distances (N, Ks)
            dists = ((sub_x[:, None, :] - cb[None, :, :]) ** 2).sum(axis=2)
            codes[:, m] = dists.argmin(axis=1).astype(np.uint8)

        return codes

    def decode(self, codes):
        """
        Decode PQ codes (N, M) back to approximate vectors (N, D).
        This is only used for analyzing reconstruction error.
        """
        assert self.codebooks is not None
        N, M = codes.shape
        assert M == self.M

        X_rec = np.zeros((N, self.dim), dtype=np.float32)
        for m in range(self.M):
            cb = self.codebooks[m]    # (Ks, subdim)
            sub_codes = codes[:, m]   # (N,)
            X_rec[:, m * self.subdim:(m + 1) * self.subdim] = cb[sub_codes]
        return X_rec

    def search(self, query, codes, topk):
        """
        Approximate nearest neighbor search for a single query:
          query: (D,)
          codes: (N, M) uint8
        Returns:
          - indices of topk vectors (sorted by increasing distance)
          - corresponding distances
        """
        assert self.codebooks is not None
        D = query.shape[0]
        assert D == self.dim
        N, M = codes.shape
        assert M == self.M

        # Build lookup table for each subspace: (M, Ks)
        q_sub = query.reshape(self.M, self.subdim)
        lut = np.empty((self.M, self.Ks), dtype=np.float32)
        for m in range(self.M):
            cb = self.codebooks[m]   # (Ks, subdim)
            diff = cb - q_sub[m][None, :]
            lut[m] = (diff ** 2).sum(axis=1)

        # Aggregate distances for each vector
        dists = np.zeros((N,), dtype=np.float32)
        for m in range(self.M):
            dists += lut[m][codes[:, m]]

        # Select topk smallest distances
        if topk >= N:
            idx = np.argsort(dists)
        else:
            idx_part = np.argpartition(dists, topk - 1)[:topk]
            idx = idx_part[np.argsort(dists[idx_part])]
        return idx, dists[idx]


# ==========================
# Evaluation: recall@K + latency + memory
# ==========================

def evaluate_retrieval_fp32(image_feats, labels, text_feats, phase_ids, topk_list=(1, 5, 10, 50)):
    """
    Text-to-image retrieval on FP32 embeddings using cosine similarity.

    Returns:
        dict K -> average recall@K across phases.
    """
    N, D = image_feats.shape
    recalls = {k: 0.0 for k in topk_list}

    for pid, text_feat in zip(phase_ids, text_feats):
        mask = (labels == pid)
        num_rel = mask.sum()
        if num_rel == 0:
            continue

        # Similarity scores (N,)
        sims = image_feats @ text_feat  # dot-product of normalized vectors
        order = np.argsort(-sims)       # descending order

        rel_ranks = np.where(mask[order])[0]  # ranks of relevant items

        for k in topk_list:
            hit = (rel_ranks < k).sum()
            recalls[k] += hit / float(num_rel)

    # Average over phases
    num_phases = len(phase_ids)
    for k in topk_list:
        recalls[k] /= num_phases

    return recalls


def evaluate_retrieval_pq(pq, codes, image_labels, text_feats, phase_ids, topk_list=(1, 5, 10, 50)):
    """
    Text-to-image retrieval using PQ distances (approximate).
    """
    N, M = codes.shape
    recalls = {k: 0.0 for k in topk_list}
    topk_max = max(topk_list)

    for pid, text_feat in zip(phase_ids, text_feats):
        mask = (image_labels == pid)
        num_rel = mask.sum()
        if num_rel == 0:
            continue

        idx, _ = pq.search(text_feat, codes, topk_max)  # (topk_max,)

        for k in topk_list:
            topk_idx = idx[:k]
            hit = mask[topk_idx].sum()
            recalls[k] += hit / float(num_rel)

    num_phases = len(phase_ids)
    for k in topk_list:
        recalls[k] /= num_phases

    return recalls


def benchmark_latency_fp32(image_feats, text_feats, num_iters=50):
    """
    Simple baseline latency benchmark:
    repeatedly compute similarity between one text query and all image embeddings.
    """
    N, D = image_feats.shape
    q = text_feats[0]

    t0 = time.time()
    for _ in range(num_iters):
        _ = image_feats @ q
    t1 = time.time()
    return (t1 - t0) / num_iters


def benchmark_latency_pq(pq, codes, text_feats, num_iters=50, topk=50):
    """
    Latency benchmark for PQ-based search:
    repeatedly run pq.search(query, codes, topk).
    """
    q = text_feats[0]

    t0 = time.time()
    for _ in range(num_iters):
        _ = pq.search(q, codes, topk)
    t1 = time.time()
    return (t1 - t0) / num_iters


def estimate_memory_fp32(num_vecs, dim):
    """
    Estimate memory usage of FP32 embeddings.
    """
    # float32: 4 bytes
    return num_vecs * dim * 4 / 1024 / 1024  # MB


def estimate_memory_pq(num_vecs, M, Ks, dim):
    """
    Estimate memory usage for:
      - PQ codes: N * M bytes (uint8)
      - Codebooks: M * Ks * (dim/M) * 4 bytes (float32)
    """
    codes_bytes = num_vecs * M  # uint8
    subdim = dim // M
    codebook_bytes = M * Ks * subdim * 4
    total_mb = (codes_bytes + codebook_bytes) / 1024 / 1024
    return total_mb


# ==========================
# Plotting helpers
# ==========================

def plot_recall_bar(recalls_fp32, recalls_pq, fig_path):
    ks = sorted(recalls_fp32.keys())
    fp32_vals = [recalls_fp32[k] for k in ks]
    pq_vals = [recalls_pq[k] for k in ks]

    x = np.arange(len(ks))
    width = 0.35

    plt.figure(figsize=(7, 5))
    b1 = plt.bar(x - width / 2, fp32_vals, width, label="FP32")
    b2 = plt.bar(x + width / 2, pq_vals, width, label="PQ (INT-encoded)")

    plt.yscale("log") 

    plt.xticks(x, [f"R@{k}" for k in ks])
    plt.ylabel("Average recall (log scale)")
    plt.title("Text-to-Image Retrieval: FP32 vs PQ")
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_path, dpi=300)
    plt.close()
    print(f"[INFO] Saved recall figure to {fig_path}")


def plot_memory_bar(mem_fp32, mem_pq, fig_path):
    labels = ["FP32 embeddings", "PQ codes"]
    vals = [mem_fp32, mem_pq]

    plt.figure(figsize=(6, 4))
    bars = plt.bar(labels, vals)
    for bar, v in zip(bars, vals):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                 f"{v:.2f} MB", ha="center", va="bottom", fontsize=10)
    plt.ylabel("Memory (MB)")
    plt.title("Embedding Storage: FP32 vs PQ")
    plt.tight_layout()
    plt.savefig(fig_path, dpi=300)
    plt.close()
    print(f"[INFO] Saved memory figure to {fig_path}")


def plot_latency_bar(lat_fp32, lat_pq, fig_path):
    labels = ["FP32 full search", "PQ search"]
    vals = [lat_fp32 * 1000, lat_pq * 1000]  # ms

    plt.figure(figsize=(6, 4))
    bars = plt.bar(labels, vals)
    for bar, v in zip(bars, vals):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                 f"{v:.2f} ms", ha="center", va="bottom", fontsize=10)
    plt.ylabel("Avg latency per query (ms)")
    plt.title("Per-Query Retrieval Latency: FP32 vs PQ")
    plt.tight_layout()
    plt.savefig(fig_path, dpi=300)
    plt.close()
    print(f"[INFO] Saved latency figure to {fig_path}")


def plot_representation_error_hist(image_feats, pq, codes, fig_path, num_samples=2000):
    """
    Plot histogram of per-dimension reconstruction errors (FP32 - PQ).
    This helps visualize how much PQ distorts the original embeddings.
    """
    N = image_feats.shape[0]
    num_samples = min(num_samples, N)
    idx = np.random.RandomState(123).choice(N, num_samples, replace=False)

    orig = image_feats[idx]
    rec = pq.decode(codes[idx])

    diff = (orig - rec).reshape(num_samples, -1)
    diff_flat = diff.flatten()

    plt.figure(figsize=(6, 4))
    plt.hist(diff_flat, bins=40)
    plt.xlabel("FP32 - PQ reconstructed value")
    plt.ylabel("Frequency")
    plt.title("Per-dimension Reconstruction Error Distribution (PQ)")
    plt.tight_layout()
    plt.savefig(fig_path, dpi=300)
    plt.close()
    print(f"[INFO] Saved PQ representation error hist to {fig_path}")


# ==========================
# Main
# ==========================

def main():
    print(f"[INFO] Using device: {DEVICE}")

    # 1) Load CLIP
    clip_model, preprocess, tokenizer = load_clip_model()

    # 2) Prepare dataset
    dataset = FrameDataset(INDEX_CSV, preprocess, sample_fraction=EVAL_SUBSAMPLE_FRACTION)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=4)

    # 3) Compute FP32 image & text embeddings
    image_feats, image_labels, frame_paths = compute_image_embeddings(clip_model, dataloader)
    phase_ids, text_feats = compute_text_embeddings(clip_model, tokenizer)

    dim = image_feats.shape[1]
    num_vecs = image_feats.shape[0]

    # 4) Baseline (FP32) retrieval + latency + memory
    topk_list = (1, 5, 10, 50)
    recalls_fp32 = evaluate_retrieval_fp32(image_feats, image_labels, text_feats, phase_ids, topk_list)
    lat_fp32 = benchmark_latency_fp32(image_feats, text_feats, num_iters=50)
    mem_fp32 = estimate_memory_fp32(num_vecs, dim)

    print("[RESULT] FP32 recall:", recalls_fp32)
    print(f"[RESULT] FP32 latency per query: {lat_fp32*1000:.3f} ms")
    print(f"[RESULT] FP32 memory: {mem_fp32:.2f} MB\n")

    # 5) Train PQ and encode all image embeddings
    pq = ProductQuantizer(dim=dim, M=PQ_M, Ks=PQ_KS, n_iter=15, seed=123)
    pq.fit(image_feats.astype(np.float32), max_train_samples=PQ_TRAIN_SAMPLES)
    codes = pq.encode(image_feats.astype(np.float32))  # (N, M)

    # 6) PQ-based retrieval + latency + memory
    recalls_pq = evaluate_retrieval_pq(pq, codes, image_labels, text_feats, phase_ids, topk_list)
    lat_pq = benchmark_latency_pq(pq, codes, text_feats, num_iters=50, topk=max(topk_list))
    mem_pq = estimate_memory_pq(num_vecs, PQ_M, PQ_KS, dim)

    print("[RESULT] PQ recall:", recalls_pq)
    print(f"[RESULT] PQ latency per query: {lat_pq*1000:.3f} ms")
    print(f"[RESULT] PQ memory: {mem_pq:.2f} MB\n")

    # 7) Save numeric results
    summary = {
        "config": {
            "M": PQ_M,
            "Ks": PQ_KS,
            "train_samples": PQ_TRAIN_SAMPLES,
            "eval_fraction": EVAL_SUBSAMPLE_FRACTION,
        },
        "fp32": {
            "recall": recalls_fp32,
            "latency_ms_per_query": lat_fp32 * 1000,
            "memory_MB": mem_fp32,
        },
        "pq": {
            "recall": recalls_pq,
            "latency_ms_per_query": lat_pq * 1000,
            "memory_MB": mem_pq,
        },
        "num_vectors": int(num_vecs),
        "dim": int(dim),
    }

    json_path = RESULT_DIR / "pq_benchmark_summary.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"[INFO] Saved PQ summary JSON to {json_path}")

    # Small CSV for recall curves
    csv_path = RESULT_DIR / "pq_recall_summary.csv"
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("K,recall_fp32,recall_pq\n")
        for k in topk_list:
            f.write(f"{k},{recalls_fp32[k]:.6f},{recalls_pq[k]:.6f}\n")
    print(f"[INFO] Saved PQ recall CSV to {csv_path}")

    # 8) Plot figures
    plot_recall_bar(recalls_fp32, recalls_pq, FIG_DIR / "pq_recall_bar.png")
    plot_memory_bar(mem_fp32, mem_pq, FIG_DIR / "pq_memory_comparison.png")
    plot_latency_bar(lat_fp32, lat_pq, FIG_DIR / "pq_latency_comparison.png")
    plot_representation_error_hist(
        image_feats, pq, codes, FIG_DIR / "pq_representation_error_hist.png", num_samples=2000
    )

    print("[DONE] PQ benchmark finished.")


if __name__ == "__main__":
    main()
