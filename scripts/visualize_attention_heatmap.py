from pathlib import Path

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import open_clip


ROOT = Path(r"D:\GU\paper\surgclip")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

IMAGE_PATH = ROOT / "frames" / "video01" / "frame_000100.jpg"   # 你可以自己换一个
OUT_PATH = ROOT / "results" / "figures" / "attention_example.png"


def get_vit_patches(model, image_tensor):
    """
    Try to get patch embeddings before pooling.
    This uses open_clip VisionTransformer interface and may need minor tweaks
    depending on version.
    """
    visual = model.visual   # VisionTransformer

    # Most ViT-like models in CLIP/open_clip have a forward method that can
    # return tokens if we bypass the final projection. We implement a small
    # manual forward based on standard CLIP ViT code.

    x = image_tensor

    # the following attributes are typical; if any line fails, check your
    # open_clip version and adjust accordingly.
    x = visual.conv1(x)                        # (B, C, H', W')
    x = x.reshape(x.shape[0], x.shape[1], -1)  # (B, C, HW)
    x = x.permute(0, 2, 1)                     # (B, HW, C)
    cls_token = visual.class_embedding.to(x.dtype)
    cls_tokens = cls_token.expand(x.shape[0], 1, -1)
    x = torch.cat([cls_tokens, x], dim=1)      # (B, 1+HW, C)
    x = x + visual.positional_embedding.to(x.dtype)
    x = visual.ln_pre(x)

    for blk in visual.transformer.resblocks:
        x = blk(x)

    x = visual.ln_post(x)                      # (B, 1+HW, C)

    return x   # return all tokens (cls + patches)


def main():
    print(f"[INFO] Using device: {DEVICE}")
    model_name = "ViT-B-32"
    pretrained = "openai"
    clip_model, preprocess, _ = open_clip.create_model_and_transforms(
        model_name, pretrained=pretrained, device=DEVICE
    )

    img = Image.open(IMAGE_PATH).convert("RGB")
    img_t = preprocess(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        tokens = get_vit_patches(clip_model, img_t)   # (1, 1+HW, C)

    cls = tokens[:, 0, :]               # (1, C)
    patches = tokens[:, 1:, :]          # (1, HW, C)

    # cosine similarity between each patch and cls token
    cls_norm = F.normalize(cls, dim=-1)         # (1, C)
    patch_norm = F.normalize(patches, dim=-1)   # (1, HW, C)
    sim = torch.einsum("bd,bnd->bn", cls_norm, patch_norm)  # (1, HW)
    sim_map = sim[0].reshape(int(sim.shape[1] ** 0.5), -1)  # (H_p, W_p)

    # normalize to [0,1]
    sim_map = (sim_map - sim_map.min()) / (sim_map.max() - sim_map.min() + 1e-6)

    # upsample to image size
    sim_map_upsampled = F.interpolate(
        sim_map.unsqueeze(0).unsqueeze(0),
        size=img.size[::-1],  # (H, W)
        mode="bilinear",
        align_corners=False,
    )[0, 0].cpu().numpy()

    # plot
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].imshow(img)
    ax[0].axis("off")
    ax[0].set_title("Original")

    ax[1].imshow(img)
    ax[1].imshow(sim_map_upsampled, cmap="jet", alpha=0.45)
    ax[1].axis("off")
    ax[1].set_title("Approx. Attention Heatmap")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PATH, dpi=220)
    plt.close(fig)
    print(f"[INFO] Saved attention visualization to {OUT_PATH}")


if __name__ == "__main__":
    main()
