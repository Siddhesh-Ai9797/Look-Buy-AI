import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import sys
from pathlib import Path
import numpy as np
from PIL import Image

import torch
import open_clip


MODEL_NAME = "ViT-B-32"
PRETRAINED = "openai"


def pick_device():
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def main():
    if len(sys.argv) != 3:
        print("Usage: python tools/embed_one_image.py <image_path> <out_npy_path>")
        sys.exit(1)

    image_path = Path(sys.argv[1])
    out_path = Path(sys.argv[2])

    if not image_path.exists():
        raise FileNotFoundError(image_path)

    device = pick_device()
    print("Using device:", device)

    model, _, preprocess = open_clip.create_model_and_transforms(
        MODEL_NAME, pretrained=PRETRAINED
    )
    model = model.to(device)
    model.eval()

    img = Image.open(image_path).convert("RGB")
    x = preprocess(img).unsqueeze(0).to(device)

    with torch.no_grad():
        e = model.encode_image(x)
        e = e / e.norm(dim=-1, keepdim=True)

    vec = e.detach().cpu().numpy().astype("float32")[0]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_path, vec)

    print("✅ Saved embedding:", out_path.resolve())
    print("Shape:", vec.shape)


if __name__ == "__main__":
    main()
