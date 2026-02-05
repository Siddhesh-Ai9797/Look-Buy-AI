import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

import torch
import open_clip


PRODUCTS_PATH = Path("data/processed/products_10k.parquet")
IMAGE_DIR = Path("data/images")
OUT_DIR = Path("data/index_image")
OUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "ViT-B-32"
PRETRAINED = "openai"


def pick_device() -> str:
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def main():
    device = pick_device()
    print("Using device:", device)

    df = pd.read_parquet(PRODUCTS_PATH)

    model, _, preprocess = open_clip.create_model_and_transforms(
        MODEL_NAME, pretrained=PRETRAINED
    )
    model = model.to(device)
    model.eval()

    embeddings = []
    kept_rows = []

    for _, row in tqdm(df.iterrows(), total=len(df)):
        item_id = row["item_id"]
        img_path = next(IMAGE_DIR.glob(f"{item_id}.*"), None)

        if img_path is None:
            continue

        try:
            img = Image.open(img_path).convert("RGB")
            x = preprocess(img).unsqueeze(0).to(device)

            with torch.no_grad():
                e = model.encode_image(x)
                e = e / e.norm(dim=-1, keepdim=True)

            embeddings.append(e.cpu().numpy()[0])
            kept_rows.append(row)

        except Exception:
            continue

    if not embeddings:
        raise RuntimeError("No image embeddings created")

    embeddings = np.vstack(embeddings).astype("float32")
    meta = pd.DataFrame(kept_rows)

    np.save(OUT_DIR / "clip_embs.npy", embeddings)
    meta.to_parquet(OUT_DIR / "meta.parquet", index=False)

    print("✅ Saved embeddings:", embeddings.shape)
    print("✅ Saved meta:", (OUT_DIR / "meta.parquet").resolve())


if __name__ == "__main__":
    main()
