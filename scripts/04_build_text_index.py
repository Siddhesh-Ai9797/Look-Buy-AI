from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm

import faiss
from sentence_transformers import SentenceTransformer


PRODUCTS_PATH = Path("data/processed/products_10k.parquet")
OUT_DIR = Path("data/index_text")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Local embedding model (free). Good quality for semantic retrieval.
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def make_text(row) -> str:
    # For now we only have title + a few IDs; later we'll enrich with more metadata.
    return f"Title: {row['title']}"


def main():
    df = pd.read_parquet(PRODUCTS_PATH)
    texts = [make_text(r) for _, r in df.iterrows()]

    model = SentenceTransformer(MODEL_NAME)

    # Encode to embeddings (float32)
    embs = model.encode(
        texts,
        batch_size=64,
        show_progress_bar=True,
        normalize_embeddings=True
    ).astype("float32")

    dim = embs.shape[1]

    # FAISS index (cosine similarity via inner product on normalized vectors)
    index = faiss.IndexFlatIP(dim)
    index.add(embs)

    # Save index + metadata mapping
    faiss.write_index(index, str(OUT_DIR / "faiss.index"))
    df[["item_id", "title", "main_image_id", "path"]].to_parquet(OUT_DIR / "meta.parquet", index=False)

    print("✅ Saved FAISS index to:", (OUT_DIR / "faiss.index").resolve())
    print("✅ Saved metadata to:", (OUT_DIR / "meta.parquet").resolve())
    print("Embeddings:", embs.shape)


if __name__ == "__main__":
    main()
