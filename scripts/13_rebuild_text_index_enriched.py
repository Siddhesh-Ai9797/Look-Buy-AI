from pathlib import Path
import numpy as np
import pandas as pd
import faiss
from tqdm import tqdm
from sentence_transformers import SentenceTransformer


BASE_DIR = Path(__file__).resolve().parent.parent

IN_PATH = BASE_DIR / "data" / "processed" / "products_10k_enriched.parquet"

OUT_DIR = BASE_DIR / "data" / "index_text"
OUT_INDEX = OUT_DIR / "faiss.index"
OUT_META = OUT_DIR / "meta.parquet"

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def main():
    if not IN_PATH.exists():
        raise FileNotFoundError(f"Missing: {IN_PATH}")

    df = pd.read_parquet(IN_PATH)

    # We will index text_for_index
    if "text_for_index" not in df.columns:
        raise ValueError("Expected column 'text_for_index' in products_10k_enriched.parquet")

    texts = df["text_for_index"].fillna("").astype(str).tolist()

    # Keep metadata to show in results
    keep_cols = ["item_id", "title", "caption", "brand", "category", "color"]
    meta = df[[c for c in keep_cols if c in df.columns]].copy()

    print("Loading embedding model:", MODEL_NAME)
    model = SentenceTransformer(MODEL_NAME)

    # Encode in batches
    batch_size = 64
    emb_list = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Embedding batches"):
        chunk = texts[i : i + batch_size]
        vec = model.encode(chunk, normalize_embeddings=True)
        emb_list.append(vec)

    embs = np.vstack(emb_list).astype("float32")
    print("Embeddings:", embs.shape)

    # Build FAISS index (cosine similarity via inner product since normalized)
    dim = embs.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embs)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(OUT_INDEX))
    meta.to_parquet(OUT_META, index=False)

    print(f"✅ Saved FAISS text index -> {OUT_INDEX}")
    print(f"✅ Saved text metadata -> {OUT_META}")
    print("Rows:", len(meta))


if __name__ == "__main__":
    main()
