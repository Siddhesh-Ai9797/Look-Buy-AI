from pathlib import Path
import faiss
import pandas as pd
from sentence_transformers import SentenceTransformer
import numpy as np


INDEX_DIR = Path("data/index_text")
INDEX_PATH = INDEX_DIR / "faiss.index"
META_PATH = INDEX_DIR / "meta.parquet"

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def main():
    # Load index + metadata
    index = faiss.read_index(str(INDEX_PATH))
    meta = pd.read_parquet(META_PATH)

    # Load embedding model
    model = SentenceTransformer(MODEL_NAME)

    print("\nType a search query (or 'exit'):\n")

    while True:
        query = input("🔎 Query> ").strip()
        if query.lower() in {"exit", "quit"}:
            break

        # Embed query
        q_emb = model.encode(
            [query],
            normalize_embeddings=True
        ).astype("float32")

        # Search
        scores, idxs = index.search(q_emb, k=5)

        print("\nTop results:\n")
        for rank, (i, score) in enumerate(zip(idxs[0], scores[0]), 1):
            row = meta.iloc[i]
            print(f"{rank}. {row['title']}")
            print(f"   item_id: {row['item_id']}")
            print(f"   score: {score:.4f}\n")


if __name__ == "__main__":
    main()
