from pathlib import Path
import numpy as np
import pandas as pd
import faiss


INDEX_DIR = Path("data/index_image")
INDEX_PATH = INDEX_DIR / "faiss.index"
META_PATH = INDEX_DIR / "meta.parquet"
EMBS_PATH = INDEX_DIR / "clip_embs.npy"


def main():
    if not INDEX_PATH.exists():
        raise FileNotFoundError(f"Missing {INDEX_PATH}. Run 06b first.")
    if not META_PATH.exists():
        raise FileNotFoundError(f"Missing {META_PATH}. Run 06a first.")
    if not EMBS_PATH.exists():
        raise FileNotFoundError(f"Missing {EMBS_PATH}. Run 06a first.")

    # Load FAISS index + metadata + embeddings (saved from 06a)
    index = faiss.read_index(str(INDEX_PATH))
    meta = pd.read_parquet(META_PATH)

    # Use mmap to avoid loading whole file into RAM if not needed
    embs = np.load(EMBS_PATH, mmap_mode="r")  # shape: (N, 512)

    # Build item_id -> row_index lookup
    item_to_row = {item_id: i for i, item_id in enumerate(meta["item_id"].tolist())}

    print("\n✅ Image search ready (using precomputed CLIP embeddings).")
    print("👉 IMPORTANT: For this test, use an image from data/images/ (downloaded ABO images).")
    print("Type 'exit' to quit.\n")

    while True:
        user_in = input("🖼 Image path (from data/images)> ").strip()
        if user_in.lower() in {"exit", "quit"}:
            break

        p = Path(user_in)
        if not p.exists():
            print("❌ File not found. Paste a valid path.\n")
            continue

        # Expect filename like data/images/<ITEM_ID>.jpg
        item_id = p.stem
        if item_id not in item_to_row:
            print("❌ This image filename doesn't match an item_id in meta.parquet.")
            print("   Use an image from data/images/ where the filename is the item_id.\n")
            continue

        q_idx = item_to_row[item_id]
        q = np.array([embs[q_idx]], dtype="float32")  # shape (1, 512)

        scores, idxs = index.search(q, k=6)  # 6 because top-1 will be itself

        print("\nTop similar products:\n")
        shown = 0
        for i, s in zip(idxs[0], scores[0]):
            i = int(i)
            if i == q_idx:
                continue  # skip itself
            row = meta.iloc[i]
            shown += 1
            print(f"{shown}. {row['title']}")
            print(f"   item_id: {row['item_id']}")
            print(f"   score: {float(s):.4f}\n")
            if shown >= 5:
                break


if __name__ == "__main__":
    main()
