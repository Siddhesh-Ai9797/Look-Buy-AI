from pathlib import Path
import subprocess
import numpy as np
import pandas as pd
import faiss


INDEX_DIR = Path("data/index_image")
INDEX_PATH = INDEX_DIR / "faiss.index"
META_PATH = INDEX_DIR / "meta.parquet"

# This calls the stable Python 3.11 embedder
PY311 = Path(".venv311/bin/python")
EMBED_SCRIPT = Path("tools/embed_one_image.py")
TMP_EMB = Path("data/index_image/tmp_query_emb.npy")


def embed_with_py311(image_path: Path) -> np.ndarray:
    cmd = [str(PY311), str(EMBED_SCRIPT), str(image_path), str(TMP_EMB)]
    subprocess.run(cmd, check=True)
    vec = np.load(TMP_EMB).astype("float32")
    return vec.reshape(1, -1)


def main():
    index = faiss.read_index(str(INDEX_PATH))
    meta = pd.read_parquet(META_PATH)

    print("\n✅ Any-image search ready (uploads supported). Type 'exit' to quit.\n")

    while True:
        user_in = input("🖼 Image path> ").strip()
        if user_in.lower() in {"exit", "quit"}:
            break

        p = Path(user_in)
        if not p.exists():
            print("❌ File not found.\n")
            continue

        q = embed_with_py311(p)
        scores, idxs = index.search(q, k=5)

        print("\nTop 5 similar products:\n")
        for rank, (i, s) in enumerate(zip(idxs[0], scores[0]), 1):
            row = meta.iloc[int(i)]
            print(f"{rank}. {row['title']}")
            print(f"   item_id: {row['item_id']}")
            print(f"   score: {float(s):.4f}\n")


if __name__ == "__main__":
    main()
