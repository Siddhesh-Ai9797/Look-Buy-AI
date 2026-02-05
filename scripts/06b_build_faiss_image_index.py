from pathlib import Path
import numpy as np
import faiss

OUT_DIR = Path("data/index_image")
EMBS_PATH = OUT_DIR / "clip_embs.npy"
INDEX_PATH = OUT_DIR / "faiss.index"


def main():
    if not EMBS_PATH.exists():
        raise FileNotFoundError(
            f"Missing {EMBS_PATH}. Run scripts/06a_embed_images_clip.py first."
        )

    embs = np.load(EMBS_PATH).astype("float32")
    if embs.ndim != 2:
        raise ValueError(f"Expected a 2D embeddings array, got shape {embs.shape}")

    dim = embs.shape[1]

    # Cosine similarity = inner product if vectors are normalized (we normalized in 06a)
    index = faiss.IndexFlatIP(dim)
    index.add(embs)

    faiss.write_index(index, str(INDEX_PATH))

    print("✅ FAISS image index saved:", INDEX_PATH.resolve())
    print("Embeddings:", embs.shape)


if __name__ == "__main__":
    main()
