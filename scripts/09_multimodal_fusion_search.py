from pathlib import Path
import subprocess
import numpy as np
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer

TEXT_DIR = Path("data/index_text")
TEXT_INDEX_PATH = TEXT_DIR / "faiss.index"
TEXT_META_PATH = TEXT_DIR / "meta.parquet"

IMG_DIR = Path("data/index_image")
IMG_INDEX_PATH = IMG_DIR / "faiss.index"
IMG_META_PATH = IMG_DIR / "meta.parquet"

PY311 = Path(".venv311/bin/python")
EMBED_SCRIPT = Path("tools/embed_one_image.py")
TMP_EMB = IMG_DIR / "tmp_query_emb.npy"

TEXT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

GENERIC_TEXT = {
    "", "similar", "similar products", "like this", "same", "same product", "same products",
    "find similar", "show similar", "recommend", "recommendations"
}

# Very light keyword “category intent” detector (we'll improve later using real categories)
STRONG_CATEGORY_WORDS = {
    "kitchen", "cook", "cooking", "knife", "spoon", "pan", "pot",
    "shoe", "sneaker", "boots", "sandals",
    "chair", "sofa", "table", "desk",
    "phone case", "iphone", "ipad",
    "tool", "hammer", "wrench", "pliers",
    "earrings", "necklace", "ring", "jewelry", "jewellery",
    "gum", "nicotine", "vitamins", "supplement"
}


def run_py311_embed(image_path: Path) -> np.ndarray:
    cmd = [str(PY311), str(EMBED_SCRIPT), str(image_path), str(TMP_EMB)]
    subprocess.run(cmd, check=True)
    vec = np.load(TMP_EMB).astype("float32")
    return vec.reshape(1, -1)


def embed_text(model: SentenceTransformer, query: str) -> np.ndarray:
    vec = model.encode([query], normalize_embeddings=True)
    return np.array(vec, dtype="float32")


def rrf_fuse(ranked_lists: list[list[str]], weights: list[float], k: int = 60) -> dict[str, float]:
    scores: dict[str, float] = {}
    for items, w in zip(ranked_lists, weights):
        for rank, item_id in enumerate(items, start=1):
            scores[item_id] = scores.get(item_id, 0.0) + (w / (k + rank))
    return scores


def has_strong_category_intent(q: str) -> bool:
    ql = q.lower()
    for w in STRONG_CATEGORY_WORDS:
        if w in ql:
            return True
    return False


def choose_weights(mode: str, text_query: str) -> tuple[float, float]:
    q = text_query.strip().lower()

    # If user explicitly wants "refine this", image dominates
    if mode == "1":
        return 0.75, 0.25

    # If user wants "find by text category", text dominates
    if mode == "2":
        return 0.30, 0.70

    # Auto mode:
    if q in GENERIC_TEXT:
        return 0.70, 0.30
    if has_strong_category_intent(q):
        return 0.35, 0.65
    return 0.50, 0.50


def main():
    text_index = faiss.read_index(str(TEXT_INDEX_PATH))
    img_index = faiss.read_index(str(IMG_INDEX_PATH))

    text_meta = pd.read_parquet(TEXT_META_PATH)
    img_meta = pd.read_parquet(IMG_META_PATH)

    txt_model = SentenceTransformer(TEXT_MODEL)

    text_ids = set(text_meta["item_id"].tolist())
    img_ids = set(img_meta["item_id"].tolist())

    print("\n✅ Multimodal Fusion Search (Image + Text)")
    print("Modes:")
    print("  1 = Refine this item (image dominates)")
    print("  2 = Find by text category (text dominates)")
    print("  Enter = Auto\n")
    print("Rules:")
    print(" - Press Enter on Image path => text-only search")
    print(" - Press Enter on Text query => image-only search")
    print(" - Use 'exit' to quit\n")

    while True:
        img_path_in = input("🖼 Image path (or Enter)> ").strip()
        if img_path_in.lower() in {"exit", "quit"}:
            break

        query = input("�� Text query (or Enter)> ").strip()
        if query.lower() in {"exit", "quit"}:
            break

        mode = input("🧠 Mode (1/2/Enter=Auto)> ").strip()
        if mode.lower() in {"exit", "quit"}:
            break
        if mode not in {"", "1", "2"}:
            mode = ""

        use_image = bool(img_path_in)
        use_text = bool(query)

        if not use_image and not use_text:
            print("❌ Provide at least an image path or a text query.\n")
            continue

        ranked_lists = []
        weights = []

        if use_image:
            img_path = Path(img_path_in)
            if not img_path.exists():
                print("❌ Image file not found.\n")
                continue

            q_img = run_py311_embed(img_path)
            _, img_idxs = img_index.search(q_img, k=80)
            img_ranked_ids = [img_meta.iloc[int(i)]["item_id"] for i in img_idxs[0]]
            ranked_lists.append(img_ranked_ids)

        if use_text:
            q_txt = embed_text(txt_model, query)
            _, txt_idxs = text_index.search(q_txt, k=80)
            txt_ranked_ids = [text_meta.iloc[int(i)]["item_id"] for i in txt_idxs[0]]
            ranked_lists.append(txt_ranked_ids)

        if use_image and use_text:
            w_img, w_txt = choose_weights(mode, query)
            weights = [w_img, w_txt]
        else:
            weights = [1.0]

        fused = rrf_fuse(ranked_lists, weights, k=60)
        top = sorted(fused.items(), key=lambda x: x[1], reverse=True)[:10]

        if use_image and use_text:
            print(f"\nWeights used -> image: {weights[0]:.2f}, text: {weights[1]:.2f}\n")
        else:
            print("\nWeights used -> single modality\n")

        print("⭐ Top results:\n")
        for rank, (item_id, score) in enumerate(top, 1):
            if item_id in text_ids:
                title = text_meta.loc[text_meta["item_id"] == item_id, "title"].values[0]
            else:
                title = img_meta.loc[img_meta["item_id"] == item_id, "title"].values[0]

            print(f"{rank}. {title}")
            print(f"   item_id: {item_id}")
            print(f"   fused_score: {score:.6f}\n")

        print("----\n")


if __name__ == "__main__":
    main()
