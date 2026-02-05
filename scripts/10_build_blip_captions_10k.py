from pathlib import Path
import pandas as pd
from PIL import Image
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
from tqdm import tqdm


BASE_DIR = Path(__file__).resolve().parent.parent
PRODUCTS_PATH = BASE_DIR / "data" / "processed" / "products_10k.parquet"
IMAGES_DIR = BASE_DIR / "data" / "images"
OUT_CAPTIONS = BASE_DIR / "data" / "processed" / "captions_10k.parquet"


def pick_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def safe_open_image(p: Path):
    try:
        return Image.open(p).convert("RGB")
    except Exception:
        return None


def main():
    if not PRODUCTS_PATH.exists():
        raise FileNotFoundError(f"Missing: {PRODUCTS_PATH}")

    df = pd.read_parquet(PRODUCTS_PATH)
    if "item_id" not in df.columns:
        raise ValueError("products_10k.parquet must contain item_id")

    device = pick_device()
    print(f"Using device: {device}")

    model_id = "Salesforce/blip-image-captioning-base"
    processor = BlipProcessor.from_pretrained(model_id)
    model = BlipForConditionalGeneration.from_pretrained(model_id).to(device)
    model.eval()

    rows = []
    for item_id in tqdm(df["item_id"].tolist(), total=len(df)):
        # try jpg/png/jpeg/webp
        img_path = None
        for ext in [".jpg", ".jpeg", ".png", ".webp"]:
            p = IMAGES_DIR / f"{item_id}{ext}"
            if p.exists():
                img_path = p
                break

        if img_path is None:
            rows.append({"item_id": item_id, "caption": None, "ok": False})
            continue

        image = safe_open_image(img_path)
        if image is None:
            rows.append({"item_id": item_id, "caption": None, "ok": False})
            continue

        inputs = processor(images=image, return_tensors="pt").to(device)

        try:
            with torch.no_grad():
                out = model.generate(
                    **inputs,
                    max_new_tokens=30,
                    num_beams=5,
                    do_sample=False
                )
            caption = processor.decode(out[0], skip_special_tokens=True).strip()
            rows.append({"item_id": item_id, "caption": caption, "ok": True})
        except Exception:
            rows.append({"item_id": item_id, "caption": None, "ok": False})

    cap_df = pd.DataFrame(rows)
    OUT_CAPTIONS.parent.mkdir(parents=True, exist_ok=True)
    cap_df.to_parquet(OUT_CAPTIONS, index=False)

    print(f"Saved captions -> {OUT_CAPTIONS}")
    print(cap_df["ok"].value_counts(dropna=False))


if __name__ == "__main__":
    main()
