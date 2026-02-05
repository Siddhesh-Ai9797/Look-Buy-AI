import gzip
import json
from pathlib import Path

import pandas as pd
from tqdm import tqdm


RAW_LISTINGS_DIR = Path("data/raw/listings")
IMAGE_META_PATH = Path("data/raw/images_meta/images.csv.gz")
OUT_PATH = Path("data/processed/products_10k.parquet")

TARGET_N = 10_000


def pick_en_us_text(multilang_field):
    """
    ABO fields like item_name are lists of dicts:
      [{"language_tag":"en_US","value":"..."}, ...]
    Return the en_US value if present.
    """
    if not isinstance(multilang_field, list):
        return None
    for item in multilang_field:
        if isinstance(item, dict) and item.get("language_tag") == "en_US":
            return item.get("value")
    return None


def iter_jsonl_gz(path: Path):
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def main():
    # Image metadata: image_id -> relative path (used with s3://.../images/small/)
    image_meta = pd.read_csv(IMAGE_META_PATH)
    # Keep only what we need
    image_meta = image_meta[["image_id", "path"]]

    rows = []
    listing_files = sorted(RAW_LISTINGS_DIR.glob("listings_*.json.gz"))

    if not listing_files:
        raise FileNotFoundError("No listings_*.json.gz files found. Did you run 01_download_metadata.sh?")

    # Read shards until we get 10k items that have:
    # - en_US title
    # - main_image_id that exists in images.csv.gz
    for lf in listing_files:
        for obj in iter_jsonl_gz(lf):
            item_id = obj.get("item_id")
            main_image_id = obj.get("main_image_id")
            title_en = pick_en_us_text(obj.get("item_name"))

            if not item_id or not main_image_id or not title_en:
                continue

            rows.append(
                {
                    "item_id": item_id,
                    "title": title_en,
                    "main_image_id": main_image_id,
                }
            )

        # Quick stop if we already collected enough raw candidates
        if len(rows) >= TARGET_N * 2:
            break

    df = pd.DataFrame(rows).drop_duplicates(subset=["item_id"])

    # Join to get image relative path (e.g., "000000/....jpg")
    df = df.merge(image_meta, left_on="main_image_id", right_on="image_id", how="inner")

    # Keep final columns
    df = df[["item_id", "title", "main_image_id", "path"]].drop_duplicates(subset=["item_id"])

    # Sample exactly 10k (or max available)
    if len(df) >= TARGET_N:
        df = df.sample(TARGET_N, random_state=42).reset_index(drop=True)
    else:
        df = df.reset_index(drop=True)
        print(f"Warning: Only found {len(df)} eligible products so far. You can download more listing shards later.")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT_PATH, index=False)

    print("Saved:", OUT_PATH)
    print("Rows:", len(df))
    print(df.head(5))


if __name__ == "__main__":
    main()
