from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
PRODUCTS_PATH = BASE_DIR / "data" / "processed" / "products_10k.parquet"
CAPTIONS_PATH = BASE_DIR / "data" / "processed" / "captions_10k.parquet"

def main():
    prod = pd.read_parquet(PRODUCTS_PATH)
    caps = pd.read_parquet(CAPTIONS_PATH)

    merged = prod.merge(caps[["item_id", "caption"]], on="item_id", how="left")
    merged.to_parquet(PRODUCTS_PATH, index=False)

    print("Updated:", PRODUCTS_PATH)
    print("caption nulls:", merged["caption"].isna().sum(), "out of", len(merged))

if __name__ == "__main__":
    main()
