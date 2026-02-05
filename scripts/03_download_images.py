from pathlib import Path
import subprocess

import pandas as pd
from tqdm import tqdm


PRODUCTS_PATH = Path("data/processed/products_10k.parquet")
IMG_DIR = Path("data/images")

S3_ROOT = "s3://amazon-berkeley-objects/images/small/"


def aws_cp(s3_uri: str, out_path: Path) -> bool:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "aws", "s3", "cp",
        s3_uri,
        str(out_path),
        "--no-sign-request",
        "--only-show-errors",
    ]
    try:
        subprocess.run(cmd, check=True)
        return True
    except subprocess.CalledProcessError:
        return False


def main():
    df = pd.read_parquet(PRODUCTS_PATH)
    IMG_DIR.mkdir(parents=True, exist_ok=True)

    ok = 0
    fail = 0

    for _, row in tqdm(df.iterrows(), total=len(df)):
        item_id = row["item_id"]
        rel_path = row["path"]

        ext = Path(rel_path).suffix or ".jpg"
        out_file = IMG_DIR / f"{item_id}{ext}"

        if out_file.exists():
            ok += 1
            continue

        s3_uri = f"{S3_ROOT}{rel_path}"

        if aws_cp(s3_uri, out_file):
            ok += 1
        else:
            fail += 1

    print(f"Downloaded images. ok={ok}, fail={fail}")
    print(f"Images saved in: {IMG_DIR.resolve()}")


if __name__ == "__main__":
    main()
