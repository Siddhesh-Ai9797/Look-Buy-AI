from pathlib import Path
import re
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
IN_PATH = BASE_DIR / "data" / "processed" / "products_10k.parquet"
OUT_PATH = BASE_DIR / "data" / "processed" / "products_10k_enriched.parquet"


COLORS = {
    "black","white","silver","gold","gray","grey","blue","navy","red","green","yellow","orange",
    "pink","purple","violet","brown","beige","tan","cream","clear","transparent","rose","rose gold"
}

# super lightweight category rules (good enough for submission; we’ll refine later)
CATEGORY_RULES = [
    ("phone case", ["case", "iphone", "samsung", "galaxy", "pixel", "cover"]),
    ("earrings", ["earring", "stud", "hoop", "piercing"]),
    ("necklace", ["necklace", "pendant", "chain"]),
    ("ring", ["ring", "band"]),
    ("watch", ["watch", "smartwatch"]),
    ("laptop accessory", ["laptop", "macbook", "notebook", "keyboard", "mouse", "trackpad"]),
    ("kitchen tool", ["kitchen", "peeler", "spatula", "knife", "cookware", "pan", "pot", "utensil", "mop"]),
    ("grocery", ["organic", "snack", "scone", "tofu", "chicken", "food", "drink", "beverage"]),
    ("beauty", ["shampoo", "conditioner", "serum", "lotion", "cream", "makeup", "perfume"]),
    ("home", ["chair", "table", "sofa", "lamp", "bedding", "pillow", "curtain"]),
    ("tools", ["cutter", "pliers", "wrench", "screwdriver", "drill"]),
    ("clothing", ["shirt", "hoodie", "jacket", "jeans", "dress", "pants", "shoes", "sneaker"]),
]


STOPWORDS = {
    "the","a","an","and","or","with","for","of","to","in","on","by","from","this","that","these","those",
    "new","set","pack","pcs","piece","pieces","count","inch","inches","cm","mm","oz","lbs","lb"
}


def clean_text(s: str) -> str:
    s = (s or "").lower()
    s = re.sub(r"[^a-z0-9\s\-]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def extract_color(text: str):
    t = clean_text(text)
    # check multiword first
    if "rose gold" in t:
        return "rose gold"
    for c in COLORS:
        if re.search(rf"\b{re.escape(c)}\b", t):
            return c
    return None


def guess_category(text: str):
    t = clean_text(text)
    for cat, keys in CATEGORY_RULES:
        for k in keys:
            if re.search(rf"\b{re.escape(k)}\b", t):
                return cat
    return "other"


def extract_brand(title: str):
    # Simple heuristic: brand often appears at the start like "AmazonBasics", "IGI", etc.
    # We'll take first token if it's Capitalized/alpha-ish OR "AmazonBasics"/"Amazon"/etc.
    title = (title or "").strip()
    if not title:
        return None

    first = title.split()[0]
    first_clean = re.sub(r"[^A-Za-z0-9\-&]", "", first)

    if len(first_clean) < 2:
        return None

    common = {"amazonbasics", "amazon", "igd", "igi", "rivet", "essentials", "365"}
    if first_clean.lower() in common:
        return first_clean

    # If it has letters and starts with uppercase, likely brand-like
    if re.match(r"^[A-Z][A-Za-z0-9\-&]+$", first_clean):
        return first_clean

    return None


def extract_keywords(text: str):
    t = clean_text(text)
    tokens = [w for w in t.split() if w not in STOPWORDS and len(w) >= 3]
    # keep unique but stable order
    seen = set()
    out = []
    for w in tokens:
        if w not in seen:
            out.append(w)
            seen.add(w)
    return out[:40]


def main():
    if not IN_PATH.exists():
        raise FileNotFoundError(IN_PATH)

    df = pd.read_parquet(IN_PATH)

    # Ensure caption exists
    if "caption" not in df.columns:
        df["caption"] = ""

    brands, cats, cols, keywords, text_index = [], [], [], [], []

    for _, row in df.iterrows():
        title = str(row.get("title", "") or "")
        caption = str(row.get("caption", "") or "")
        joined = f"{title} {caption}"

        b = extract_brand(title)
        c = extract_color(joined)
        cat = guess_category(joined)
        kw = extract_keywords(joined)

        brands.append(b)
        cols.append(c)
        cats.append(cat)
        keywords.append(kw)

        parts = [title, caption]
        if b: parts.append(f"brand {b}")
        if cat: parts.append(f"category {cat}")
        if c: parts.append(f"color {c}")
        parts.append("keywords " + " ".join(kw))
        text_index.append(" | ".join([p for p in parts if p]))

    df["brand"] = brands
    df["category"] = cats
    df["color"] = cols
    df["keywords"] = keywords
    df["text_for_index"] = text_index

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT_PATH, index=False)

    print("Saved:", OUT_PATH)
    print("Rows:", len(df))
    print("Category distribution (top 10):")
    print(df["category"].value_counts().head(10))
    print("Brand nulls:", int(df["brand"].isna().sum()))
    print("Color nulls:", int(df["color"].isna().sum()))


if __name__ == "__main__":
    main()
