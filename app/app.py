# app/app.py
from __future__ import annotations

from pathlib import Path
import subprocess
import time
import json
import os
import base64
import hashlib
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import faiss
import streamlit as st
from sentence_transformers import SentenceTransformer

# Optional local env loader (Phase F)
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

# ---------- Paths ----------
# Your streamlit file is app/app.py, and data/tools are at repo root.
# So BASE_DIR should be repo root (parent of app/).
BASE_DIR = Path(__file__).resolve().parents[1]

TEXT_DIR = BASE_DIR / "data" / "index_text"
TEXT_INDEX_PATH = TEXT_DIR / "faiss.index"
TEXT_META_PATH = TEXT_DIR / "meta.parquet"

IMG_DIR = BASE_DIR / "data" / "index_image"
IMG_INDEX_PATH = IMG_DIR / "faiss.index"
IMG_META_PATH = IMG_DIR / "meta.parquet"

LOCAL_IMAGES_DIR = BASE_DIR / "data" / "images"

# Stable embedder (Python 3.11) for arbitrary uploaded images
PY311 = BASE_DIR / ".venv311" / "bin" / "python"
EMBED_SCRIPT = BASE_DIR / "tools" / "embed_one_image.py"

# ✅ Fix: your repo has tools/blip_caption_one.py
BLIP_SCRIPT = BASE_DIR / "tools" / "blip_caption_one.py"

# temp files
TMP_DIR = BASE_DIR / "tmp"
TMP_DIR.mkdir(parents=True, exist_ok=True)
TMP_UPLOAD = TMP_DIR / "uploaded.png"
TMP_EMB = TMP_DIR / "uploaded_emb.npy"
TMP_CAPTION = TMP_DIR / "uploaded_caption.txt"

# Model for text queries
TEXT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# ---------- Utilities ----------
def norm(x: Optional[str]) -> str:
    return (x or "").strip().lower()

def detect_color(s: str) -> Optional[str]:
    s = norm(s)
    colors = [
        "black", "white", "red", "blue", "green", "yellow", "pink", "purple",
        "orange", "brown", "gray", "grey", "beige", "gold", "silver"
    ]
    for c in colors:
        if c in s:
            return "gray" if c == "grey" else c
    return None

def detect_category(s: str) -> Optional[str]:
    s = norm(s)
    cats = [
        "phone case", "earrings", "ring", "necklace", "kitchen tool",
        "home", "clothing", "beauty", "grocery"
    ]
    for c in cats:
        if c in s:
            return c
    return None

def ensure_files():
    missing = []
    for p in [TEXT_INDEX_PATH, TEXT_META_PATH, IMG_INDEX_PATH, IMG_META_PATH]:
        if not p.exists():
            missing.append(str(p))
    if missing:
        st.error("Missing required index files:\n\n" + "\n".join(missing))
        st.stop()

    # If you want upload-image embedding:
    if not PY311.exists():
        st.warning(f"Missing Python 3.11 env at: {PY311}\nUploaded-image embedding will be disabled.")
    if not EMBED_SCRIPT.exists():
        st.warning(f"Missing embed script at: {EMBED_SCRIPT}\nUploaded-image embedding will be disabled.")

    # Caption is optional
    if not BLIP_SCRIPT.exists():
        st.warning(f"BLIP caption script not found at: {BLIP_SCRIPT}. Auto-caption will be disabled.")

@st.cache_resource
def load_text_model():
    return SentenceTransformer(TEXT_MODEL)

@st.cache_resource
def load_indexes_and_meta():
    text_index = faiss.read_index(str(TEXT_INDEX_PATH))
    img_index = faiss.read_index(str(IMG_INDEX_PATH))
    text_meta = pd.read_parquet(TEXT_META_PATH)
    img_meta = pd.read_parquet(IMG_META_PATH)
    return text_index, img_index, text_meta, img_meta

def embed_uploaded_image(img_path: Path) -> Optional[np.ndarray]:
    if not (PY311.exists() and EMBED_SCRIPT.exists()):
        return None
    cmd = [str(PY311), str(EMBED_SCRIPT), str(img_path), str(TMP_EMB)]
    subprocess.run(cmd, check=True)
    vec = np.load(TMP_EMB).astype("float32").reshape(1, -1)
    return vec

def caption_uploaded_image(img_path: Path) -> Optional[str]:
    if not BLIP_SCRIPT.exists():
        return None
    cmd = ["python", str(BLIP_SCRIPT), str(img_path), str(TMP_CAPTION)]
    subprocess.run(cmd, check=True)
    txt = TMP_CAPTION.read_text(encoding="utf-8").strip()
    return txt or None

def embed_text(model: SentenceTransformer, query: str) -> np.ndarray:
    vec = model.encode([query], normalize_embeddings=True)
    return np.array(vec, dtype="float32")

def rrf_fuse(ranked_lists: List[List[str]], weights: List[float], k: int = 60) -> Dict[str, float]:
    scores: Dict[str, float] = {}
    for items, w in zip(ranked_lists, weights):
        for rank, item_id in enumerate(items, start=1):
            scores[item_id] = scores.get(item_id, 0.0) + (w / (k + rank))
    return scores

def choose_weights(mode: str) -> Tuple[float, float]:
    if mode == "Image-Dominant":
        return 0.75, 0.25
    if mode == "Text-Dominant":
        return 0.30, 0.70
    return 0.55, 0.45  # Hybrid

def clock_loader(container, seconds: float = 1.2):
    ticks = ["🕛", "🕐", "🕑", "🕒", "🕓", "🕔", "🕕", "🕖", "🕗", "🕘", "🕙", "🕚"]
    start = time.time()
    i = 0
    while time.time() - start < seconds:
        container.markdown(
            f"<div class='lb-loading'>Searching {ticks[i % len(ticks)]}</div>",
            unsafe_allow_html=True,
        )
        time.sleep(0.10)
        i += 1

# ---------- Phase E: attributes (real if available; else stable placeholders) ----------
def stable_hash_int(s: str, mod: int) -> int:
    h = hashlib.sha256(s.encode("utf-8")).hexdigest()
    return int(h[:8], 16) % mod

def placeholder_price(item_id: str, category: str) -> float:
    base = {
        "phone case": (9.99, 39.99),
        "earrings": (12.99, 79.99),
        "ring": (14.99, 129.99),
        "necklace": (14.99, 99.99),
        "kitchen tool": (6.99, 49.99),
        "home": (19.99, 199.99),
        "clothing": (14.99, 89.99),
        "beauty": (7.99, 69.99),
        "grocery": (1.99, 24.99),
        "other": (9.99, 59.99),
    }
    lo, hi = base.get(norm(category), base["other"])
    cents = stable_hash_int(item_id, 10_000) / 10_000
    return round(lo + (hi - lo) * cents, 2)

def placeholder_rating(item_id: str) -> float:
    x = stable_hash_int(item_id, 1300) / 1000.0
    r = 3.6 + min(1.3, x)
    return round(r, 1)

def placeholder_url(item_id: str) -> str:
    return f"https://www.amazon.com/dp/{item_id}"

def is_nan(x) -> bool:
    try:
        return bool(isinstance(x, float) and np.isnan(x))
    except Exception:
        return False

def get_attr_with_flags(row: dict) -> Tuple[float, float, str, bool, bool, bool]:
    item_id = str(row.get("item_id", ""))
    category = str(row.get("category", "") or "other")

    price = row.get("price", None)
    used_p = False
    if price is None or is_nan(price) or str(price).strip() == "":
        price = placeholder_price(item_id, category)
        used_p = True
    else:
        try:
            price = float(price)
        except Exception:
            price = placeholder_price(item_id, category)
            used_p = True

    rating = row.get("rating", None)
    used_r = False
    if rating is None or is_nan(rating) or str(rating).strip() == "":
        rating = placeholder_rating(item_id)
        used_r = True
    else:
        try:
            rating = float(rating)
        except Exception:
            rating = placeholder_rating(item_id)
            used_r = True

    url = row.get("url", None)
    used_u = False
    if not url or str(url).strip() == "":
        url = placeholder_url(item_id)
        used_u = True
    else:
        url = str(url)

    return price, rating, url, used_p, used_r, used_u

# ---------- Premium + Phase C/D (OpenAI) ----------
def _data_url_for_image(path: Path) -> str:
    ext = path.suffix.lower().replace(".", "")
    mime = "image/jpeg"
    if ext == "png":
        mime = "image/png"
    elif ext == "webp":
        mime = "image/webp"
    b64 = base64.b64encode(path.read_bytes()).decode("utf-8")
    return f"data:{mime};base64,{b64}"

def premium_extract_hints_openai(image_path: Path, user_text: str, model_name: str = "gpt-4.1") -> Dict[str, Optional[str]]:
    try:
        from openai import OpenAI
    except Exception as e:
        raise RuntimeError("OpenAI SDK not installed. Run: pip install openai") from e

    api_key = os.getenv("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set.")

    client = OpenAI(api_key=api_key)
    img_url = _data_url_for_image(image_path)

    instruction = (
        "Return ONLY valid JSON with keys: brand, category, color, short_query.\n"
        "If unknown, set value to null."
    )

    resp = client.responses.create(
        model=model_name,
        input=[
            {"role": "system", "content": instruction},
            {"role": "user", "content": [
                {"type": "input_text", "text": f"User text: {user_text or ''}"},
                {"type": "input_image", "image_url": img_url},
            ]},
        ],
    )

    raw = resp.output_text.strip()
    try:
        return json.loads(raw)
    except Exception:
        start = raw.find("{")
        end = raw.rfind("}")
        if start >= 0 and end > start:
            return json.loads(raw[start:end+1])
        raise

def rag_explain_results_openai(query_text: str, blip_caption: Optional[str], premium_hints: Optional[dict], top_rows: List[dict], model_name: str = "gpt-4.1") -> str:
    try:
        from openai import OpenAI
    except Exception as e:
        raise RuntimeError("OpenAI SDK not installed. Run: pip install openai") from e

    api_key = os.getenv("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set.")

    client = OpenAI(api_key=api_key)

    context = {
        "query_text": query_text,
        "auto_caption": blip_caption,
        "premium_hints": premium_hints or {},
        "top_results": top_rows[:10],
    }

    instruction = (
        "You are grounded. Use ONLY the provided JSON context.\n"
        "Explain why results match. If info is missing, say what is missing."
    )

    resp = client.responses.create(
        model=model_name,
        input=[
            {"role": "system", "content": instruction},
            {"role": "user", "content": f"CONTEXT:\n{json.dumps(context, ensure_ascii=False)}"},
        ],
    )
    return resp.output_text.strip()

def chat_assistant_openai(user_message: str, chat_history: List[dict], context_products: List[dict], model_name: str = "gpt-4.1") -> str:
    try:
        from openai import OpenAI
    except Exception as e:
        raise RuntimeError("OpenAI SDK not installed. Run: pip install openai") from e

    api_key = os.getenv("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set.")

    client = OpenAI(api_key=api_key)

    system = (
        "You are a conversational product assistant.\n"
        "You MUST answer using ONLY the provided product context.\n"
        "Never invent prices/ratings/links beyond what is shown."
    )

    payload_context = {"products": context_products[:15]}

    messages = [{"role": "system", "content": system}]
    messages.append({"role": "user", "content": f"PRODUCT_CONTEXT_JSON:\n{json.dumps(payload_context, ensure_ascii=False)}"})

    for m in chat_history[-12:]:
        messages.append({"role": m["role"], "content": m["content"]})
    messages.append({"role": "user", "content": user_message})

    resp = client.responses.create(model=model_name, input=messages)
    return resp.output_text.strip()

# ---------- UI ----------
def inject_css():
    st.markdown(
        """
        <style>
          .block-container {padding-top: 2.0rem; padding-bottom: 2.0rem;}
          .lb-title {font-size: 40px; font-weight: 900; letter-spacing: -0.02em;}
          .lb-sub {color: rgba(15,23,42,0.65); margin-top:-6px;}
          .lb-card{border:1px solid rgba(15,23,42,0.10);border-radius:18px;padding:14px;background:#fff;box-shadow:0 10px 28px rgba(15,23,42,0.05);}
          .lb-score{display:inline-block;padding:4px 10px;border-radius:999px;background:rgba(37,99,235,0.12);color:rgba(37,99,235,0.98);font-weight:900;font-size:12px;margin-top:6px;}
          .lb-attr{margin-top:8px;font-size:13px;color:rgba(15,23,42,0.75);}
          .lb-note{margin-top:6px;font-size:12px;color:rgba(15,23,42,0.62);}
          .lb-why{margin-top:8px;font-size:12px;color:rgba(15,23,42,0.70);background:rgba(15,23,42,0.03);border:1px solid rgba(15,23,42,0.07);padding:8px 10px;border-radius:12px;}
          .lb-loading{font-weight:900;color:rgba(37,99,235,0.95);letter-spacing:0.02em;}
          .lb-rag{margin-top:12px;padding:14px;border-radius:18px;border:1px solid rgba(37,99,235,0.20);background:rgba(37,99,235,0.04);color:rgba(15,23,42,0.86);}
        </style>
        """,
        unsafe_allow_html=True,
    )

def render_logo_svg():
    st.markdown(
        """
        <svg width="48" height="48" viewBox="0 0 64 64" fill="none" xmlns="http://www.w3.org/2000/svg" style="margin-bottom:-6px">
          <rect x="6" y="10" width="52" height="44" rx="16" fill="#2563EB" opacity="0.12"/>
          <path d="M20 26c0-4.4 3.6-8 8-8h8c4.4 0 8 3.6 8 8v16c0 4.4-3.6 8-8 8h-8c-4.4 0-8-3.6-8-8V26Z" stroke="#2563EB" stroke-width="3"/>
          <path d="M26 28h16M26 34h16M26 40h12" stroke="#2563EB" stroke-width="3" stroke-linecap="round"/>
        </svg>
        """,
        unsafe_allow_html=True,
    )

def compute_context_hash(context_products: List[dict]) -> str:
    compact = [{"item_id": p.get("item_id"), "score": p.get("score")} for p in context_products]
    raw = json.dumps(compact, sort_keys=True)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]

def minmax_normalize(vals: List[float]) -> List[float]:
    if not vals:
        return []
    lo = min(vals)
    hi = max(vals)
    if abs(hi - lo) < 1e-12:
        return [0.75 for _ in vals]
    return [(v - lo) / (hi - lo) for v in vals]

def render_chat_panel(premium_on: bool, premium_model: str):
    st.markdown("### 💬 Chat")
    if not premium_on:
        st.info("Turn on Premium (OpenAI) to enable chat.")
        return

    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []

    ctx = st.session_state.get("last_context_products", [])
    if not ctx:
        st.info("Run a search first (text or image), then chat using the results.")
        return

    for m in st.session_state["chat_history"]:
        with st.chat_message(m["role"]):
            st.write(m["content"])

    user_msg = st.chat_input("Ask: best one? under $30? similar to #2? compare top 3…")
    if not user_msg:
        return

    st.session_state["chat_history"].append({"role": "user", "content": user_msg})
    with st.chat_message("user"):
        st.write(user_msg)

    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            try:
                ans = chat_assistant_openai(
                    user_message=user_msg,
                    chat_history=st.session_state["chat_history"],
                    context_products=ctx,
                    model_name=premium_model.strip() or "gpt-4.1",
                )
            except Exception as e:
                ans = f"Error: {e}"
        st.write(ans)
    st.session_state["chat_history"].append({"role": "assistant", "content": ans})

# ✅ Use meta.parquet "path" if present; fallback to old item_id.* only if needed
def find_local_image_from_row(row_meta: dict) -> Optional[Path]:
    p = row_meta.get("path", None)
    if p:
        candidate = LOCAL_IMAGES_DIR / str(p)
        return candidate if candidate.exists() else None
    # fallback
    item_id = str(row_meta.get("item_id", ""))
    matches = list(LOCAL_IMAGES_DIR.glob(f"{item_id}.*"))
    return matches[0] if matches else None

# ---------- Main ----------
def main():
    st.set_page_config(page_title="Look&Buy", layout="wide")
    inject_css()

    ensure_files()
    text_index, img_index, text_meta, img_meta = load_indexes_and_meta()
    model = load_text_model()

    # Header
    cA, cB = st.columns([1, 7], gap="small")
    with cA:
        render_logo_svg()
    with cB:
        st.markdown("<div class='lb-title'>Look&Buy</div>", unsafe_allow_html=True)
        st.markdown("<div class='lb-sub'>Image + Text Search · Hybrid Retrieval · RAG · Chat</div>", unsafe_allow_html=True)

    st.divider()

    # Sidebar controls
    with st.sidebar:
        st.markdown("### ⚙️ Controls")
        mode = st.selectbox("Search Mode", ["Hybrid", "Image-Dominant", "Text-Dominant"], index=0)

        top_k = st.slider("Results to show", min_value=6, max_value=40, value=12, step=2)
        retrieve_k = st.slider("Candidates retrieved (FAISS k)", min_value=50, max_value=300, value=140, step=10)

        st.markdown("### 🎛 Filters")
        category_filter = st.text_input("Category contains", value="")
        color_filter = st.text_input("Color contains", value="")
        brand_filter = st.text_input("Brand contains", value="")

        st.markdown("### 🧠 Premium (OpenAI)")
        premium_on = st.toggle("Enable Premium", value=False)
        premium_model = st.text_input("Model name", value="gpt-4.1", disabled=not premium_on)
        phase_c_rag = st.toggle("Phase C: RAG explanation", value=True, disabled=not premium_on)
        phase_d_chat = st.toggle("Phase D: Chat assistant", value=True, disabled=not premium_on)

        st.markdown("### ✅ Chat correctness")
        reset_chat_on_new_results = st.toggle("Reset chat when results change", value=True)

    # Inputs
    left, right = st.columns([1.15, 1.0], gap="large")
    with left:
        st.markdown("### 🔎 Search input")
        uploaded = st.file_uploader("Upload an image (png/jpg/webp)", type=["png", "jpg", "jpeg", "webp"])

        # ✅ Fix “Search similar” reliability: consume after rendering input
        prefill = st.session_state.get("prefill_query", "")
        user_text = st.text_input(
            "Optional text hint (e.g., 'black nike hoodie', 'gold ring', 'iphone case')",
            value=prefill,
            key="query_text_input",
        )
        if prefill:
            # clear only AFTER the input is rendered
            st.session_state["prefill_query"] = ""

        run = st.button("🔍 DISCOVER", use_container_width=True)

        if uploaded is not None:
            TMP_UPLOAD.write_bytes(uploaded.read())
            st.image(str(TMP_UPLOAD), caption="Uploaded image", use_container_width=True)

    # Right column: chat panel
    with right:
        if phase_d_chat:
            render_chat_panel(premium_on=premium_on, premium_model=premium_model)

    # Search execution
    if not run:
        return

    loader = st.empty()
    clock_loader(loader, seconds=1.0)
    loader.empty()

    # Build final query text
    final_text_query = (user_text or "").strip()
    auto_caption = None
    premium_hints = None

    if uploaded is not None:
        try:
            auto_caption = caption_uploaded_image(TMP_UPLOAD)
        except Exception:
            auto_caption = None

    if premium_on and uploaded is not None:
        try:
            premium_hints = premium_extract_hints_openai(
                image_path=TMP_UPLOAD,
                user_text=final_text_query,
                model_name=premium_model.strip() or "gpt-4.1",
            )
        except Exception:
            premium_hints = None

    hint_bits = []
    if auto_caption:
        hint_bits.append(auto_caption)

    if premium_hints:
        for k in ["short_query", "brand", "category", "color"]:
            if premium_hints.get(k):
                hint_bits.append(str(premium_hints[k]))

    if final_text_query:
        hint_bits.insert(0, final_text_query)

    combined_query = " ".join([b for b in hint_bits if b and str(b).strip()]).strip()
    if not combined_query and uploaded is None:
        st.warning("Upload an image and/or enter text.")
        return

    inferred_category = detect_category(combined_query) if not category_filter else category_filter
    inferred_color = detect_color(combined_query) if not color_filter else color_filter
    inferred_brand = brand_filter if brand_filter else (premium_hints.get("brand") if premium_hints else "")

    w_img, w_txt = choose_weights(mode)

    img_ids_ranked: List[str] = []
    img_rank_pos: Dict[str, int] = {}
    if uploaded is not None:
        img_vec = embed_uploaded_image(TMP_UPLOAD)
        if img_vec is not None:
            _, I_img = img_index.search(img_vec, int(retrieve_k))
            img_ids_ranked = [str(img_meta.iloc[i]["item_id"]) for i in I_img[0] if i >= 0]
            img_rank_pos = {item_id: r for r, item_id in enumerate(img_ids_ranked, start=1)}
        else:
            st.info("Uploaded-image embedding not available (missing .venv311 or embed_one_image.py). Running text-only.")

    txt_ids_ranked: List[str] = []
    txt_rank_pos: Dict[str, int] = {}
    if combined_query:
        txt_vec = embed_text(model, combined_query)
        _, I_txt = text_index.search(txt_vec, int(retrieve_k))
        txt_ids_ranked = [str(text_meta.iloc[i]["item_id"]) for i in I_txt[0] if i >= 0]
        txt_rank_pos = {item_id: r for r, item_id in enumerate(txt_ids_ranked, start=1)}

    if not img_ids_ranked and not txt_ids_ranked:
        st.warning("No candidates found. Try a different image or add a text hint.")
        return

    fused = rrf_fuse([img_ids_ranked, txt_ids_ranked], [w_img, w_txt], k=60)
    fused_sorted = sorted(fused.items(), key=lambda x: x[1], reverse=True)

    def row_from_meta(item_id: str) -> dict:
        rows = text_meta[text_meta["item_id"].astype(str) == str(item_id)]
        if len(rows) > 0:
            return rows.iloc[0].to_dict()
        rows2 = img_meta[img_meta["item_id"].astype(str) == str(item_id)]
        if len(rows2) > 0:
            return rows2.iloc[0].to_dict()
        return {"item_id": item_id, "title": item_id}

    # ✅ Filter fallback: if meta lacks category/color/brand, don't kill results
    def safe_contains(val: str, needle: str) -> bool:
        if not needle:
            return True
        return norm(needle) in norm(val)

    filtered_items: List[Tuple[str, float]] = []
    for item_id, score in fused_sorted:
        rowm = row_from_meta(item_id)
        ok = True
        ok = ok and safe_contains(str(rowm.get("category", "")), inferred_category or "")
        ok = ok and safe_contains(str(rowm.get("color", "")), inferred_color or "")
        ok = ok and safe_contains(str(rowm.get("brand", "")), inferred_brand or "")
        if ok:
            filtered_items.append((item_id, score))
        if len(filtered_items) >= int(top_k):
            break

    if not filtered_items:
        st.warning("No matches after filters. Try clearing filters.")
        return

    scores = [float(s) for _, s in filtered_items]
    rel_norm = minmax_normalize(scores)

    context_products = []
    for idx, ((item_id, fused_score), rel) in enumerate(zip(filtered_items, rel_norm), start=1):
        rowm = row_from_meta(item_id)

        title = str(rowm.get("title", item_id))
        brand = str(rowm.get("brand", "") or "")
        category = str(rowm.get("category", "") or "")
        color = str(rowm.get("color", "") or "")
        caption = str(rowm.get("caption", "") or "")

        price, rating, url, used_p, used_r, used_u = get_attr_with_flags(rowm)

        img_r = img_rank_pos.get(str(item_id), None)
        txt_r = txt_rank_pos.get(str(item_id), None)

        context_products.append(
            {
                "rank": str(idx),
                "item_id": str(item_id),
                "title": title,
                "brand": brand,
                "category": category,
                "color": color,
                "caption": caption,
                "price_usd": f"{float(price):.2f}",
                "rating": f"{float(rating):.1f}",
                "url": str(url),
                "score": f"{float(fused_score):.6f}",
                "relevance": float(rel),
                "img_rank": img_r,
                "txt_rank": txt_r,
                "demo_price": bool(used_p),
                "demo_rating": bool(used_r),
                "demo_url": bool(used_u),
                "path": rowm.get("path", None),
            }
        )

    new_hash = compute_context_hash(context_products)
    prev_hash = st.session_state.get("last_context_hash")
    st.session_state["last_context_products"] = context_products
    st.session_state["last_query_summary"] = (combined_query or "Search").strip()[:80]
    st.session_state["last_context_hash"] = new_hash

    if reset_chat_on_new_results and prev_hash and prev_hash != new_hash:
        st.session_state["chat_history"] = []

    st.markdown("### 🧾 Results")
    cols = st.columns(2, gap="large")

    for idx, row in enumerate(context_products, start=1):
        item_id = row["item_id"]
        title = row["title"]
        rel_badge = float(row["relevance"])

        # use path-based image loading
        img_path = None
        if row.get("path"):
            cand = LOCAL_IMAGES_DIR / str(row["path"])
            img_path = cand if cand.exists() else None
        if img_path is None:
            # fallback legacy
            img_path = find_local_image_from_row({"item_id": item_id})

        with cols[(idx - 1) % 2]:
            st.markdown("<div class='lb-card'>", unsafe_allow_html=True)
            c1, c2 = st.columns([1, 2], gap="medium")

            with c1:
                if img_path and img_path.exists():
                    st.image(str(img_path), use_container_width=True)
                else:
                    st.write("No image")

            with c2:
                st.markdown(f"**{idx}. {title}**")
                meta_line = " · ".join([x for x in [row.get("brand"), row.get("category"), row.get("color")] if x and x.strip()])
                if meta_line:
                    st.caption(meta_line)

                st.markdown(f"<span class='lb-score'>Relevance: {rel_badge:.2f}</span>", unsafe_allow_html=True)

                st.markdown(
                    f"<div class='lb-attr'>⭐ {row['rating']} · <b>${row['price_usd']}</b></div>",
                    unsafe_allow_html=True,
                )

                demo_bits = []
                if row.get("demo_rating"):
                    demo_bits.append("rating")
                if row.get("demo_price"):
                    demo_bits.append("price")
                if row.get("demo_url"):
                    demo_bits.append("link")
                if demo_bits:
                    st.markdown(
                        f"<div class='lb-note'>Demo values used for: <b>{', '.join(demo_bits)}</b></div>",
                        unsafe_allow_html=True,
                    )

                cap = row.get("caption")
                if cap:
                    st.write(f"_{cap}_")

                why_parts = []
                if row.get("img_rank") is not None:
                    why_parts.append(f"Image rank: #{row.get('img_rank')}")
                if row.get("txt_rank") is not None:
                    why_parts.append(f"Text rank: #{row.get('txt_rank')}")
                why_parts.append(f"Fused score: {row.get('score')}")
                st.markdown(
                    f"<div class='lb-why'><b>Why matched</b><br/>{' · '.join(why_parts)}</div>",
                    unsafe_allow_html=True,
                )

                st.write(f"Item ID: `{item_id}`")

                b1, b2 = st.columns(2, gap="small")
                with b1:
                    st.link_button("Open product", row["url"], use_container_width=True)
                with b2:
                    if st.button("Search similar", key=f"sim_{item_id}", use_container_width=True):
                        seed = " ".join([row.get("brand", ""), row.get("title", ""), row.get("category", "")]).strip()
                        st.session_state["prefill_query"] = (seed or row.get("title", ""))[:120]
                        st.rerun()

            st.markdown("</div>", unsafe_allow_html=True)

    if phase_c_rag and premium_on:
        st.markdown("### 🧠 Why these results?")
        with st.spinner("Generating explanation (RAG)…"):
            try:
                explanation = rag_explain_results_openai(
                    query_text=combined_query,
                    blip_caption=auto_caption,
                    premium_hints=premium_hints,
                    top_rows=context_products[:10],
                    model_name=premium_model.strip() or "gpt-4.1",
                )
                st.markdown(f"<div class='lb-rag'>{explanation}</div>", unsafe_allow_html=True)
            except Exception as e:
                st.error(f"RAG failed: {e}")

if __name__ == "__main__":
    main()
