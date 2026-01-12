import os, re, pickle
from typing import List, Tuple, Dict

import torch
import chromadb
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize as sk_normalize

from bdh import BDHTokenizer, bdh_embed_text, load_ckpt

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BASE_DIR = r"C:\Users\vaibh\Desktop\KDSH_Solution_2026\new_model"
NOV_DIR = os.path.join(BASE_DIR, "data")
DB_PATH = os.path.join(BASE_DIR, "my_novel_db")
COLLECTION_NAME = "novels_bdh"

CKPT_DIR = os.path.join(BASE_DIR, "checkpoints")
CKPT_LM = os.path.join(CKPT_DIR, "bdh_lm.pt") 

NOVELS = {
    "In Search of the Castaways": os.path.join(NOV_DIR, "In search of the castaways.txt"),
    "The Count of Monte Cristo": os.path.join(NOV_DIR, "The Count of Monte Cristo.txt"),
}

# Chunking tuned for consistency constraints
CHUNK_CHARS = 3500
OVERLAP_CHARS = 1500

OUT_INDEX_DIR = "indexes"
os.makedirs(OUT_INDEX_DIR, exist_ok=True)


def read_text(p: str) -> str:
    try:
        return open(p, "r", encoding="utf-8").read()
    except UnicodeDecodeError:
        return open(p, "r", encoding="latin-1").read()


def chunk_text(text: str, chunk_size: int, overlap: int) -> List[Tuple[int, str]]:
    out = []
    n = len(text)
    if n <= chunk_size:
        return [(0, text)]
    start = 0
    while start < n:
        end = min(n, start + chunk_size)
        out.append((start, text[start:end]))
        if end >= n:
            break
        start = max(0, end - overlap)
    return out


def slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", s.lower()).strip("")


def main():
    assert os.path.exists(CKPT_LM), f"Missing {CKPT_LM}. Run train.py pretrain first."

    model, cfg, _ = load_ckpt(CKPT_LM, DEVICE)
    tok = BDHTokenizer.from_pretrained("bdh-base")

    client = chromadb.PersistentClient(path=DB_PATH)
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    col = client.get_or_create_collection(name=COLLECTION_NAME, metadata={"hnsw:space": "cosine"})

    for book, path in NOVELS.items():
        assert os.path.exists(path), f"Missing novel: {path}"
        text = read_text(path)
        chunks = chunk_text(text, CHUNK_CHARS, OVERLAP_CHARS)

        ids, docs, metas, embs, offsets = [], [], [], [], []
        for i, (off, ch) in enumerate(chunks):
            cid = f"{slug(book)}_{i}"
            e = bdh_embed_text(model, tok, ch, DEVICE, max_len=512, windows=1, pool="mean").detach().cpu().float()
            ids.append(cid); docs.append(ch); offsets.append(off)
            metas.append({"book_name": book, "chunk_id": i, "offset": off})
            embs.append(e)

        E = torch.stack(embs, dim=0)  # (N,D)

        # Save chunk embedding cache (NO re-encoding needed later)
        torch.save(
            {"book": book, "ids": ids, "offsets": offsets, "embs": E},
            os.path.join(OUT_INDEX_DIR, f"{slug(book)}_chunks.pt"),
        )

        # TF-IDF store for reranking
        vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), max_features=200000)
        X = vec.fit_transform(docs)
        X = sk_normalize(X, axis=1)
        with open(os.path.join(OUT_INDEX_DIR, f"{slug(book)}_tfidf.pkl"), "wb") as f:
            pickle.dump({"book": book, "vectorizer": vec, "X": X, "ids": ids, "docs": docs, "offsets": offsets}, f)

        # Add to Chroma with embeddings
        # (store embeddings directly so Chroma doesn't recompute)
        col.add(ids=ids, documents=docs, metadatas=metas, embeddings=E.tolist())
        print(f"✅ Indexed {book}: chunks={len(ids)}")

    print(f"\n✅ Done. Chroma at {DB_PATH}, caches in {OUT_INDEX_DIR}/")


if __name__ == "__main__":
    main()