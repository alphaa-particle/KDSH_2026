import os, re, pickle
from typing import List, Dict, Tuple

import torch
import pandas as pd
import chromadb
from sklearn.preprocessing import normalize as sk_normalize

from bdh import BDHTokenizer, bdh_embed_text, load_ckpt
from train import TrackBHead 
from paths import DATA_DIR, INDEX_DIR, CKPT_DIR, DB_PATH


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- PATHS ---
# DB_PATH = r"C:\Users\vaibh\Desktop\KDSH_Solution_2026\new_model\my_novel_db"
# COLLECTION_NAME = "novels_bdh"

# CKPT_DIR = r"C:\Users\vaibh\Desktop\KDSH_Solution_2026\new_model\checkpoints"
# CKPT_LM = os.path.join(CKPT_DIR, "bdh_lm.pt")
# ENSEMBLE_DIR = os.path.join(CKPT_DIR, "ensemble")

# INDEX_DIR = "indexes"
# TEST_CSV = os.path.join("data", "test.csv")
# OUT_CSV = "submission.csv"
COLLECTION_NAME = "novels_bdh"
CKPT_LM = os.path.join(CKPT_DIR, "bdh_lm.pt")
ENSEMBLE_DIR = os.path.join(CKPT_DIR, "ensemble")


TEST_CSV = os.path.join(DATA_DIR, "test.csv")
OUT_CSV = "results.csv"

# Configuration
WIDE_N = 80
FINAL_K = 16
MIN_THRESHOLD = 0.40  # Strictness floor

def slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", s.lower()).strip("")

def split_claims(text: str, max_claims: int = 6) -> List[str]:
    import re
    t = str(text or "").strip()
    if not t:
        return []
    parts = re.split(r"(?<=[\.\!\?])\s+", t)
    out = [p.strip() for p in parts if len(p.strip()) >= 5]
    return out[:max_claims] if out else [t[:300]]

def load_tfidf(book: str):
    path = os.path.join(INDEX_DIR, f"{slug(book)}_tfidf.pkl")
    with open(path, "rb") as f:
        return pickle.load(f)

def load_chunks(book: str):
    path = os.path.join(INDEX_DIR, f"{slug(book)}_chunks.pt")
    return torch.load(path, map_location="cpu") 

def load_ensemble(book: str, d: int):
    models = []
    thresholds = []
    safe_book_prefix = book.replace(" ", "_")
    
    print(f"Loading ensemble for: {book} (looking for prefix '{safe_book_prefix}')")
    
    if os.path.exists(ENSEMBLE_DIR):
        for fn in os.listdir(ENSEMBLE_DIR):
            if fn.endswith(".pt") and fn.startswith(safe_book_prefix):
                path = os.path.join(ENSEMBLE_DIR, fn)
                try:
                    obj = torch.load(path, map_location=DEVICE)
                    m = TrackBHead(d=d, dropout=float(obj.get("dropout", 0.0))).to(DEVICE)
                    m.load_state_dict(obj["head_state"], strict=True)
                    m.eval()
                    models.append(m)
                    thresholds.append(float(obj["threshold"]))
                except Exception as e:
                    print(f"Error loading {fn}: {e}")

    if not models:
        print(f"⚠️ WARNING: No models found for {book}! Defaulting to dummy threshold 0.5")
        return [], 0.5

    avg_thr = sum(thresholds) / len(thresholds)
    final_thr = max(avg_thr, MIN_THRESHOLD)
    
    print(f"   > Loaded {len(models)} models. Avg Thr: {avg_thr:.3f} -> Clamped Thr: {final_thr:.3f}")
    return models, final_thr

def make_query_emb(lm, tok, row) -> torch.Tensor:
    char = str(row["char"])
    caption = str(row.get("caption", ""))
    content = str(row.get("content", ""))
    base = f"CHAR: {char}\nCAPTION: {caption}\nBACKSTORY: {content}"
    
    q = bdh_embed_text(lm, tok, base, DEVICE, max_len=512, windows=3, pool="mean")
    claims = split_claims(content, max_claims=6)
    if claims:
        ce = []
        for c in claims[:3]:
            ce.append(bdh_embed_text(lm, tok, f"CHAR: {char}\nCLAIM: {c}", DEVICE, max_len=256, windows=1))
        q = torch.nn.functional.normalize((q + torch.stack(ce).mean(dim=0)) / 2, dim=-1)
    
    return q

def multi_queries(row) -> List[str]:
    char = str(row["char"])
    caption = str(row.get("caption", ""))
    content = str(row.get("content", ""))
    base = f"CHAR: {char}\nCAPTION: {caption}\nBACKSTORY: {content}"
    qs = [base, f"CHAR: {char}\nCAPTION: {caption}"]
    for c in split_claims(content, 6)[:3]:
        qs.append(f"CHAR: {char}\nCLAIM: {c}")
    return qs

def main():
    lm, cfg, _ = load_ckpt(CKPT_LM, DEVICE)
    tok = BDHTokenizer.from_pretrained("bdh-base")

    client = chromadb.PersistentClient(path=DB_PATH)
    col = client.get_collection(name=COLLECTION_NAME)

    df = pd.read_csv(TEST_CSV, encoding_errors="ignore")
    book_res = {}
    preds = []
    rationales = [] # New list for rationale strings
    
    print(f"Starting prediction on {len(df)} rows...")

    for i in range(len(df)):
        row = df.iloc[i]
        book = str(row["book_name"]).strip()

        if book not in book_res:
            try:
                tf = load_tfidf(book)
                ch = load_chunks(book)
                models, thr = load_ensemble(book, d=cfg.n_embd)
                book_res[book] = {"tfidf": tf, "chunks": ch, "models": models, "thr": thr}
            except Exception as e:
                print(f"Skipping book setup for {book} due to error: {e}")
                book_res[book] = None

        res_pack = book_res[book]
        if res_pack is None or not res_pack["models"]:
             preds.append(1) # Default consistent
             rationales.append("Model setup failed; defaulting to consistent.")
             continue

        tfidf = res_pack["tfidf"]
        chunks = res_pack["chunks"]
        models = res_pack["models"]
        thr = res_pack["thr"]

        # ---- Retrieval (Stage 1 & 2) ----
        cand_ids = set()
        cand_docs = {}
        cand_meta = {}

        for qtxt in multi_queries(row):
            qemb = bdh_embed_text(lm, tok, qtxt, DEVICE, max_len=512, windows=3, pool="mean").detach().cpu().tolist()
            res = col.query(
                query_embeddings=[qemb],
                n_results=WIDE_N,
                where={"book_name": book},
                include=["documents", "metadatas"], 
            )
            if res["ids"] and res["ids"][0]:
                for cid, doc, meta in zip(res["ids"][0], res["documents"][0], res["metadatas"][0]):
                    cand_ids.add(cid)
                    cand_docs[cid] = doc
                    cand_meta[cid] = meta

        cand_ids = list(cand_ids)
        if not cand_ids:
            preds.append(1) # Consistent
            rationales.append("No relevant text found in book; assumed consistent.")
            continue

        qfull_text = f"CHAR: {row['char']}\nCAPTION: {row.get('caption','')}\nBACKSTORY: {row.get('content','')}"
        qv = tfidf["vectorizer"].transform([qfull_text])
        qv = sk_normalize(qv, axis=1)

        id_to_idx = {cid: j for j, cid in enumerate(tfidf["ids"])}
        idxs = [id_to_idx[cid] for cid in cand_ids if cid in id_to_idx]
        if not idxs:
            preds.append(1) # Consistent
            rationales.append("No lexical matches found; assumed consistent.")
            continue

        Xc = tfidf["X"][idxs]
        scores = (Xc @ qv.T).toarray().reshape(-1)

        top = sorted(range(len(idxs)), key=lambda t: scores[t], reverse=True)[:FINAL_K]
        chosen_ids = [cand_ids[top[t]] for t in range(len(top))]
        chosen_ids = sorted(chosen_ids, key=lambda cid: int(cand_meta[cid].get("offset", 0)))

        # ---- Evidence & Prediction ----
        cache_ids = chunks["ids"]
        cache_map = {cid: j for j, cid in enumerate(cache_ids)}
        ev = []
        sims = []
        
        # Track IDs to retrieve text for rationale
        evidence_chunk_ids = []

        q = make_query_emb(lm, tok, row) 
        q_cpu = q.detach().cpu()

        E_cache = torch.nn.functional.normalize(chunks["embs"], dim=-1)
        for cid in chosen_ids:
            if cid in cache_map:
                j = cache_map[cid]
                ev.append(E_cache[j])
                sims.append(float((E_cache[j] @ q_cpu).item()))
                evidence_chunk_ids.append(cid)

        if not ev:
            preds.append(1) # Consistent
            rationales.append("No valid embeddings found; assumed consistent.")
            continue

        cT = torch.stack(ev, dim=0).to(DEVICE)
        sT = torch.tensor(sims, dtype=torch.float32, device=DEVICE)

        with torch.no_grad():
            ps = []
            for m in models:
                p = torch.sigmoid(m(q.unsqueeze(0), cT.unsqueeze(0), sT.unsqueeze(0)))[0].item()
                ps.append(p)
            avg_p = sum(ps) / len(ps)

        # ---- Rationale Generation ----
        # Find the chunk with highest similarity score
        best_sim_idx = sims.index(max(sims))
        best_cid = evidence_chunk_ids[best_sim_idx]
        best_text = cand_docs.get(best_cid, "").replace("\n", " ").strip()
        
        # Create a snippet (first 100 chars or reasonable length)
        snippet = best_text[:120] + "..." if len(best_text) > 120 else best_text
        
        # CHANGED: 1 for consistent, 0 for inconsistent
        prediction = 1 if avg_p >= thr else 0
        preds.append(prediction)

        if prediction == 1:
            rationale = f"Confidence {avg_p:.2f}: Evidence supports claim. Text ref: '{snippet}'"
        else:
            rationale = f"Confidence {avg_p:.2f} (Low): Insufficient evidence match. Closest text: '{snippet}'"
            
        rationales.append(rationale)

        if i % 50 == 0:
            print(f"Processed {i}/{len(df)} | Prob: {avg_p:.3f}")

    out = df.copy()
    out["prediction"] = preds
    out["rationale"] = rationales # Add column
    out.to_csv(OUT_CSV, index=False)
    print(f"✅ Wrote {OUT_CSV}")

if __name__ == "__main__":
    main()
