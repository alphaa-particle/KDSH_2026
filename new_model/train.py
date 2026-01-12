import os, random, pickle
from typing import List, Dict, Tuple

import torch
from torch import nn
from torch.optim import AdamW
import pandas as pd
from sklearn.model_selection import StratifiedKFold
import numpy as np
from bdh import BDH, BDHConfig, BDHTokenizer, bdh_embed_text, save_ckpt, load_ckpt

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu") 

BASE_DIR = r"C:\Users\vaibh\Desktop\KDSH_Solution_2026\new_model"

DATA_DIR = os.path.join(BASE_DIR, "data")
TRAIN_CSV = os.path.join(DATA_DIR, "train.csv")
NOV_DIR = os.path.join(BASE_DIR, "data")
CKPT_DIR = r"C:\Users\vaibh\Desktop\KDSH_Solution_2026\new_model\checkpoints"
os.makedirs(CKPT_DIR, exist_ok=True)

CKPT_LM = os.path.join(CKPT_DIR, "bdh_lm.pt")
ENSEMBLE_DIR = os.path.join(CKPT_DIR, "ensemble")
os.makedirs(ENSEMBLE_DIR, exist_ok=True)

# ---- Pretrain (use your stable params) ----
PRETRAIN_STEPS = 20000
PRETRAIN_BATCH = 2
PRETRAIN_BLOCK = 256
PRETRAIN_LR = 1e-5

# ---- Head training ----
TOPK_EVIDENCE = 16         # important: smaller = less noise
EPOCHS = 25
LR = 3e-4                 # more stable on tiny data
DROPOUT = 0.3
SEEDS = [11, 22, 33, 44, 55]

INDEX_DIR = "indexes"  # uses *_chunks.pt caches
# ------------------------------------------


def read_text(p: str) -> str:
    try:
        return open(p, "r", encoding="utf-8").read()
    except UnicodeDecodeError:
        return open(p, "r", encoding="latin-1").read()


def sample_lm_batch(stream: bytes, batch: int, block: int, device):
    n = len(stream)
    ix = torch.randint(0, max(1, n - block - 2), (batch,), device=device)
    x = torch.stack([torch.tensor(list(stream[i:i+block]), device=device, dtype=torch.long) for i in ix])
    y = torch.stack([torch.tensor(list(stream[i+1:i+block+1]), device=device, dtype=torch.long) for i in ix])
    return x, y


def pretrain_lm():
    if os.path.exists(CKPT_LM):
        print(f"Found existing LM ckpt: {CKPT_LM} (skip pretrain)")
        return

    # build stream from both books in data/
    novels = [
        os.path.join(NOV_DIR, "The Count of Monte Cristo.txt"),
        os.path.join(NOV_DIR, "In search of the castaways.txt"),
    ]
    stream = ("\n\n".join(read_text(p) for p in novels)).encode("utf-8", errors="ignore")

    cfg = BDHConfig(vocab_size=256)
    model = BDH(cfg).to(DEVICE).train()
    opt = AdamW(model.parameters(), lr=PRETRAIN_LR)

    ema = None
    for step in range(PRETRAIN_STEPS):
        x, y = sample_lm_batch(stream, PRETRAIN_BATCH, PRETRAIN_BLOCK, DEVICE)
        _, loss, _ = model(x, targets=y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        l = float(loss.item())
        ema = l if ema is None else (0.99 * ema + 0.01 * l)
        if step % 500 == 0:
            print(f"pretrain step={step}/{PRETRAIN_STEPS} loss={l:.3f} ema={ema:.3f}")

    save_ckpt(CKPT_LM, model.eval(), cfg, extra={"stage": "lm"})
    print(f"✅ Saved LM: {CKPT_LM}")


class BeliefUpdater(nn.Module):
    def __init__(self, d: int):
        super().__init__()
        self.gate = nn.Sequential(nn.Linear(d*3, d), nn.Sigmoid())
        self.proj = nn.Sequential(nn.Linear(d*3, d), nn.Tanh())

    def forward(self, b, q, c, w):
        # w is similarity weight in [0,1], stabilize updates
        x = torch.cat([b, q, c], dim=-1)
        g = self.gate(x) * w
        h = self.proj(x)
        return (1 - g) * b + g * h


class TrackBHead(nn.Module):
    def __init__(self, d: int, dropout: float):
        super().__init__()
        self.upd = BeliefUpdater(d)
        self.fc = nn.Sequential(
            nn.LayerNorm(d*4 + 2),
            nn.Linear(d*4 + 2, d),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d, 1),
        )

    def forward(self, q, chunks, sims):
        B, K, D = chunks.shape
        b = torch.zeros((B, D), device=q.device)
        for t in range(K):
            w = sims[:, t].clamp(0, 1).unsqueeze(-1)
            b = self.upd(b, q, chunks[:, t, :], w)
        max_sim = sims.max(dim=1).values.unsqueeze(-1)
        mean_sim = sims.mean(dim=1).unsqueeze(-1)
        feats = torch.cat([b, q, (b-q).abs(), b*q, max_sim, mean_sim], dim=-1)
        return self.fc(feats).squeeze(-1)


def label_to_y(lbl: str) -> int:
    lbl = str(lbl).strip().lower()
    return 1 if lbl == "consistent" else 0


def load_chunk_cache(book_name: str):
    import re
    def slug(s): return re.sub(r"[^a-z0-9]+", "", s.lower()).strip("")
    path = os.path.join(INDEX_DIR, f"{slug(book_name)}_chunks.pt")
    obj = torch.load(path, map_location="cpu")
    return obj  # ids, offsets, embs


def split_claims(text: str, max_claims: int = 6) -> List[str]:
    import re
    t = str(text or "").strip()
    if not t:
        return []
    parts = re.split(r"(?<=[\.\!\?])\s+", t)
    out = [p.strip() for p in parts if len(p.strip()) >= 5]
    return out[:max_claims] if out else [t[:300]]


@torch.no_grad()
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


def topk_by_cosine(chunk_embs: torch.Tensor, q: torch.Tensor, offsets: List[int], k: int):
    # chunk_embs: (N,D) cpu float
    E = torch.nn.functional.normalize(chunk_embs, dim=-1)
    sims = (E @ q.cpu()).float()  # (N,)
    vals, inds = torch.topk(sims, k=min(k, sims.numel()))
    offs = torch.tensor([offsets[i] for i in inds.tolist()])
    order = torch.argsort(offs)
    inds = inds[order]
    vals = vals[order]
    return E[inds].to(DEVICE), vals.to(DEVICE)


def best_threshold(probs: List[float], y: List[int]) -> float:
    # sweep thresholds to maximize F1
    import numpy as np
    best_t, best_f1 = 0.5, -1
    for t in np.linspace(0.05, 0.95, 19):
        pred = [1 if p >= t else 0 for p in probs]
        tp = sum(1 for a,b in zip(pred,y) if a==1 and b==1)
        fp = sum(1 for a,b in zip(pred,y) if a==1 and b==0)
        fn = sum(1 for a,b in zip(pred,y) if a==0 and b==1)
        prec = tp/(tp+fp) if (tp+fp) else 0
        rec  = tp/(tp+fn) if (tp+fn) else 0
        f1 = (2*prec*rec/(prec+rec)) if (prec+rec) else 0
        if f1 > best_f1:
            best_f1, best_t = f1, float(t)
    return best_t


def train_per_book_ensemble():
    df = pd.read_csv(TRAIN_CSV, encoding_errors="ignore")
    df["y"] = df["label"].apply(label_to_y)

    lm, cfg, _ = load_ckpt(CKPT_LM, DEVICE)
    tok = BDHTokenizer.from_pretrained("bdh-base")
    for p in lm.parameters():
        p.requires_grad_(False)
    lm.eval()

    books = sorted(df["book_name"].unique().tolist())
    print("Books:", books)

    # Config for K-Fold (replaces random seeds)
    N_SPLITS = 5
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)

    for book in books:
        dbook = df[df["book_name"] == book].reset_index(drop=True)
        cache = load_chunk_cache(book)
        offsets = cache["offsets"]
        chunk_embs = cache["embs"]  # cpu
        
        # Prepare arrays for splitting
        X_dummy = np.zeros(len(dbook))
        y_labels = dbook["y"].values

        print(f"\nTraining {N_SPLITS}-Fold Ensemble for: {book}")

        # Loop over Folds instead of Seeds
        for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_dummy, y_labels)):
            
            # --- Data Split for this Fold ---
            # Convert numpy indices to lists for existing logic compatibility
            tr = train_idx.tolist()
            va = val_idx.tolist()

            head = TrackBHead(cfg.n_embd, DROPOUT).to(DEVICE)
            opt = AdamW(head.parameters(), lr=LR)

            # Class weight calculation (Robust per fold)
            y_train = dbook.loc[tr, "y"]
            pos = float(y_train.sum())
            neg = float(len(tr) - pos)
            
            # Safety for cases with 0 positives in a tiny fold (rare with Stratified, but safe)
            pos_weight_val = neg / max(1.0, pos)
            pos_weight = torch.tensor([pos_weight_val], device=DEVICE)
            loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

            best_f1, best_state, best_t = -1, None, 0.5
            
            # Training Loop
            for epoch in range(EPOCHS):
                head.train()
                
                # Shuffle training data within the epoch (optional but good for SGD)
                np.random.shuffle(tr) 
                
                for i in tr:
                    row = dbook.iloc[i]
                    q = make_query_emb(lm, tok, row)
                    chunks, sims = topk_by_cosine(chunk_embs, q, offsets, TOPK_EVIDENCE)

                    qB = q.unsqueeze(0)
                    cB = chunks.unsqueeze(0)
                    sB = sims.unsqueeze(0)

                    logit = head(qB, cB, sB)
                    y = torch.tensor([float(row["y"])], device=DEVICE)
                    loss = loss_fn(logit, y)

                    opt.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
                    opt.step()

                # --- Validation Step ---
                head.eval()
                probs, yt = [], []
                with torch.no_grad():
                    for i in va:
                        row = dbook.iloc[i]
                        q = make_query_emb(lm, tok, row)
                        chunks, sims = topk_by_cosine(chunk_embs, q, offsets, TOPK_EVIDENCE)
                        
                        # Forward pass
                        p = torch.sigmoid(head(q.unsqueeze(0), chunks.unsqueeze(0), sims.unsqueeze(0)))[0].item()
                        probs.append(p)
                        yt.append(int(row["y"]))

                # Metric Calculation
                if probs:
                    t = best_threshold(probs, yt)
                    pred = [1 if p >= t else 0 for p in probs]
                    
                    tp = sum(1 for a, b in zip(pred, yt) if a == 1 and b == 1)
                    fp = sum(1 for a, b in zip(pred, yt) if a == 1 and b == 0)
                    fn = sum(1 for a, b in zip(pred, yt) if a == 0 and b == 1)
                    
                    prec = tp / (tp + fp) if (tp + fp) else 0
                    rec  = tp / (tp + fn) if (tp + fn) else 0
                    f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) else 0
                else:
                    t, f1 = 0.5, 0.0

                # Save best state for this specific fold
                if f1 >= best_f1:
                    best_f1 = f1
                    best_t = t
                    best_state = head.state_dict()

            # Save Checkpoint (Naming changed from seedX to foldX)
            out = os.path.join(ENSEMBLE_DIR, f"{book.replace(' ','_')}_fold{fold_idx}.pt")
            torch.save(
                {
                    "book": book, 
                    "bdh_config": cfg.__dict__, 
                    "head_state": best_state, 
                    "threshold": best_t,
                    "topk": TOPK_EVIDENCE, 
                    "dropout": DROPOUT, 
                    "fold": fold_idx
                },
                out,
            )
            print(f"   > Fold {fold_idx}: Saved {out} | best_f1={best_f1:.3f} thr={best_t:.2f}")

if __name__ == "__main__":
    # pretrain_lm()
    # print("Now run: python create_db.py  (build caches + chroma)\n")
    train_per_book_ensemble()
