# FILE: data/train.py
import os
import csv
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import chromadb
from chromadb.utils import embedding_functions
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score

from bdh.bdh import BDH, BDHConfig
from bdh import BDHTokenizer

torch.set_num_threads(4)

# ---------------- CONFIG ----------------
BATCH_SIZE = 8
EPOCHS = 10
LR = 1e-4
MAX_SEQ_LEN = 128
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

DB_PATH = "./my_novel_db"
TRAIN_CSV = "./data/final_train.csv"
VAL_CSV   = "./data/final_val.csv"

TOP_K_EVIDENCE = 3
MARGIN_THRESHOLD = 0.15

LOG_FILE = "training_log.csv"   # saved in ROOT directory

# ---------------- FOCAL LOSS ----------------
class FocalLoss(nn.Module):
    def __init__(self, gamma=1.5, alpha=None):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, logits, targets):
        ce = nn.functional.cross_entropy(logits, targets, reduction="none")
        pt = torch.exp(-ce)
        loss = ((1 - pt) ** self.gamma) * ce
        if self.alpha is not None:
            loss = self.alpha[targets] * loss
        return loss.mean()

# ---------------- DATASET ----------------
class ConsistencyDataset(Dataset):
    def __init__(self, df, tokenizer, collection):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.collection = collection

    def __len__(self):
        return len(self.df)

    def safe_str(self, x):
        if pd.isna(x):
            return ""
        return str(x)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        caption   = self.safe_str(row.get("caption"))
        backstory = self.safe_str(row.get("content"))
        book_name = self.safe_str(row.get("book_name"))

        # -------- RAG (SAFE) --------
        evidence = ""
        if caption.strip() and book_name.strip():
            try:
                results = self.collection.query(
                    query_texts=[caption],
                    n_results=TOP_K_EVIDENCE,
                    where={"novel": book_name},
                )
                docs = results.get("documents", [[]])[0]
                evidence = "\n".join(docs)
            except Exception:
                evidence = ""

        prompt = (
            f"Evidence:\n{evidence}\n\n"
            f"Backstory:\n{backstory}\n\n"
            f"Caption:\n{caption}\n\n"
            f"Task: Decide whether the caption inconsistent or is consistent.\n"
            f"Answer:"
        )

        enc = self.tokenizer(
            prompt,
            padding="max_length",
            truncation=True,
            max_length=MAX_SEQ_LEN,
            return_tensors="pt",
        )

        # label: 1 = consistent, 0 = contradict
        label = 1 if row["label"] == "consistent" else 0

        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "label": torch.tensor(label, dtype=torch.long),
        }

# ---------------- TRAIN ----------------
def train():
    print(f"🚀 Training BDH on {DEVICE}")

    # -------- TOKENIZER --------
    tokenizer = BDHTokenizer.from_pretrained("bdh-base")

    # -------- VECTOR DB --------
    db = chromadb.PersistentClient(path=DB_PATH)
    emb_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )
    collection = db.get_collection("novel_collection", embedding_function=emb_fn)

    # -------- LOAD DATA --------
    train_df = pd.read_csv(TRAIN_CSV, engine="python").dropna(subset=["label"])
    val_df   = pd.read_csv(VAL_CSV, engine="python").dropna(subset=["label"])

    train_ds = ConsistencyDataset(train_df, tokenizer, collection)
    val_ds   = ConsistencyDataset(val_df, tokenizer, collection)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE)

    # -------- MODEL --------
    config = BDHConfig(
        n_neurons=2048,
        sparsity=0.02,
        vocab_size=tokenizer.vocab_size
    )
    model = BDH(config).to(DEVICE)

    criterion = FocalLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)

    # -------- LOG FILE --------
    with open(LOG_FILE, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "val_loss", "accuracy", "f1_score"])

    best_f1 = 0.0
    os.makedirs("checkpoints", exist_ok=True)

    # -------- TRAIN LOOP --------
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0

        for batch in train_loader:
            optimizer.zero_grad()

            ids = batch["input_ids"].to(DEVICE)
            labels = batch["label"].to(DEVICE)

            out = model(ids, cls_labels=labels)
            loss = out["loss"]

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)

        # -------- VALIDATION --------
        model.eval()
        preds, trues = [], []
        val_loss = 0.0

        with torch.no_grad():
            for batch in val_loader:
                ids = batch["input_ids"].to(DEVICE)
                labels = batch["label"].to(DEVICE)

                out = model(ids, cls_labels=labels)
                val_loss += out["loss"].item()

                logits = out["cls_logits"]
                probs = torch.softmax(logits, dim=-1)

                margin = probs[:, 1] - probs[:, 0]
                pred = (margin > MARGIN_THRESHOLD).long()

                preds.extend(pred.cpu().numpy())
                trues.extend(labels.cpu().numpy())

        val_loss /= len(val_loader)
        acc = accuracy_score(trues, preds)
        f1  = f1_score(trues, preds, zero_division=0)

        print(
            f"\nEpoch {epoch+1}/{EPOCHS} | "
            f"Train Loss: {avg_train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Acc: {acc:.4f} | F1: {f1:.4f}"
        )

        # -------- CSV LOG --------
        with open(LOG_FILE, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([epoch + 1, avg_train_loss, val_loss, acc, f1])

        # -------- SAVE BEST --------
        if f1 > best_f1:
            best_f1 = f1
            torch.save(model.state_dict(), "checkpoints/bdh_best_model.pth")
            print("✅ Best model saved")

    print(f"\n🔥 Training finished | Best F1: {best_f1:.4f}")

# ---------------- ENTRY ----------------
if __name__ == "__main__":
    train()
