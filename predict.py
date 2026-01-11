# FILE: predict.py
import torch
import pandas as pd
import chromadb
from chromadb.utils import embedding_functions

from bdh.bdh import BDH, BDHConfig
from bdh import BDHTokenizer

# ---------------- CONFIG ----------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MODEL_PATH = "/Users/lucky/Desktop/KDSH/checkpoints/bdh_best_model.pth"   # adjust if name differs
TEST_CSV = "data/test.csv"
OUTPUT_CSV = "submission.csv"
DB_PATH = "./my_novel_db"

MAX_SEQ_LEN = 128
TOP_K_EVIDENCE = 3
MARGIN_THRESHOLD = 0.15

# ---------------- SAFE STRING ----------------
def safe_str(x):
    if pd.isna(x):
        return ""
    return str(x)

# ---------------- LOAD MODEL ----------------
def load_model():
    print("🔄 Loading tokenizer & model...")

    tokenizer = BDHTokenizer.from_pretrained("bdh-base")

    config = BDHConfig(
        n_neurons=2048,
        sparsity=0.02,
        vocab_size=tokenizer.vocab_size
    )

    model = BDH(config).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    print("✅ Model loaded")
    return model, tokenizer

# ---------------- MAIN ----------------
def main():
    model, tokenizer = load_model()

    print("🔗 Connecting to ChromaDB...")
    db = chromadb.PersistentClient(path=DB_PATH)
    emb_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )
    collection = db.get_collection("novel_collection", embedding_function=emb_fn)
    print("✅ ChromaDB ready")

    df = pd.read_csv(TEST_CSV, engine="python")
    print(f"📄 Loaded test.csv with {len(df)} rows")

    predictions = []

    with torch.no_grad():
        for idx, row in df.iterrows():
            caption = safe_str(row.get("caption"))
            backstory = safe_str(row.get("content"))
            book_name = safe_str(row.get("book_name"))

            # ---------- RAG ----------
            evidence = ""
            if caption and book_name:
                try:
                    results = collection.query(
                        query_texts=[caption],
                        n_results=TOP_K_EVIDENCE,
                        where={"novel": book_name},
                    )
                    docs = results.get("documents", [[]])[0]
                    evidence = "\n".join(docs)
                except:
                    evidence = ""

            # ---------- PROMPT ----------
            prompt = (
                f"Evidence:\n{evidence}\n\n"
                f"Backstory:\n{backstory}\n\n"
                f"Caption:\n{caption}\n\n"
                f"Task: Decide whether the caption inconsistent or is consistent.\n"
                f"Answer:"
            )

            enc = tokenizer(
                prompt,
                padding="max_length",
                truncation=True,
                max_length=MAX_SEQ_LEN,
                return_tensors="pt"
            )

            input_ids = enc["input_ids"].to(DEVICE)

            logits = model(input_ids)["cls_logits"]
            probs = torch.softmax(logits, dim=-1)

            margin = probs[:, 1] - probs[:, 0]
            pred = "consistent" if margin.item() > MARGIN_THRESHOLD else "inconsistent"

            predictions.append(pred)

            if idx % 50 == 0:
                print(f"Processed {idx}/{len(df)}")

    df["prediction"] = predictions
    df.to_csv(OUTPUT_CSV, index=False)

    print(f"\n🎉 DONE! Predictions saved to `{OUTPUT_CSV}`")

# ---------------- RUN ----------------
if __name__ == "__main__":
    main()
