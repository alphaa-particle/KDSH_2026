import os
import chromadb
from chromadb.utils import embedding_functions
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ---------------- CONFIG ----------------
DB_PATH = "./my_novel_db"

NOVELS = {
    "In Search of the Castaways": "data/In search of the castaways.txt",
    "The Count of Monte Cristo": "data/The Count of Monte Cristo.txt",
}

CHUNK_SIZE = 500
CHUNK_OVERLAP = 100
BATCH_SIZE = 2000


def create_vector_db():
    print("🔗 Initializing ChromaDB...")
    client = chromadb.PersistentClient(path=DB_PATH)

    embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )

    # Reset collection (fresh DB)
    try:
        client.delete_collection("novel_collection")
        print("♻️ Existing collection deleted")
    except Exception:
        pass

    collection = client.create_collection(
        name="novel_collection",
        embedding_function=embedding_fn
    )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ".", " ", ""]
    )

    print("📚 Processing novels...")
    for novel_name, path in NOVELS.items():
        if not os.path.exists(path):
            print(f"⚠️ File not found: {path}")
            continue

        print(f"➡️ Loading: {novel_name}")
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()

        chunks = splitter.split_text(text)
        print(f"   🔹 Total chunks: {len(chunks)}")

        ids = [f"{novel_name}_{i}" for i in range(len(chunks))]
        metadatas = [{"novel": novel_name} for _ in chunks]

        for i in range(0, len(chunks), BATCH_SIZE):
            collection.add(
                documents=chunks[i:i+BATCH_SIZE],
                ids=ids[i:i+BATCH_SIZE],
                metadatas=metadatas[i:i+BATCH_SIZE],
            )
            print(f"   ✅ Added chunks {i} → {min(i+BATCH_SIZE, len(chunks))}")

    print(f"\n🎉 Vector DB created at `{DB_PATH}`")
    print("✅ Collection name: novel_collection")


if __name__ == "__main__":
    create_vector_db()
