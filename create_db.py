import os
import chromadb
from chromadb.utils import embedding_functions
from langchain_text_splitters import RecursiveCharacterTextSplitter

# --- CONFIGURATION ---
NOVELS = {
    "In Search of the Castaways": r"C:\\Users\\vaibh\\Desktop\\KDSH_Solution_2026\\data\\novels\\In Search of the Castaways.txt",
    "The Count of Monte Cristo": r"C:\\Users\\vaibh\\Desktop\\KDSH_Solution_2026\\data\\novels\\The Count of Monte Cristo.txt"
}
DB_PATH = "./my_novel_db_1"  # Changed path to avoid overwriting previous DB


def create_vector_db():
    # 1. Initialize ChromaDB (Persistent means it saves to disk)
    client = chromadb.PersistentClient(path=DB_PATH)
    
    # 2. Use a standard embedding model (MiniLM is fast and good)
    # This automatically downloads the model.
    emb_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )
    
    # 3. Create (or reset) the collection
    try:
        client.delete_collection(name="novel_collection") # specific for fresh start
    except:
        pass
    collection = client.create_collection(name="novel_collection", embedding_function=emb_fn)

    # 4. Initialize Splitter (Keeps paragraphs together)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=2500,    # ~3-4 paragraphs per chunk
        chunk_overlap=500,  # overlap to maintain context
        separators=["\n\n", "\n", ".", " ", ""]
    )

    print("Processing novels...")
    
    for novel_name, file_path in NOVELS.items():
        if not os.path.exists(file_path):
            print(f"Skipping {novel_name}: File {file_path} not found.")
            continue
            
        print(f"  -> Loading {novel_name}...")
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
            
        # Split text into chunks
        chunks = splitter.split_text(text)
        
        # Prepare data for Chroma
        ids = [f"{novel_name}_{i}" for i in range(len(chunks))]
        metadatas = [{"novel": novel_name} for _ in chunks] # KEY STEP: Tagging the novel
        
# --- NEW BATCHING LOGIC START ---
        # The limit is 5461, so we use a safe batch size like 2000
        BATCH_SIZE = 2000 
        
        total_chunks = len(chunks)
        print(f"     Total chunks to add: {total_chunks}")
        
        for i in range(0, total_chunks, BATCH_SIZE):
            end_index = i + BATCH_SIZE
            # Slice the lists to get a smaller batch
            batch_chunks = chunks[i : end_index]
            batch_ids = ids[i : end_index]
            batch_metadatas = metadatas[i : end_index]
            
            # Add this small batch to the DB
            collection.add(
                documents=batch_chunks,
                ids=batch_ids,
                metadatas=batch_metadatas
            )
            print(f"     Added batch {i} to {min(end_index, total_chunks)}")
        # --- NEW BATCHING LOGIC END ---

    print(f"Success! Database saved to {DB_PATH}")

if __name__ == "__main__":
    create_vector_db()