import torch
import chromadb
from chromadb.utils import embedding_functions
import os
import numpy as np

# --- LOCAL IMPORTS ---
from bdh.bdh import BDH, BDHConfig

# --- CONFIG ---
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
SEQ_LEN = 256
CHUNK_OVERLAP = 50
PRETRAINED_PATH = "checkpoints_PRETRAIN_BDH/bdh_pretrained_epoch_5.pth" # Use your best epoch
DB_PATH = "./my_novel_db"
NOVELS = {
    "In search of the castaways": "In search of the castaways.txt",
    "The Count of Monte Cristo": "The Count of Monte Cristo.txt"
}

# --- 1. MODEL LOADER ---
def load_pretrained_extractor(path):
    config = BDHConfig()
    config.vocab_size = 256
    model = BDH(config).to(DEVICE)
    
    print(f"Loading pretrained weights from {path}...")
    state_dict = torch.load(path, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.eval()
    return model

# --- 2. EMBEDDING FUNCTION ---
@torch.no_grad()
def get_bdh_embedding(model, byte_data):
    # Convert bytes to tensor
    inputs = torch.tensor(list(byte_data), dtype=torch.long).unsqueeze(0).to(DEVICE)
    
    # Get hidden states (Assuming BDH returns: logits, hidden_states)
    _, hidden_states = model(inputs)
    
    # Mean Pooling: Average across the sequence dimension
    # hidden_states shape usually: [Batch, Seq_Len, Hidden_Dim]
    embeddings = hidden_states.mean(dim=1).cpu().numpy()
    return embeddings.flatten().tolist()

# --- 3. MAIN DB BUILDER ---
def build_db():
    model = load_pretrained_extractor(PRETRAINED_PATH)
    
    client = chromadb.PersistentClient(path=DB_PATH)
    # We create/get a collection. We don't provide an embedding_function 
    # because we are supplying our own custom BDH vectors manually.
    collection = client.get_or_create_collection(name="novel_collection")

    for novel_name, file_path in NOVELS.items():
        print(f"Processing {novel_name}...")
        
        with open(file_path, 'rb') as f:
            full_text = f.read()

        step = SEQ_LEN - CHUNK_OVERLAP
        count = 0
        
        for i in range(0, len(full_text) - SEQ_LEN, step):
            chunk = full_text[i : i + SEQ_LEN]
            
            # Extract Vector
            vector = get_bdh_embedding(model, chunk)
            
            # Add to DB
            collection.add(
                embeddings=[vector],
                documents=[chunk.decode('utf-8', errors='ignore')],
                metadatas=[{"novel": novel_name, "offset": i}],
                ids=[f"{novel_name}_{i}"]
            )
            
            count += 1
            if count % 100 == 0:
                print(f"  Inserted {count} chunks...")

    print(f"✅ Database built successfully at {DB_PATH}")

if __name__ == "__main__":
    build_db()