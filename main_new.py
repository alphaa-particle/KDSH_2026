import torch
import pandas as pd
import chromadb
from chromadb.utils import embedding_functions
import os
import sys

# --- IMPORTS ---
from bdh.bdh import BDH, BDHConfig 
from bdh import BDHTokenizer

# --- CONFIGURATION ---
weights_path = "checkpoints/bdh_epoch_3.pth"  # <--- Using your best epoch
test_csv_path = r"C:\\Users\\vaibh\\Desktop\\KDSH_Solution_2026\\data\\test.csv" # <--- Check this path
output_csv_path = "submission.csv"
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# --- 1. SETUP ---
def load_model():
    print(f"Loading model from {weights_path}...")
    tokenizer = BDHTokenizer.from_pretrained("bdh-base")
    
    config = BDHConfig()
    config.vocab_size = tokenizer.vocab_size
    
    model = BDH(config).to(device)
    
    # Load Weights
    checkpoint = torch.load(weights_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
        
    model.eval()
    return model, tokenizer

# --- 2. VECTOR DB CONNECTION ---
print("Connecting to Vector DB...")
db_client = chromadb.PersistentClient(path="./my_novel_db")
emb_fn = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
collection = db_client.get_collection("novel_collection", embedding_function=emb_fn)

def get_evidence(novel, caption):
    try:
        results = collection.query(
            query_texts=[caption],
            n_results=1,
            where={"novel": novel}
        )
        if results['documents'] and len(results['documents'][0]) > 0:
            return results['documents'][0][0]
    except:
        pass
    return ""

# --- 3. PREDICTION LOOP ---
def main():
    model, tokenizer = load_model()
    
    # Load Test Data
    df = pd.read_csv(test_csv_path)
    print(f"Loaded Test CSV: {len(df)} rows")
    
    # Prepare Token IDs for comparison
    id_cons = tokenizer.encode("consistent", add_special_tokens=False)[0]
    id_incons = tokenizer.encode("inconsistent", add_special_tokens=False)[0]
    
    predictions = []
    
    print("Starting Inference...")
    for idx, row in df.iterrows():
        # Handle potential column name differences in Test vs Train
        # Usually Test CSVs have similar structures
        novel = row.get('book_name', row.get('novel', '')) 
        caption = row.get('caption', '')
        backstory = row.get('content', row.get('back_story', ''))
        
        # 1. Get Evidence
        evidence = get_evidence(novel, caption)
        
        # 2. Build Prompt (MUST MATCH TRAINING FORMAT)
        prompt = (
            f"Evidence: {evidence}\n"
            f"Backstory: {backstory}\n"
            f"Caption: {caption}\n"
            f"Status:"
        )
        
        # 3. Run Model
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(device)
        
        with torch.no_grad():
            logits, _ = model(inputs['input_ids'])
            
        # 4. Compare Logits (Last Token)
        last_token_logits = logits[0, -1, :]
        score_cons = last_token_logits[id_cons].item()
        score_incons = last_token_logits[id_incons].item()
        
        # 5. Decide
        if score_cons > score_incons:
            predictions.append("consistent") # Or "Consistent"
        else:
            predictions.append("inconsistent") # Or "Inconsistent"
            
        if idx % 50 == 0:
            print(f"Processed {idx}/{len(df)}")

    # Save
    df['prediction'] = predictions
    df.to_csv(output_csv_path, index=False)
    print(f"🎉 Done! Submission saved to {output_csv_path}")

if __name__ == "__main__":
    main()