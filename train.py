import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import chromadb
from chromadb.utils import embedding_functions
import os
import csv
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, recall_score

# --- IMPORTS FROM YOUR CODE ---
from bdh.bdh import BDH, BDHConfig
from bdh import BDHTokenizer

# --- CONFIGURATION ---
BATCH_SIZE = 4
EPOCHS = 40
PATIENCE = 3
LEARNING_RATE = 2e-5
MAX_SEQ_LEN = 512
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
DB_PATH = "./my_novel_db"  # Ensure this matches the DB created

# Update these paths to your local files
REAL_CSV_PATH = r"C:\Users\vaibh\Desktop\KDSH_Solution_2026\data\train.csv"
SYNTHETIC_CSV_PATH = r"C:\Users\vaibh\Desktop\KDSH_Solution_2026\data\synthetic.csv" 

LOG_FILE = "training_log1.csv"
synthetic_df = pd.read_csv(SYNTHETIC_CSV_PATH)

# --- 1. DATASET CLASS ---
class ConsistencyDataset(Dataset):
    def __init__(self, dataframe, tokenizer, db_collection):
        self.data = dataframe.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.collection = db_collection
        
        self.id_cons = tokenizer.encode("consistent", add_special_tokens=False)[0]
        self.id_incons = tokenizer.encode("inconsistent", add_special_tokens=False)[0]
        
        # Enhanced label map to handle various formats from synthetic generation
        self.label_map = {
            "consistent": self.id_cons,
            "inconsistent": self.id_incons,
            "Consistent": self.id_cons,
            "Inconsistent": self.id_incons,
            0: self.id_incons,
            1: self.id_cons
        }

    def __len__(self):
        return len(self.data)

    def get_evidence(self, novel, caption):
        try:
            results = self.collection.query(
                query_texts=[caption],
                n_results=1,
                where={"novel": novel}
            )
            if results['documents'] and len(results['documents'][0]) > 0:
                return results['documents'][0][0]
        except:
            pass
        return ""

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        
        novel_name = row['book_name'] 
        caption_text = row['caption']
        backstory = row['content']
        
        evidence = self.get_evidence(novel_name, caption_text)
        
        prompt = (
            f"Evidence: {evidence}\n"
            f"Backstory: {backstory}\n"
            f"Caption: {caption_text}\n"
            f"Status:"
        )
        
        encoding = self.tokenizer(
            prompt,
            padding='max_length',
            truncation=True,
            max_length=MAX_SEQ_LEN,
            return_tensors="pt"
        )
        
        input_ids = encoding['input_ids'].squeeze(0)
        label_val = row['label']
        target_id = self.label_map.get(label_val, self.id_incons)
        
        is_consistent_true = 1 if target_id == self.id_cons else 0

        return {
            "input_ids": input_ids,
            "target": torch.tensor(target_id, dtype=torch.long),
            "true_binary_label": torch.tensor(is_consistent_true, dtype=torch.long)
        }

# --- 2. EVALUATE FUNCTION ---
def evaluate(model, dataloader, tokenizer, device):
    model.eval()
    all_preds = []
    all_labels = []
    total_loss = 0
    criterion = nn.CrossEntropyLoss()
    
    id_cons = tokenizer.encode("consistent", add_special_tokens=False)[0]
    id_incons = tokenizer.encode("inconsistent", add_special_tokens=False)[0]

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            targets = batch['target'].view(-1).to(device)
            true_labels = batch['true_binary_label'].numpy()
            
            logits, _ = model(input_ids)
            last_token_logits = logits[:, -1, :]
            
            loss = criterion(last_token_logits, targets)
            total_loss += loss.item()

            score_cons = last_token_logits[:, id_cons]
            score_incons = last_token_logits[:, id_incons]
            
            preds = (score_cons > score_incons).long().cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(true_labels)
            
    avg_loss = total_loss / len(dataloader)
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='binary')
    recall = recall_score(all_labels, all_preds, average='binary')
    
    return avg_loss, acc, f1, recall

# --- 3. TRAINING LOOP ---
def train():
    print(f"Initializing on {DEVICE}...")
    tokenizer = BDHTokenizer.from_pretrained("bdh-base")
    
    print("Connecting to DB...")
    db_client = chromadb.PersistentClient(path=DB_PATH)
    emb_fn = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
    collection = db_client.get_collection("novel_collection", embedding_function=emb_fn)
    
    # --- STRATEGIC DATA SPLITTING ---
    print("Loading Real and Synthetic Data...")
    real_df = pd.read_csv(REAL_CSV_PATH)
    
    # Split real data into Train and Val
    train_real_df, val_df = train_test_split(real_df, test_size=0.3, random_state=42)
    
    # Load Synthetic Data
    if os.path.exists(SYNTHETIC_CSV_PATH):
        synthetic_df = pd.read_csv(SYNTHETIC_CSV_PATH)
        # Combine Real Train with ALL Synthetic Data
        train_df = pd.concat([train_real_df, synthetic_df], axis=0).sample(frac=1).reset_index(drop=True)
        print(f"Total Training: {len(train_df)} ({len(train_real_df)} Real + {len(synthetic_df)} Synthetic)")
    else:
        train_df = train_real_df
        print("Warning: Synthetic CSV not found. Proceeding with real data only.")
        
    print(f"Validation set (Strictly Real): {len(val_df)}")

    train_dataset = ConsistencyDataset(train_df, tokenizer, collection)
    val_dataset = ConsistencyDataset(val_df, tokenizer, collection)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    config = BDHConfig()
    config.vocab_size = tokenizer.vocab_size 
    model = BDH(config).to(DEVICE)
    
    # --- WEIGHTED LOSS ---
    class_weights = torch.ones(config.vocab_size).to(DEVICE)
    id_cons = tokenizer.encode("consistent", add_special_tokens=False)[0]
    id_incons = tokenizer.encode("inconsistent", add_special_tokens=False)[0]
    class_weights[id_incons] = 1.5
    class_weights[id_cons] = 1.0
    
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    
    if not os.path.exists("checkpoints_newdb"):
        os.makedirs("checkpoints_newdb")

    with open(LOG_FILE, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Epoch", "Train_Loss", "Val_Loss", "Val_Acc", "Val_Recall", "Val_F1"])

    best_f1 = 0.0
    best_val_loss = float('inf') 
    lowest_val_loss = float('inf') 
    patience_counter = 0

    print("--- Starting Training (Synthetic in Train, Real in Val) ---")
    
    for epoch in range(EPOCHS):
        model.train()
        total_train_loss = 0
        
        for batch in train_loader:
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(DEVICE)
            targets = batch['target'].view(-1).to(DEVICE)
            logits, _ = model(input_ids)
            last_token_logits = logits[:, -1, :]
            loss = criterion(last_token_logits, targets)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()

        # --- VALIDATION ---
        avg_train_loss = total_train_loss / len(train_loader)
        val_loss, val_acc, val_f1, val_recall = evaluate(model, val_loader, tokenizer, DEVICE)
        
        print(f"Epoch {epoch+1} | Train Loss: {avg_train_loss:.4f} | Val F1: {val_f1:.4f} | Acc: {val_acc:.4f} | val_recall: {val_recall:.4f}")

        # Logging
        with open(LOG_FILE, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch + 1, avg_train_loss, val_loss, val_acc, val_recall, val_f1])

        # Checkpoints
        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save(model.state_dict(), "checkpoints_1/bdh_best_model.pth")
            print(f"   ✅ Saved Best Model!")

        # Early Stopping
        if val_loss < lowest_val_loss:
            lowest_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print("🛑 Early Stopping triggered.")
                break
        
        torch.save(model.state_dict(), f"checkpoints/bdh_epoch_{epoch+1}.pth")

if __name__ == "__main__":
    train()