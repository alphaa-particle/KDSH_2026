import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import gc

# --- LOCAL IMPORTS ---
from bdh.bdh import BDH, BDHConfig
from pretrain_data import ByteLevelNovelDataset

# --- CONFIG ---
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
SEQ_LEN = 256
BATCH_SIZE = 2      
ACCUMULATION_STEPS = 8 
EPOCHS = 5
LR = 1e-5 
NOVEL_PATHS = [
    r"In search of the castaways.txt",
    r"The Count of Monte Cristo.txt"
]

def initialize_weights(model):
    """Stable weight initialization to prevent early NaNs."""
    for name, param in model.named_parameters():
        if 'weight' in name and param.dim() > 1:
            nn.init.xavier_uniform_(param)
        elif 'bias' in name:
            nn.init.constant_(param, 0.0)

def pretrain():
    torch.cuda.empty_cache()
    gc.collect()

    if not os.path.exists("checkpoints_PRETRAIN_BDH"):
        os.makedirs("checkpoints_PRETRAIN_BDH")

    # 1. Load Data
    dataset = ByteLevelNovelDataset(NOVEL_PATHS, seq_len=SEQ_LEN)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # 2. Initialize Model
    config = BDHConfig()
    config.vocab_size = 256
    model = BDH(config).to(DEVICE)
    
    # Apply stable weight initialization
    initialize_weights(model)

    # 3. Setup Loss and Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LR, eps=1e-8, weight_decay=0.01)

    print(f"Starting Byte-Level Pretraining on {DEVICE} (FP32 Stable Mode)...")
    print("Press Ctrl+C at any time to save progress and exit safely.")

    try:
        for epoch in range(EPOCHS):
            model.train()
            total_loss = 0
            optimizer.zero_grad() 
            
            for batch_idx, (x, y) in enumerate(loader):
                x, y = x.to(DEVICE), y.to(DEVICE)
                
                # --- FORWARD PASS ---
                logits, _ = model(x)
                
                # Reshape for loss (Batch * Seq, Vocab)
                logits = logits.view(-1, 256)
                y = y.view(-1)
                
                loss = criterion(logits, y)
                loss = loss / ACCUMULATION_STEPS 

                # --- BACKWARD PASS ---
                loss.backward()

                # --- GRADIENT CLIPPING ---
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)

                # --- TRACKING ---
                current_loss = loss.item() * ACCUMULATION_STEPS
                if torch.isnan(torch.tensor(current_loss)):
                    print(f"❌ NaN detected at Batch {batch_idx}. Stopping training.")
                    return
                
                total_loss += current_loss

                # --- OPTIMIZER STEP ---
                if (batch_idx + 1) % ACCUMULATION_STEPS == 0:
                    optimizer.step()
                    optimizer.zero_grad()

                # --- LOGGING ---
                if batch_idx % 50 == 0:
                    print(f"Epoch {epoch+1} | Batch {batch_idx}/{len(loader)} | Loss: {current_loss:.4f}")

                # --- C: PERIODIC AUTO-SAVE ---
                if batch_idx % 1000 == 0 and batch_idx > 0:
                    print(f"--- Periodic Auto-Save at Batch {batch_idx} ---")
                    torch.save(model.state_dict(), "checkpoints_PRETRAIN_BDH/bdh_auto_save.pth")
            
            # --- A & B: STANDARD END-OF-EPOCH SAVE ---
            avg_loss = total_loss / len(loader)
            print(f"✅ Epoch {epoch+1} Complete. Avg Loss: {avg_loss:.4f}")
            
            save_path = f"checkpoints_PRETRAIN_BDH/bdh_pretrained_epoch_{epoch+1}.pth"
            torch.save(model.state_dict(), save_path)
            print(f"Saved: {save_path}")

    except KeyboardInterrupt:
        # --- A & B: MANUAL INTERRUPTION SAVE ---
        print("\n[!] Manual Interruption detected. Saving progress before exiting...")
        save_path = "checkpoints_PRETRAIN_BDH/bdh_pretrained_INTERRUPTED.pth"
        torch.save(model.state_dict(), save_path)
        print(f"✅ Emergency checkpoint saved to: {save_path}")

    print("Pretraining script finished.")

if __name__ == "__main__":
    pretrain()