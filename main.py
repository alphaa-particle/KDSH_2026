import torch
import csv
import os
import numpy as np
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import accuracy_score, f1_score

# Import your custom modules
from src.data_loader import NovelBackstoryDataset, collate_fn
from src.model import BDHConsistencyClassifier
from src.trainer import run_pipeline
from bdh import BDHTokenizer 

def calculate_metrics(predictions, true_labels):
    """Helper to calculate accuracy and F1 score."""
    acc = accuracy_score(true_labels, predictions)
    f1 = f1_score(true_labels, predictions, average='binary')
    return acc, f1

def main():
    # ---------------------------------------------------------
    # 1. Configuration 
    # ---------------------------------------------------------
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running on: {DEVICE}")

    NOVELS_DIR = "./data/novels"
    TRAIN_CSV = "./data/train.csv"
    TEST_CSV = "./data/test.csv"
    CONFIG_PATH = "./bdh/config.json"
    
    # Hyperparameters
    BATCH_SIZE = 4         # Keep small for 4060
    EPOCHS = 10            
    LEARNING_RATE = 5e-6   # Lower LR to prevent oscillation
    WEIGHT_DECAY = 1e-2    # Regularization
    VAL_SPLIT = 0.2        

    # ---------------------------------------------------------
    # 2. Initialize Tokenizer & Datasets
    # ---------------------------------------------------------
    print("Loading Tokenizer...")
    tokenizer = BDHTokenizer.from_pretrained("bdh-base") 
    
    print("Preparing Datasets...")
    if not os.path.exists(TRAIN_CSV) or not os.path.exists(TEST_CSV):
        raise FileNotFoundError("Could not find train.csv or test.csv in ./data/")

    # Load Full Training Data
    full_train_ds = NovelBackstoryDataset(NOVELS_DIR, TRAIN_CSV, tokenizer)
    
    # Create Train/Validation Split
    train_size = int((1 - VAL_SPLIT) * len(full_train_ds))
    val_size = len(full_train_ds) - train_size
    train_ds, val_ds = random_split(full_train_ds, [train_size, val_size])
    
    print(f"Data Split: {len(train_ds)} Training samples | {len(val_ds)} Validation samples")

    # ---------------------------------------------------------
    # 3. Calculate Class Weights (The Fix)
    # ---------------------------------------------------------
    all_labels = [item['labels'].item() for item in train_ds]
    num_0 = all_labels.count(0)
    num_1 = all_labels.count(1)
    
    print(f"Training Distribution: Class 0 (Inconsistent): {num_0} | Class 1 (Consistent): {num_1}")
    
    # Inverse frequency weights
    if num_0 > 0 and num_1 > 0:
        total = num_0 + num_1
        weight_0 = (1 / num_0) * (total / 2.0)
        weight_1 = (1 / num_1) * (total / 2.0)
        class_weights = [weight_0, weight_1]
    else:
        # Fallback if split is bad (e.g. 0 samples of one class)
        class_weights = [1.0, 1.0]
        
    print(f"Using Class Weights: {class_weights}")

    # ---------------------------------------------------------
    # 4. Initialize Model ONCE
    # ---------------------------------------------------------
    print("Initializing Model...")
    model = BDHConsistencyClassifier(config_path=CONFIG_PATH)
    
    # Apply the weights we just calculated
    model.class_weights = class_weights 
    model = model.to(DEVICE)
    
    # Optimizer with Weight Decay
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    # Dataloaders
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, collate_fn=collate_fn, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, collate_fn=collate_fn)
    
    # Load Test Data (for submission)
    test_ds = NovelBackstoryDataset(NOVELS_DIR, TEST_CSV, tokenizer)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, collate_fn=collate_fn)

    # ---------------------------------------------------------
    # 5. Training Loop
    # ---------------------------------------------------------
    print(f"\nStarting training for {EPOCHS} epochs...")
    
    best_val_acc = 0.0

    for epoch in range(EPOCHS):
        print(f"\n--- Epoch {epoch+1}/{EPOCHS} ---")
        
        # A. Train Step
        avg_train_loss = run_pipeline(model, train_loader, tokenizer, DEVICE, mode="train", optimizer=optimizer)
        print(f"Train Loss: {avg_train_loss:.4f}")
        
        # B. Validation Step
        val_preds = run_pipeline(model, val_loader, tokenizer, DEVICE, mode="test")
        
        # Extract labels
        val_labels = []
        for batch in val_loader:
            val_labels.extend(batch['labels'].cpu().numpy())
            
        # Metrics
        acc, f1 = calculate_metrics(val_preds, val_labels)
        print(f"Validation Accuracy: {acc:.4f} | F1 Score: {f1:.4f}")

        # Debug Distribution
        unique, counts = np.unique(val_preds, return_counts=True)
        print(f"DEBUG - Prediction Distribution: {dict(zip(unique, counts))}")

        # Save Best
        if acc > best_val_acc:
            best_val_acc = acc
            torch.save(model.state_dict(), "best_model.pth")
            print(">>> New Best Model Saved!")

    # ---------------------------------------------------------
    # 6. Final Prediction
    # ---------------------------------------------------------
    print("\nLoading best model for submission...")
    if os.path.exists("best_model.pth"):
        model.load_state_dict(torch.load("best_model.pth"))
    else:
        print("Warning: No best model saved (did accuracy never improve?). Using current model.")
    
    print("Generating Submission for Test Data...")
    predictions = run_pipeline(model, test_loader, tokenizer, DEVICE, mode="test")
    
    output_file = "results.csv"
    print(f"Saving predictions to {output_file}...")
    
    with open(output_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "label", "rationale"])
        
        for i, pred in enumerate(predictions):
            story_id = test_ds.data.iloc[i].get('id', i)
            label_out = "consistent" if pred == 1 else "inconsistent"
            writer.writerow([story_id, label_out, "Generated by BDH Recurrent State"])

    print("Done! Baseline Evaluation Complete.")

if __name__ == "__main__":
    main()