import torch
import pandas as pd
import os
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

class NovelBackstoryDataset(Dataset):
    def __init__(self, novels_dir, csv_path, tokenizer, max_len=512):
        self.novels_dir = novels_dir
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV file not found: {csv_path}")
            
        self.data = pd.read_csv(csv_path)
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.novel_cache = {} 

    def _read_novel(self, filename):
        if filename not in self.novel_cache:
            path = os.path.join(self.novels_dir, filename)
            if not os.path.exists(path): 
                # Fallback: Try checking if the file exists without .txt extension or vice versa
                if path.endswith('.txt') and os.path.exists(path[:-4]):
                    path = path[:-4]
                elif not path.endswith('.txt') and os.path.exists(path + ".txt"):
                    path = path + ".txt"
                else:
                    print(f"Warning: Novel file not found at {path}")
                    return ""
                    
            with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                self.novel_cache[filename] = f.read()
        return self.novel_cache[filename]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        
        # --- FIX 1: Map 'book_name' to filename ---
        novel_filename = row['book_name']
        
        # Ensure it ends with .txt (assuming your files on disk are .txt)
        if not str(novel_filename).endswith('.txt'):
            novel_filename = str(novel_filename) + ".txt"

        # --- FIX 2: Map 'content' to backstory ---
        # We use 'content' based on your screenshot. 
        backstory = row['content']
        
        novel_text = self._read_novel(novel_filename)
        
        inputs = self.tokenizer(
            str(backstory), # Ensure string
            str(novel_text),
            truncation=True,
            max_length=self.max_len,
            padding="max_length",
            return_tensors="pt"
        )

        item = {
            "input_ids": inputs["input_ids"].squeeze(0),
            "attention_mask": inputs["attention_mask"].squeeze(0),
            # Map 'id' from your CSV to story_id
            "story_id": row['id'] 
        }

# ... inside __getitem__ ...
        if 'label' in row:
            raw_label = row['label']
            
            if isinstance(raw_label, str):
                raw_label = raw_label.strip().lower()
                # Map standard NLI labels to 0/1
                if raw_label in ['consistent', 'entailment']:
                    label_val = 1
                elif raw_label in ['inconsistent', 'contradiction', 'contradict']:
                    label_val = 0
                else:
                    try:
                        label_val = int(float(raw_label))
                    except ValueError:
                        print(f"Warning: Unknown label '{raw_label}'. Defaulting to 0.")
                        label_val = 0
            else:
                label_val = int(raw_label)
        
            item['labels'] = torch.tensor(label_val, dtype=torch.long)

        return item

def collate_fn(batch):
    input_ids = pad_sequence([b['input_ids'] for b in batch], batch_first=True, padding_value=0)
    attention_mask = pad_sequence([b['attention_mask'] for b in batch], batch_first=True, padding_value=0)
    
    batch_out = {
        "input_ids": input_ids,
        "attention_mask": attention_mask
    }
    
    if 'labels' in batch[0]:
        batch_out['labels'] = torch.stack([b['labels'] for b in batch])
        
    return batch_out