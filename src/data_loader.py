import torch
import pandas as pd
import os
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

class NovelBackstoryDataset(Dataset):
    def __init__(self, novels_dir, csv_path, tokenizer, max_len=512):
        self.novels_dir = novels_dir
        # Check if file exists to avoid crashes
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV file not found: {csv_path}")
            
        self.data = pd.read_csv(csv_path)
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.novel_cache = {} # Cache to avoid re-reading large book files

    def _read_novel(self, filename):
        # Only read the file if we haven't already
        if filename not in self.novel_cache:
            path = os.path.join(self.novels_dir, filename)
            if not os.path.exists(path): 
                print(f"Warning: Novel {filename} not found.")
                return ""
            with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                self.novel_cache[filename] = f.read()
        return self.novel_cache[filename]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        
        # NOTE: Ensure your CSV has columns 'novel_filename' and 'backstory'
        novel_filename = row.get('novel_filename', row.get('novel', '')) 
        backstory = row.get('backstory', row.get('text', ''))
        
        novel_text = self._read_novel(novel_filename)
        
        # Tokenize: [CLS] Backstory [SEP] Novel_Snippet [SEP]
        inputs = self.tokenizer(
            backstory,
            novel_text,
            truncation=True,
            max_length=self.max_len,
            padding="max_length",
            return_tensors="pt"
        )

        item = {
            "input_ids": inputs["input_ids"].squeeze(0),
            "attention_mask": inputs["attention_mask"].squeeze(0),
            "story_id": row.get('story_id', idx) # Pass ID for submission
        }

        # If training data (has label), include it
        if 'label' in row:
            item['labels'] = torch.tensor(row['label'], dtype=torch.long)
            
        return item

def collate_fn(batch):
    # Stack the inputs into a batch
    input_ids = pad_sequence([b['input_ids'] for b in batch], batch_first=True, padding_value=0)
    attention_mask = pad_sequence([b['attention_mask'] for b in batch], batch_first=True, padding_value=0)
    
    batch_out = {
        "input_ids": input_ids,
        "attention_mask": attention_mask
    }
    
    # Handle labels if they exist
    if 'labels' in batch[0]:
        batch_out['labels'] = torch.stack([b['labels'] for b in batch])
        
    return batch_out