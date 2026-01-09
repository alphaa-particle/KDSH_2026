import torch
import pandas as pd
import os
import re
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

    def _clean_text(self, text):
        """Fixes common encoding artifacts in the dataset."""
        if not isinstance(text, str):
            return ""
        # Fix Windows-1252 / UTF-8 mixups common in this dataset
        text = text.replace("â€”", "—")
        text = text.replace("â€™", "'")
        text = text.replace("â€œ", '"')
        text = text.replace("â€", '"')
        text = text.replace("Ã©", "é")
        text = text.replace("ChÃ¢teau", "Château")
        return text.strip()

    def _read_novel(self, filename):
        if filename not in self.novel_cache:
            # Handle potentially missing extensions
            path = os.path.join(self.novels_dir, filename)
            if not os.path.exists(path): 
                if path.endswith('.txt') and os.path.exists(path[:-4]):
                    path = path[:-4]
                elif not path.endswith('.txt') and os.path.exists(path + ".txt"):
                    path = path + ".txt"
                else:
                    # Return empty string if book not found so code doesn't crash
                    print(f"Warning: Novel file not found at {path}")
                    return ""
                    
            with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                self.novel_cache[filename] = f.read()
        return self.novel_cache[filename]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        
        # 1. Get Metadata
        character = self._clean_text(str(row.get('char', '')))
        caption = self._clean_text(str(row.get('caption', '')))
        
        # 2. Get and Clean Content (The Backstory)
        raw_content = row['content']
        backstory = self._clean_text(str(raw_content))
        
        # 3. Construct Rich Input
        # Format: "[CLS] Character: <Name> | Topic: <Caption> | Claim: <Backstory> [SEP] <Novel Text>"
        # This tells the model WHO and WHAT to look for.
        
        # Handle empty captions gracefully
        if caption and caption.lower() != 'nan':
            combined_input = f"Character: {character} | Topic: {caption} | Claim: {backstory}"
        else:
            combined_input = f"Character: {character} | Claim: {backstory}"

        # 4. Load Novel Context
        novel_filename = row['book_name']
        if not str(novel_filename).endswith('.txt'):
            novel_filename = str(novel_filename) + ".txt"
            
        novel_text = self._read_novel(novel_filename)
        
        # 5. Tokenize
        inputs = self.tokenizer(
            combined_input,     # The specific claim (Context)
            novel_text,         # The evidence (Novel)
            truncation=True,    # Truncate if too long
            max_length=self.max_len,
            padding="max_length",
            return_tensors="pt"
        )

        item = {
            "input_ids": inputs["input_ids"].squeeze(0),
            "attention_mask": inputs["attention_mask"].squeeze(0),
            "story_id": row.get('id', idx) # Use 'id' from CSV
        }

        # 6. Handle Labels
        if 'label' in row:
            raw_label = row['label']
            if isinstance(raw_label, str):
                raw_label = raw_label.strip().lower()
                # Map dataset specific strings to 0/1
                if raw_label in ['consistent', 'entailment', '1']:
                    label_val = 1
                elif raw_label in ['inconsistent', 'contradiction', 'contradict', '0']:
                    label_val = 0
                else:
                    label_val = 0 # Default to inconsistent if unsure
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