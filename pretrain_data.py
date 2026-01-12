import torch
from torch.utils.data import Dataset
import numpy as np

class ByteLevelNovelDataset(Dataset):
    def __init__(self, file_paths, seq_len=512):
        self.seq_len = seq_len
        self.data = []
        
        for path in file_paths:
            with open(path, 'rb') as f:
                # Read entire book as bytes
                self.data.extend(list(f.read()))
        
        self.data = np.array(self.data, dtype=np.uint8)
        print(f"Total bytes loaded for pretraining: {len(self.data)}")

    def __len__(self):
        # Number of available sequences
        return len(self.data) // self.seq_len - 1

    def __getitem__(self, idx):
        # Grab a window of bytes
        start_idx = idx * self.seq_len
        chunk = self.data[start_idx : start_idx + self.seq_len + 1]
        
        # X is the sequence, Y is the sequence shifted by one (next-byte prediction)
        x = torch.from_numpy(chunk[:-1].astype(np.int64))
        y = torch.from_numpy(chunk[1:].astype(np.int64))
        
        return x, y