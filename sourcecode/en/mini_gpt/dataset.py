"""Dataset that creates training batches of (input, target) from text"""
import torch
from torch.utils.data import Dataset


class TextDataset(Dataset):
    def __init__(self, token_ids, seq_len):
        """token_ids: A long list of integers representing the tokenized full text"""
        self.token_ids = torch.tensor(token_ids, dtype=torch.long)
        self.seq_len = seq_len

    def __len__(self):
        # Number of starting positions where non-overlapping chunks of size seq_len+1 can be extracted
        return len(self.token_ids) - self.seq_len

    def __getitem__(self, idx):
        chunk = self.token_ids[idx: idx + self.seq_len + 1]
        x = chunk[:-1]   # Input: the first seq_len tokens
        y = chunk[1:]    # Target: the next seq_len tokens (shifted by one)
        return x, y
