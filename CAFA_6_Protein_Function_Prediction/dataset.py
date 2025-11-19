import torch
from torch.utils.data import Dataset
import pandas as pd 
import numpy as np

AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
AA2IDX = {aa: i + 1 for i, aa in enumerate(AMINO_ACIDS)}
UNK_IDX = len(AMINO_ACIDS) + 1

def seq_to_ids(seq, max_len=1024):
    ids = [AA2IDX.get(a, UNK_IDX) for a in seq]
    if len(ids) > max_len:
        ids = ids[:max_len]
    else:
        ids = ids + [0] * (max_len - len(ids))
    return np.array(ids, dtype=np.int64)

class CafaDataset(Dataset):
    def __init__(self, df_path, labels_path=None, max_len=1024):
        self.df = pd.read_pickle(df_path)
        self.max_len = max_len
        self.labels = None
        if labels_path is not None:
            self.labels = np.load(labels_path).astype(np.float32)
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        seq = row["sequence"]
        seq_ids = seq_to_ids(seq, self.max_len)
        x = torch.tensor(seq_ids, dtype=torch.long)

        if self.labels is not None:
            y = torch.tensor(self.labels[idx], dtype=torch.float32)
            return x, y
        else:
            return x, row["EntryID"]