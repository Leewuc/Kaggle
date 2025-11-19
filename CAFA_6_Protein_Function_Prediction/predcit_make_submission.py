import torch
from torch.utils.data import DataLoader
import pandas as pd 
import numpy as np
import json
from tqdm.auto import tqdm

from dataset import CafaDataset, AMINO_ACIDS
from model import CafaCNN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

with open("go_terms.json", "r") as f:
    go_terms = json.load(f)
num_labels = len(go_terms)

test_ds = CafaDataset("test_df.pkl", labels_path=None, max_len=1024)
test_loader = DataLoader(test_ds, batch_size=64, shuffle=False, num_workers=4)

vocab_size = len(AMINO_ACIDS) + 2
model = CafaCNN(vocab_size=vocab_size, num_labels=num_labels).to(device)
state_dict = torch.load("cafa_cnn_baseline.pt", map_location=device)
model.load_state_dict(state_dict)
model.eval()

rows = []

THRESHOLD = 0.5
MAX_TERMS_PER_PROT = 50

with torch.no_grad():
    pbar = tqdm(test_loader, desc="Predicting on test")
    for batch in pbar:
        x, entry_ids = batch
        x = x.to(device)

        logits = model(x)
        probs = torch.sigmoid(logits)
        probs = probs.cpu().numpy()

        for eid, prob_vec in zip(entry_ids, probs):
            idxs = np.where(prob_vec >= THRESHOLD)[0]

            if len(idxs) > MAX_TERMS_PER_PROT:
                topk_idxs = np.argsort(prob_vec)[::-1][:MAX_TERMS_PER_PROT]
                idxs = np.intersect1d(idxs, topk_idxs)
            
            for j in idxs:
                rows.append((eid, go_terms[j], float(prob_vec[j])))

sample = pd.read_csv("sample_submission.tsv", sep="\t")
col_names = sample.columns.tolist()

sub_df = pd.DataFrame(rows, columns=col_names)

print("submission rows:", len(sub_df))
print(sub_df.head())

# 6. 제출 파일 저장
sub_df.to_csv("submission_baseline.tsv", sep="\t", index=False)
print("Saved submission_baseline.tsv")