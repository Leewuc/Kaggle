import pandas as pd 
from collections import defaultdict
import numpy as np
import json

ia_df = pd.read_csv("IA.tsv", sep="\t", header=None, names=["go_term", "ia_weight"])

train_terms = pd.read_csv("Train/train_terms.tsv", sep="\t")
# print(train_terms.head())

USE_ASPECT = None
if USE_ASPECT is not None:
    train_terms = train_terms[train_terms["aspect"] == USE_ASPECT]

train_tax = pd.read_csv(
    "Train/train_taxonomy.tsv",
    sep="\t",
    header=None,
    names=["EntryID", "taxon_id"]
)

def read_fasta(path, header_type="uniprot"):
    seqs = {}
    with open(path) as f:
        cur_id = None
        cur_seq = []
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if cur_id is not None:
                    seqs[cur_id] = "".join(cur_seq)
                header = line[1:]
                if header_type == "uniprot":
                    parts = header.split("|")
                    cur_id = parts[1]
                else:
                    cur_id = header.split()[0]
                cur_seq = []
            else:
                cur_seq.append(line)
        if cur_id is not None:
            seqs[cur_id] = "".join(cur_seq)
    return seqs

train_seqs = read_fasta("Train/train_sequences.fasta", header_type="uniprot")
test_seqs = read_fasta("Test/testsuperset.fasta", header_type="simple")

print(f"#train sequences: {len(train_seqs)}")
print(f"#test sequences : {len(test_seqs)}")

prot2terms = train_terms.groupby("EntryID")["term"].apply(list)

all_terms = sorted(train_terms["term"].unique())
go2idx = {go: i for i, go in enumerate(all_terms)}
print(f"#unique GO terms (labels): {len(all_terms)}")

rows = []
tax_dict = dict(zip(train_tax["EntryID"], train_tax["taxon_id"]))

for entry_id, terms in prot2terms.items():
    if entry_id not in train_seqs:
        continue
    seq = train_seqs[entry_id]
    taxon_id = tax_dict.get(entry_id, None)
    rows.append(
        {
            "EntryID": entry_id,
            "sequence": seq,
            "taxon_id": taxon_id,
            "terms": terms
        }
    )

train_df = pd.DataFrame(rows)
print(train_df.head())
print("#final train samples:", len(train_df))

num_labels = len(all_terms)
label_matrix = np.zeros((len(train_df), num_labels), dtype=np.float32)

for i, terms in enumerate(train_df["terms"]):
    for t in terms:
        j = go2idx[t]
        label_matrix[i, j] = 1.0

train_df.to_pickle("train_df.pkl")
with open("go_terms.json", "w") as f:
    json.dump(all_terms, f)

with open("go2idx.json", "w") as f:
    json.dump(go2idx, f)
np.save("train_labels.npy", label_matrix)

test_rows = []
for entry_id, seq in test_seqs.items():
    test_rows.append({"EntryID": entry_id, "sequence": seq})
test_df = pd.DataFrame(test_rows)
test_df.to_pickle("test_df.pkl")

assert len(train_df) == label_matrix.shape[0]
assert label_matrix.shape[1] == len(all_terms)

print("예시 EntryID:", train_df.iloc[0]["EntryID"])
print("예시 terms:", train_df.iloc[0]["terms"])
print("예시 label vector non-zero idx:",
      np.where(label_matrix[0] > 0)[0][:10])

print("Saved: train_df.pkl, train_labels.npy, test_df.pkl")