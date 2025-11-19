import torch
from torch.utils.data import DataLoader
from dataset import CafaDataset, AMINO_ACIDS
from model import CafaCNN
from tqdm.auto import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_ds = CafaDataset("train_df.pkl", "train_labels.npy", max_len=1024)
train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=4)

num_labels = train_ds.labels.shape[1]
vocab_size = len(AMINO_ACIDS) + 2

model = CafaCNN(vocab_size=vocab_size, num_labels=num_labels).to(device)
criterion = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(3):
    model.train()
    total_loss = 0.0
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
    for i, (x,y) in enumerate(pbar):
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        if (i + 1) % 100 == 0:
            print(f"Epoch {epoch+1} Step {i+1} Loss {total_loss / (i+1):.4f}")

torch.save(model.state_dict(), "cafa_cnn_baseline.pt")
print("Saved model. to cafa_cnn_baseline.pt")