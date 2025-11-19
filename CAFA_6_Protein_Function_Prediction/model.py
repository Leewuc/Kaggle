import torch
import torch.nn as nn

class CafaCNN(nn.Module):
    def __init__(self, vocab_size, num_labels, emb_dim=128, num_filters=256, kernel_sizes=(3,5,7), dropout=0.5):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)

        self.convs = nn.ModuleList([
            nn.Conv1d(emb_dim, num_filters, k)
            for k in kernel_sizes
        ])

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(num_filters * len(kernel_sizes), num_labels)
    
    def forward(self, x):
        emb = self.embedding(x)
        emb = emb.transpose(1,2)

        conv_outs = []
        for conv in self.convs:
            c = conv(emb)
            c = torch.relu(c)
            c = torch.max(c, dim=2).values
            conv_outs.append(c)
        
        h = torch.cat(conv_outs, dim=1)
        h = self.dropout(h)
        logits = self.fc(h)
        return logits