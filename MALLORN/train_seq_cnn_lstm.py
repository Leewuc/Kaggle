import os, gc
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, accuracy_score

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from config import CONFIG
from data_utils import (set_seed, load_train_log_and_update_config, make_folds, load_all_train_lightcurves)

class SeqDataset(Dataset):
    def __init__(self, cont_feats, filt_idx, y, folds, target_fold=None):
        self.cont_feats = cont_feats
        self.filt_idx = filt_idx
        self.y = y
        self.folds = folds
        self.target_fold = target_fold

        if target_fold is None:
            self.indices = np.arange(len(y))
        else:
            self.indices = np.where(folds == target_fold)[0]
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        x_cont = self.cont_feats[real_idx]
        x_filt = self.filt_idx[real_idx]
        y = self.y[real_idx]

        return {
            "x_cont": torch.tensor(x_cont, dtype=torch.float32),
            "x_filt": torch.tensor(x_filt, dtype=torch.long),
            "y": torch.tensor(y, dtype=torch.long),
        }

class SeqCnnLstmModel(nn.Module):
    def __init__(
            self,
            num_filters: int,
            num_classes: int,
            d_model: int = 64,
            emb_dim: int = 8,
            cnn_channels: int = 64,
            cnn_kernel_size: int = 3,
            cnn_layers: int = 2,
            lstm_hidden_size: int = 128,
            lstm_num_layers: int = 1,
            bidirectional: bool = True,
            fc_hidden: int = 256,
            dropout: float = 0.3,
    ):
        super().__init__()
        self.emb = nn.Embedding(num_filters, emb_dim, padding_idx=0)
        cont_dim = 4
        in_dim = cont_dim + emb_dim
        self.input_proj = nn.Linear(in_dim, d_model)

        cnn_layers_list = []
        for i in range(cnn_layers):
            cnn_layers_list.append(
                nn.Conv1d(
                    in_channels=d_model if i == 0 else cnn_channels,
                    out_channels=cnn_channels,
                    kernel_size=cnn_kernel_size,
                    padding=cnn_kernel_size // 2,
                )
            )
            cnn_layers_list.append(nn.ReLU())
            cnn_layers_list.append(nn.BatchNorm1d(cnn_channels))
        self.cnn = nn.Sequential(*cnn_layers_list)

        self.lstm = nn.LSTM(
            input_size = cnn_channels,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_num_layers,
            batch_first=True,
            bidirectional=bidirectional,
        )
        lstm_out_dim = lstm_hidden_size * (2 if bidirectional else 1)

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Sequential(
            nn.Linear(lstm_out_dim, fc_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fc_hidden, num_classes),
        )
    
    def forward(self, x_cont, x_filt):
        emb = self.emb(x_filt)
        x = torch.cat([x_cont, emb], dim=-1)

        x = self.input_proj(x)

        x = x.permute(0, 2, 1)
        x = self.cnn(x)
        x = x.permute(0, 2, 1)

        out, _ = self.lstm(x)
        out, _ = torch.max(out, dim=1)
        out = self.dropout(out)

        logits = self.fc(out)
        return logits

def build_sequences_from_lc(
        config: Dict,
        train_lc: pd.DataFrame,
        df_log: pd.DataFrame,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    data_conf = config["data"]
    seq_conf = config["models"]["seq_cnn_lstm"]["input"]
    id_col = data_conf["id_col"]
    time_col = data_conf["time_col"]
    flux_col = data_conf["flux_col"]
    flux_err_col = data_conf["flux_err_col"]
    band_col = data_conf["band_col"]
    max_len = seq_conf["max_len"]

    unique_filters = sorted(train_lc[band_col].dropna().unique().tolist())
    filt2idx = {f: i + 1 for i, f in enumerate(unique_filters)}
    num_filters = len(filt2idx) + 1

    df_log_sorted = df_log.sort_values(id_col).reset_index(drop=True)
    obj_ids = df_log_sorted[id_col].values
    y = df_log_sorted["target_idx"].values
    folds = df_log_sorted["fold"].values

    N = len(obj_ids)
    cont_seqs = np.zeros((N, max_len, 4), dtype=np.float32)
    filt_seqs = np.zeros((N, max_len), dtype=np.int64)
    lc_group = train_lc.groupby(id_col)

    for i, oid in enumerate(obj_ids):
        if oid not in lc_group.groups:
            continue

        g = lc_group.get_group(oid).copy()
        g = g.sort_values(time_col)

        t0 = g[time_col].min()
        g["t_rel"] = g[time_col] - t0

        flux = g[flux_col].values.astype(np.float32)
        flux_err = g[flux_err_col].values.astype(np.float32)
        t_rel = g["t_rel"].values.astype(np.float32)

        flux_med = np.median(np.abs(flux)) + 1e-6
        flux_norm = flux / flux_med
        flux_err_norm = flux_err / flux_med
        snr = flux / (flux_err + 1e-6)

        filt_raw = g[band_col].values
        filt_idx = np.array([filt2idx.get(f, 0) for f in filt_raw], dtype=np.int64)

        L = len(g)
        if L >= max_len:
            sel = slice(0, max_len)
            cont_seq = np.stack(
                [t_rel[sel], flux_norm[sel], flux_err_norm[sel], snr[sel]], axis=1
            )
            filt_seq = filt_idx[sel]
        else:
            cont_seq = np.zeros((max_len, 4), dtype=np.float32)
            filt_seq = np.zeros((max_len,), dtype=np.int64)

            cont_seq[:L, 0] = t_rel
            cont_seq[:L, 1] = flux_norm
            cont_seq[:L, 2] = flux_err_norm
            cont_seq[:L, 3] = snr
            filt_seq[:L] = filt_idx
        
        cont_seqs[i] = cont_seq
        filt_seqs[i] = filt_seq
    
    print(f"[INFO] Built sequences: cont_seqs={cont_seqs.shape}, filt_seqs={filt_seqs.shape}")
    print(f"[INFO] num_filters (including padding)= {num_filters}")
    return cont_seqs, filt_seqs, y, folds, num_filters

def train_one_fold(
        config: Dict,
        fold: int,
        cont_seqs: np.ndarray,
        filt_seqs: np.ndarray,
        y: np.ndarray,
        folds: np.ndarray,
        num_filters: int,
) -> Tuple[np.ndarray, nn.Module]:
    device = config["general"]["device"]
    seq_conf = config["models"]["seq_cnn_lstm"]
    model_conf = seq_conf["model"]
    train_conf = seq_conf["train"]

    batch_size = train_conf["batch_size"]
    num_epochs = train_conf["num_epochs"]
    lr = train_conf["learning_rate"]
    weight_decay = train_conf["weight_decay"]

    num_classes = model_conf["num_classes"]
    assert num_classes is not None

    train_ds = SeqDataset(cont_seqs, filt_seqs, y, folds, target_fold=None)
    val_ds = SeqDataset(cont_seqs, filt_seqs, y, folds, target_fold=fold)

    trn_idx = np.where(folds != fold)[0]
    val_idx = np.where(folds == fold)[0]

    train_idx_mask = np.isin(train_ds.indices, trn_idx)
    train_indices = train_ds.indices[train_idx_mask]

    train_ds.indices = train_indices

    train_loader = DataLoader(
        train_ds,
        batch_size = batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    model = SeqCnnLstmModel(
        num_filters=num_filters,
        num_classes=num_classes,
        d_model=model_conf["d_model"],
        emb_dim=8,  
        cnn_channels=model_conf["cnn_channels"],
        cnn_kernel_size=model_conf["cnn_kernel_size"],
        cnn_layers=model_conf["cnn_layers"],
        lstm_hidden_size=model_conf["lstm_hidden_size"],
        lstm_num_layers=model_conf["lstm_num_layers"],
        bidirectional=model_conf["bidirectional"],
        fc_hidden=model_conf["fc_hidden"],
        dropout=model_conf["dropout"],
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    oof_pred_fold = np.zeros((len(val_idx), num_classes), dtype=np.float32)
    best_val_loss = np.inf
    best_state_dict = None

    for epoch in range(1, num_epochs + 1):
        model.train()
        train_loss = 0.0

        for batch in train_loader:
            x_cont = batch["x_cont"].to(device)
            x_filt = batch["x_filt"].to(device)
            labels = batch["y"].to(device)

            optimizer.zero_grad()
            logits = model(x_cont, x_filt)
            loss = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=train_conf["gradient_clip_val"])
            optimizer.step()

            train_loss += loss.item() * x_cont.size(0)
        
        train_loss /= len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        all_logits = []

        with torch.no_grad():
            for batch in val_loader:
                x_cont = batch["x_cont"].to(device)
                x_filt = batch["x_filt"].to(device)
                labels = batch["y"].to(device)

                logits = model(x_cont, x_filt)
                loss = criterion(logits, labels)

                val_loss += loss.item() * x_cont.size(0)
                all_logits.append(logits.softmax(dim=1).cpu().numpy())
        
        val_loss /= len(val_loader.dataset)
        val_pred = np.concatenate(all_logits, axis=0)
        oof_pred_fold[:] = val_pred

        print(
            f"[FOLD {fold}] Epoch {epoch}/{num_epochs} "
            f"train_loss={train_loss:.4f} val_loss={val_loss:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state_dict = model.state_dict()
    
    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)
    return oof_pred_fold, val_idx, model

def prepare_seq_data(config: Dict):
    set_seed(config["general"]["seed"])
    df_log, cls2idx, idx2cls, config = load_train_log_and_update_config(config)
    df_log = make_folds(config, df_log)
    train_lc = load_all_train_lightcurves(config)

    cont_seqs, filt_seqs, y, folds, num_filers = build_sequences_from_lc(config, train_lc, df_log)
    return cont_seqs, filt_seqs, y, folds, num_filers, df_log, config

def train_seq_cnn_lstm_cv(config: Dict):
    from sklearn.metrics import log_loss, accuracy_score
    (
        cont_seqs,
        filt_seqs,
        y,
        folds,
        num_filters,
        df_log,
        config,
    ) = prepare_seq_data(config)

    n_splits = config["cv"]["n_splits"]
    num_classes = config["models"]["seq_cnn_lstm"]["model"]["num_classes"]
    assert num_classes is not None

    oof_pred = np.zeros((len(y), num_classes), dtype=np.float32)
    models = {}

    for fold in range(n_splits):
        print(f"\n===================SEQ_CNN_LSTM FOLD {fold}===============================")
        oof_pred_fold, val_idx, model = train_one_fold(
            config=config,
            fold=fold,
            cont_seqs=cont_seqs,
            filt_seqs=filt_seqs,
            y=y,
            folds=folds,
            num_filters=num_filters,
        )

        oof_pred[val_idx] = oof_pred_fold
        models[fold] = model
        gc.collect()
    
    cv_logloss = log_loss(y, oof_pred)
    cv_acc = accuracy_score(y, oof_pred.argmax(axis=1))
    print(f"\n[SEQ_CNN_LSTM][CV] logloss: {cv_logloss:.6f}, acc: {cv_acc:.6f}")

    seq_oof_path = config["data"]["seq_oof_path"]
    os.makedirs(os.path.dirname(seq_oof_path), exist_ok=True)
    np.save(seq_oof_path, oof_pred)
    print(f"[INFO] Saved seq OOF to {seq_oof_path}")

    return oof_pred, models

def main():
    oof_pred, models = train_seq_cnn_lstm_cv(CONFIG)

if __name__ == "__main__":
    main()