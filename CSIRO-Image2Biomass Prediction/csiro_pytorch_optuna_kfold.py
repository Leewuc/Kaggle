import os, random
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.models as tv_models
import torchvision.transforms as T
from sklearn.model_selection import KFold, train_test_split

from tqdm.auto import tqdm
import optuna

DATA_DIR = Path("/data/AskFake/Image/CSIRO")
TRAIN_CSV = DATA_DIR / "train.csv"
IMAGE_ROOT = DATA_DIR
OUTPUT_DIR = DATA_DIR / "optuna_kfold_outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = 42
N_SPLITS = 5              # K-Fold 개수
EPOCHS_PER_TRIAL = 5      # Optuna trial당 epoch 수 (빠른 비교용)
NUM_EPOCHS_FINAL = 25     # 최종 full-train epoch 수
NUM_WORKERS = 4

# CSIRO 공식 metric 기준 target 순서 & 가중치
# Dry_Green_g (0.1), Dry_Dead_g (0.1), Dry_Clover_g (0.1), GDM_g (0.2), Dry_Total_g (0.5)
TARGET_NAMES = [
    "Dry_Green_g",
    "Dry_Dead_g",
    "Dry_Clover_g",
    "GDM_g",
    "Dry_Total_g",
]
N_TARGETS = len(TARGET_NAMES)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def set_seed(seed: int=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(RANDOM_SEED)

def load_and_pivot_train(train_csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(train_csv_path)
    df = df[["image_path", "target_name", "target"]]

    pivot = df.pivot_table(
        index="image_path",
        columns="target_name",
        values="target"
    )

    pivot = pivot[TARGET_NAMES]
    pivot = pivot.dropna(subset=TARGET_NAMES)
    pivot = pivot.reset_index()
    return pivot

pivot_df = load_and_pivot_train(TRAIN_CSV)
print("Pivoted train size: ", len(pivot_df))
print("Columns: ", pivot_df.columns.tolist())

class BiomassDataset(Dataset):
    def __init__(self, df: pd.DataFrame, image_root: Path, transforms=None, log_target: bool = True):
        self.df = df.reset_index(drop=True)
        self.image_root = image_root
        self.transforms = transforms
        self.log_target = log_target

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        img_path = self.image_root / row["image_path"]
        image = Image.open(img_path).convert("RGB")

        if self.transforms is not None:
            image = self.transforms(image)

        targets = row[TARGET_NAMES].values.astype("float32")
        if self.log_target:
            targets = np.log1p(targets)
        targets = torch.from_numpy(targets)

        return image, targets

def get_transforms(image_size: int):
    train_tfms = T.Compose([
        T.Resize((image_size + 32, image_size + 32)),
        T.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
        T.RandomHorizontalFlip(),
        T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
        T.ToTensor(),
        T.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225]),
    ])

    val_tfms = T.Compose([
        T.Resize((image_size + 32, image_size + 32)),
        T.CenterCrop(image_size),
        T.ToTensor(),
        T.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225]),
    ])

    return train_tfms, val_tfms

def create_dataloaders(train_df, val_df, image_size, batch_size):
    train_tfms, val_tfms = get_transforms(image_size)
    train_ds = BiomassDataset(train_df, IMAGE_ROOT, transforms=train_tfms, log_target=True)
    val_ds = BiomassDataset(val_df, IMAGE_ROOT, transforms=val_tfms, log_target=True)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    return train_loader, val_loader

class BiomassModel(nn.Module):
    def __init__(self, n_targets: int = 5, backbone_name: str = "efficientnet_b0", dropout: float=0.0):
        super().__init__()
        self.backbone_name = backbone_name

        if backbone_name == "resnet34":
            try:
                from torchvision.models import ResNet34_Weights
                backbone = tv_models.resnet34(weights = ResNet34_Weights.IMAGENET1K_V1)
            except Exception:
                backbone = tv_models.resnet34(pretrained=True)

            in_features = backbone.fc.in_features
            backbone.fc = nn.Identity()
        
        elif backbone_name == "efficientnet_b0":
            try:
                from torchvision.models import EfficientNet_B0_Weights
                backbone = tv_models.efficientnet_b0(
                    weights = EfficientNet_B0_Weights.IMAGENET1K_V1
                )
            except Exception:
                backbone = tv_models.efficientnet_b0(pretrained=True)
            
            in_features = backbone.classifier[1].in_features
            backbone.classifier = nn.Identity()
        
        else:
            raise ValueError(f"Unsupported backbone: {backbone_name}")
        
        self.backbone = backbone
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(in_features, n_targets)
    
    def forward(self, x):
        feat = self.backbone(x)
        feat = self.dropout(x)
        out = self.head(feat)
        return out

def csiro_weighted_r2_logspace(y_true: torch.Tensor, y_pred: torch.Tensor):
    weights = torch.tensor([0.1,0.1,0.1,0.2,0.5], device=y_true.device, dtype=y_true.dtype)
    eps = 1e-9
    y_true_mean = torch.mean(y_true, dim=0, keepdim=True)
    ss_tot = torch.sum((y_true - y_true_mean) ** 2, dim=0)
    ss_res = torch.sum((y_true - y_pred) ** 2, dim=0)

    r2_per_target = 1.0 - ss_res / (ss_tot + eps)
    weighted_r2 = torch.sum(r2_per_target * weights).item()
    return weighted_r2, r2_per_target.detach().cpu().numpy()

def train_one_epoch(model, loader, optimizer, device, loss_fn):
    model.train()
    running_loss = 0.0

    for images, targets in tqdm(loader, desc="Train", leaver=False):
        images = images.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        preds = model(images)
        loss = loss_fn(preds, targets)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)

    epoch_loss = running_loss / len(loader.dataset)
    return epoch_loss

@torch.no_grad()
def validate_one_epoch(model, loader, device, loss_fn):
    model.eval()
    running_loss = 0.0
    all_targets = []
    all_preds = []

    for images, targets in tqdm(loader, desc="Val", leave=False):
        images = images.to(device)
        targets = targets.to(device)

        preds = model(images)
        loss = loss_fn(preds, targets)

        running_loss += loss.item() * images.size(0)
        all_targets.append(targets.cpu())
        all_preds.append(preds.cpu())

    epoch_loss = running_loss / len(loader.dataset)
    all_targets = torch.cat(all_targets, dim=0)
    all_preds = torch.cat(all_preds, dim=0)

    weighted_r2, r2_per_target = csiro_weighted_r2_logspace(all_targets, all_preds)
    return epoch_loss, weighted_r2, r2_per_target

def objective(trial: optuna.Trial) -> float:
    set_seed(RANDOM_SEED)
    backbone_name = trial.suggest_categorical("backbone", ["resnet34", "efficientnet_b0"])
    lr = trial.suggest_float("lr", 1e-5, 5e-3, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)
    dropout = trial.suggest_float("dropout", 0.0, 0.5)
    image_size = trial.suggest_categorical("image_size", [224,256,288])
    batch_size = trial.suggest_categorical("batch_size", [16,32])

    print(f"\n[Trial {trial.number}] backbone={backbone_name}, lr={lr:.2e}, "
        f"wd={weight_decay:.2e}, dropout={dropout:.2f}, img={image_size}, bs={batch_size}")
    
    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_SEED)
    fold_scores = []
    step_idx = 0

    for fold, (train_idx, val_idx) in enumerate(kf.split(pivot_df), 1):
        print(f"\n  ----- Fold {fold}/{N_SPLITS} -----")

        train_df = pivot_df.iloc[train_idx].reset_index(drop=True)
        val_df = pivot_df.iloc[val_idx].reset_index(drop=True)

        train_loader, val_loader = create_dataloaders(
            train_df = train_df,
            val_df = val_df,
            image_size = image_size,
            batch_size = batch_size,
        )

        model = BiomassModel(
            n_targets = N_TARGETS,
            backbone_name = backbone_name,
            dropout=dropout,
        ).to(device)

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )
        loss_fn = nn.MSELoss()
        best_fold_score = -1e9

        for epoch in range(1, EPOCHS_PER_TRIAL + 1):
            step_idx += 1
            print(f"    [Fold {fold}] Epoch {epoch}/{EPOCHS_PER_TRIAL}")
            train_loss = train_one_epoch(model, train_loader, optimizer, device, loss_fn)
            val_loss, val_w_r2, _ = validate_one_epoch(model, val_loader, device, loss_fn)

            print(
                f"      train_loss={train_loss:.4f}, "
                f"val_loss={val_loss:.4f}, "
                f"wR2={val_w_r2:.4f}"
                )
            best_fold_score = max(best_fold_score, val_w_r2)
            trial.report(best_fold_score, step_idx)
            if trial.should_prune():
                raise optuna.TrialPruned()
        fold_scores.append(best_fold_score)
    mean_score = float(np.mean(fold_scores))
    print(f"\n[Trial {trial.number}] fold_scores={fold_scores}, mean={mean_score:.4f}")
    return mean_score

def train_full_with_best_params(best_params: dict):
    set_seed(RANDOM_SEED)
    backbone_name = best_params["backbone"]
    lr = best_params["lr"]
    weight_decay = best_params["weight_decay"]
    dropout = best_params["dropout"]
    image_size = best_params["image_size"]
    batch_size = best_params["batch_size"]

    print("\n[Final Train] Using best params:")
    for k, v in best_params.items():
        print(f"  {k}: {v}")

    train_df, val_df = train_test_split(
        pivot_df,
        test_size=0.1,
        random_state=RANDOM_SEED,
        shuffle=True,
    )
    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)

    train_loader, val_loader = create_dataloaders(
        train_df=train_df,
        val_df = val_df,
        image_size=image_size,
        batch_size=batch_size,
    )

    model = BiomassModel(
        n_targets=N_TARGETS,
        backbone_name=backbone_name,
        dropout=dropout,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_devat = weight_decay,
    )
    loss_fn = nn.MSELoss()
    best_score = -1e9
    best_model_path = OUTPUT_DIR / "best_model_full.pth"
    history = []

    for epoch in range(1, NUM_EPOCHS_FINAL + 1):
        print(f"\n[Final Train] Epoch {epoch}/{NUM_EPOCHS_FINAL}")
        train_loss = train_one_epoch(model, train_loader, optimizer, device ,loss_fn)
        val_loss, val_w_r2, r2_per_target = validate_one_epoch(model, val_loader, device, loss_fn)

        print(
            f"  train_loss={train_loss:.4f}, "
            f"val_loss={val_loss:.4f}, "
            f"wR2={val_w_r2:.4f}"
        )
        for name, r2v in zip(TARGET_NAMES, r2_per_target):
            print(f"    {name}: {r2v:.4f}")

        row = {
            "epoch": epoch,
            "train_loss": float(train_loss),
            "val_loss": float(val_loss),
            "val_weighted_r2": float(val_w_r2),
        }
        for name, r2v in zip(TARGET_NAMES, r2_per_target):
            row[f"r2_{name}"] = float(r2v)
        history.append(row)

        if val_w_r2 > best_score:
            best_score = val_w_r2
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "best_params": best_params,
                    "target_names": TARGET_NAMES,
                    "epoch": epoch,
                    "best_val_weighted_r2": best_score,
                },
                best_model_path,
            )
            print(f"  >> Saved best full model to {best_model_path} (wR2={best_score:.4f})")

        log_df = pd.DataFrame(history)
        log_path = OUTPUT_DIR / "final_training_log.csv"
        log_df.to_csv(log_path, index=False)
        print(f"  Updated training log CSV at {log_path}")

    print("\n[Final Train] Done. Best wR2:", best_score)

def main():
    study = optuna.create_study(
        direction = "maximize",
        study_name="csiro_biomass_kfold",
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=5,
            n_warmup_steps=5,
            interval_steps=5,
        ),
    )

    study.optimize(objective, n_trials=20)

    print("\n========== Optuna 결과 ==========")
    print("Best mean KFold wR2:", study.best_trial.value)
    print("Best params:")
    for k, v in study.best_trial.params.items():
        print(f"  {k}: {v}")

    best_params_path = OUTPUT_DIR / "best_params.json"
    pd.Series(study.best_trial.params).to_json(best_params_path)
    print(f"Best params saved to: {best_params_path}")

    # 3) best params로 최종 full-train
    train_full_with_best_params(study.best_trial.params)

if __name__ == "__main__":
    main()