import os, random, math
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import torchvision.transforms as T
import torchvision.models as models

from sklearn.model_selection import train_test_split

from tqdm.auto import tqdm

DATA_DIR = Path("/data/AskFake/Image/CSIRO")
TRAIN_CSV = DATA_DIR / "train.csv"
OUTPUT_DIR = DATA_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

IMAGE_ROOT = DATA_DIR

RANDOM_SEED = 42
BATCH_SIZE = 32
NUM_EPOCHS = 15
LR = 1e-3
IMAGE_SIZE = 224
NUM_WORKERS = 4

TARGET_NAMES = [
    "Dry_Clover_g",
    "Dry_Dead_g",
    "Dry_Green_g",
    "Dry_Total_g",
    "GDM_g",
]
N_TARGETS = len(TARGET_NAMES)
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(RANDOM_SEED)

def load_and_pivot_train(train_csv_path: Path):
    df = pd.read_csv(train_csv_path)
    df = df[["image_path", "target_name", "target"]]
    pivot = df.pivot_table(
        index="image_path",
        columns="target_name",
        values="target"
    )

    pivot = pivot.dropna(subset=TARGET_NAMES)
    pivot = pivot[TARGET_NAMES]
    pivot = pivot.reset_index()
    return pivot

class BiomassDataset(Dataset):
    def __init__(self, df, image_root: Path, transforms=None, log_target=True):
        self.df = df.reset_index(drop=True)
        self.image_root = image_root
        self.transforms = transforms
        self.log_target = log_target
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = self.image_root / row["image_path"]
        image = Image.open(img_path).convert("RGB")

        if self.transforms:
            image = self.transforms(image)
        
        targets = row[TARGET_NAMES].values.astype("float32")

        if self.log_target:
            targets = np.log1p(targets)
        
        targets = torch.from_numpy(targets)
        return image, targets
    
def get_transforms(image_size=224):
    train_tfms = T.Compose([
        T.Resize((image_size + 32, image_size + 32)),
        T.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
        T.RandomHorizontalFlip(),
        T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
        T.ToTensor(),
        T.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    val_tfms = T.Compose([
        T.Resize((image_size + 32, image_size + 32)),
        T.CenterCrop(image_size),
        T.ToTensor(),
        T.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    return train_tfms, val_tfms


# --------------------------
# 5. 모델 정의
# --------------------------
class BiomassModel(nn.Module):
    def __init__(self, n_targets=5):
        super().__init__()
        # ResNet18 backbone
        try:
            backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        except Exception:
            backbone = models.resnet18(pretrained=True)

        in_features = backbone.fc.in_features
        backbone.fc = nn.Identity()

        self.backbone = backbone
        self.head = nn.Linear(in_features, n_targets)

    def forward(self, x):
        feat = self.backbone(x)
        out = self.head(feat)
        return out


# --------------------------
# 6. R² 메트릭 (log-space 기준)
# --------------------------
def csiro_weighted_r2_logspace(y_true, y_pred):
    """
    y_true, y_pred : [N, 5]  (이미 log1p가 적용된 타깃 스페이스)
    논문/대회에서 쓰는 가중 R^2를 그대로 구현
    """
    # 타깃 순서가 TARGET_NAMES와 정확히 일치한다고 가정
    weights = torch.tensor(
        [0.1, 0.1, 0.1, 0.2, 0.5],  # Dry_Clover, Dry_Dead, Dry_Green, Dry_Total, GDM 순서에 맞춰줄 것
        device=y_true.device,
        dtype=y_true.dtype,
    )

    # 혹시 순서가 다르면 여기서 재정렬 필요하지만,
    # 우리는 TARGET_NAMES = ["Dry_Clover_g", "Dry_Dead_g", "Dry_Green_g", "Dry_Total_g", "GDM_g"]
    # 로 맞춰놨으니 weights도 이 순서로 맞추면 됨.
    eps = 1e-9

    y_true_mean = torch.mean(y_true, dim=0, keepdim=True)  # [1, 5]
    ss_tot = torch.sum((y_true - y_true_mean) ** 2, dim=0)  # [5]
    ss_res = torch.sum((y_true - y_pred) ** 2, dim=0)       # [5]

    r2_per_target = 1.0 - ss_res / (ss_tot + eps)           # [5]
    # 가중합 (가중치는 합이 1이라 weighted average와 같다)
    weighted_r2 = torch.sum(r2_per_target * weights).item()

    return weighted_r2, r2_per_target.detach().cpu().numpy()



# --------------------------
# 7. 학습 루프
# --------------------------
def train_one_epoch(model, loader, optimizer, device, loss_fn):
    model.train()
    running_loss = 0.0

    for images, targets in tqdm(loader, desc="Train", leave=False):
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

    r2_mean, r2_per_target = csiro_weighted_r2_logspace(all_targets, all_preds)

    return epoch_loss, r2_mean, r2_per_target


# --------------------------
# 8. main()
# --------------------------
def main():
    # 시드 고정 (혹시 위에서 이미 호출했다면 중복 호출되어도 문제는 없음)
    set_seed(RANDOM_SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

    # 1) 데이터 로딩 & pivot
    pivot_df = load_and_pivot_train(TRAIN_CSV)
    print("Pivoted train size:", len(pivot_df))

    # 2) train/val split
    train_df, val_df = train_test_split(
        pivot_df,
        test_size=0.1,
        random_state=RANDOM_SEED,
        shuffle=True,
    )

    print("Train size:", len(train_df), "Val size:", len(val_df))

    # 3) transforms
    train_tfms, val_tfms = get_transforms(IMAGE_SIZE)

    # 4) Dataset / DataLoader
    train_ds = BiomassDataset(train_df, IMAGE_ROOT, transforms=train_tfms, log_target=True)
    val_ds = BiomassDataset(val_df, IMAGE_ROOT, transforms=val_tfms, log_target=True)

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    # 5) 모델 / 옵티마이저 / 손실함수
    model = BiomassModel(n_targets=N_TARGETS).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    loss_fn = nn.MSELoss()

    best_val_r2 = -1e9
    best_model_path = OUTPUT_DIR / "best_model.pth"

    # 각 epoch 로그를 담을 리스트 (나중에 CSV로 저장)
    history = []

    for epoch in range(1, NUM_EPOCHS + 1):
        print(f"\nEpoch {epoch}/{NUM_EPOCHS}")

        # ---- Train ----
        train_loss = train_one_epoch(model, train_loader, optimizer, device, loss_fn)
        print(f"  Train loss: {train_loss:.6f}")

        # ---- Validation ----
        val_loss, val_w_r2, val_r2_per_target = validate_one_epoch(
            model, val_loader, device, loss_fn
        )
        print(f"  Val loss: {val_loss:.6f}")
        print(f"  Val CSIRO metric (weighted log-R2): {val_w_r2:.6f}")
        for name, r2v in zip(TARGET_NAMES, val_r2_per_target):
            print(f"    {name}: {r2v:.6f}")

        # ---- 1) 매 epoch마다 체크포인트 저장 ----
        epoch_ckpt_path = OUTPUT_DIR / f"model_epoch_{epoch:03d}.pth"
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_weighted_r2": val_w_r2,
                "val_r2_per_target": val_r2_per_target,
                "target_names": TARGET_NAMES,
            },
            epoch_ckpt_path,
        )
        print(f"  Saved epoch checkpoint to {epoch_ckpt_path}")

        # ---- 2) best model 갱신 시 별도 저장 ----
        if val_w_r2 > best_val_r2:
            best_val_r2 = val_w_r2
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "best_val_weighted_r2": best_val_r2,
                    "target_names": TARGET_NAMES,
                },
                best_model_path,
            )
            print(f"  >> New best model saved to {best_model_path} (score={best_val_r2:.6f})")

        # ---- 3) 로그를 history 리스트에 누적 ----
        row = {
            "epoch": epoch,
            "train_loss": float(train_loss),
            "val_loss": float(val_loss),
            "val_weighted_r2": float(val_w_r2),
        }
        for name, r2v in zip(TARGET_NAMES, val_r2_per_target):
            # 컬럼 이름 예: r2_Dry_Clover_g
            row[f"r2_{name}"] = float(r2v)
        history.append(row)

        # ---- 4) 매 epoch 끝날 때마다 CSV로 저장 (중간에 끊겨도 로그 남게) ----
        log_df = pd.DataFrame(history)
        log_path = OUTPUT_DIR / "training_log.csv"
        log_df.to_csv(log_path, index=False)
        print(f"  Updated training log CSV at {log_path}")

    print("\nTraining finished.")
    print("Best val weighted R2:", best_val_r2)



if __name__ == "__main__":
    main()

