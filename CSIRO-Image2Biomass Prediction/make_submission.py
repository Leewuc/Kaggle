# make_submission.py
import os
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torchvision.models as models


DATA_DIR = Path("/data/AskFake/Image/CSIRO")
TEST_CSV = DATA_DIR / "test.csv"
IMAGE_ROOT = DATA_DIR

MODEL_PATH = Path("/data/AskFake/Image/CSIRO/outputs_0.1/best_model.pth")
SUBMISSION_PATH = Path("/data/AskFake/Image/CSIRO/sample_submission.csv")

IMAGE_SIZE = 224
BATCH_SIZE = 64
NUM_WORKERS = 4

TARGET_NAMES = [
    "Dry_Clover_g",
    "Dry_Dead_g",
    "Dry_Green_g",
    "Dry_Total_g",
    "GDM_g",
]
N_TARGETS = len(TARGET_NAMES)


class BiomassModel(nn.Module):
    def __init__(self, n_targets=5):
        super().__init__()
        try:
            backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        except Exception:
            backbone = models.resnet18(pretrained=False)

        in_features = backbone.fc.in_features
        backbone.fc = nn.Identity()

        self.backbone = backbone
        self.head = nn.Linear(in_features, n_targets)

    def forward(self, x):
        feat = self.backbone(x)
        out = self.head(feat)
        return out


def get_infer_transform(image_size=224):
    tfms = T.Compose([
        T.Resize((image_size + 32, image_size + 32)),
        T.CenterCrop(image_size),
        T.ToTensor(),
        T.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])
    return tfms


class TestImageDataset(Dataset):
    """
    image_path 단위로만 사용 (각 이미지에서 5개 타깃 모두 예측)
    """
    def __init__(self, image_paths, image_root: Path, transforms=None):
        self.image_paths = list(image_paths)
        self.image_root = image_root
        self.transforms = transforms

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_rel_path = self.image_paths[idx]
        img_path = self.image_root / img_rel_path

        image = Image.open(img_path).convert("RGB")

        if self.transforms:
            image = self.transforms(image)

        return image, img_rel_path


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # 1) 모델 로드
    checkpoint = torch.load(MODEL_PATH, map_location=device)
    model = BiomassModel(n_targets=N_TARGETS)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    # 2) test.csv 로드
    test_df = pd.read_csv(TEST_CSV)
    print("Test rows:", len(test_df))

    # image_path 목록 추출
    unique_image_paths = test_df["image_path"].unique()
    print("Unique test images:", len(unique_image_paths))

    # 3) Dataset / DataLoader
    tfms = get_infer_transform(IMAGE_SIZE)
    test_ds = TestImageDataset(unique_image_paths, IMAGE_ROOT, transforms=tfms)

    test_loader = DataLoader(
        test_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    # 4) 이미지별 예측값 저장 딕셔너리
    #   key: image_path (str), value: numpy array [5] (각 타깃의 log1p 예측값)
    image_to_pred_log = {}

    with torch.no_grad():
        for images, img_rel_paths in test_loader:
            images = images.to(device)
            outputs = model(images)  # [B, 5] (log1p 타깃 스케일)

            outputs = outputs.cpu().numpy()
            for img_rel_path, pred_log in zip(img_rel_paths, outputs):
                image_to_pred_log[img_rel_path] = pred_log

    # 5) log1p 역변환 후, target_name에 맞게 값 매핑
    preds = []
    for _, row in test_df.iterrows():
        sample_id = row["sample_id"]
        img_rel_path = row["image_path"]
        target_name = row["target_name"]

        log_preds = image_to_pred_log[img_rel_path]  # [5]
        # target_name에 해당하는 인덱스 찾기
        idx = TARGET_NAMES.index(target_name)

        # log1p 역변환
        val = np.expm1(log_preds[idx])

        # 음수 방지 (아주 약간의 수치 오차 대비)
        val = float(max(val, 0.0))

        preds.append((sample_id, val))

    # 6) submission.csv 생성
    sub_df = pd.DataFrame(preds, columns=["sample_id", "target"])
    sub_df = sub_df.sort_values("sample_id")  # Kaggle sample_submission과 정렬 맞추는 용도
    sub_df.to_csv(SUBMISSION_PATH, index=False)

    print("Submission saved to:", SUBMISSION_PATH)


if __name__ == "__main__":
    main()
