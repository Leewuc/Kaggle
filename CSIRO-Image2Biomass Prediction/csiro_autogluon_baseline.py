from pathlib import Path
import pandas as pd
from autogluon.multimodal import MultiModalPredictor

DATA_DIR = Path("/data/AskFake/Image/CSIRO")
TRAIN_CSV = DATA_DIR / "train.csv"
OUTPUT_DIR = DATA_DIR / "autogluon_outputs"
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

df = pd.read_csv(TRAIN_CSV)

pivot = df.pivot_table(
    index=["image_path", "Sampling_Date", "State", "Species",
           "Pre_GSHH_NDVI", "Height_Ave_cm"],
    columns="target_name",
    values="target"
).reset_index()

# CSIRO의 5개 타깃
TARGET_NAMES = [
    "Dry_Clover_g",
    "Dry_Dead_g",
    "Dry_Green_g",
    "Dry_Total_g",
    "GDM_g",
]

pivot = pivot.dropna(subset=TARGET_NAMES).reset_index(drop=True)

print("Pivoted rows:", len(pivot))
print("Columns:", pivot.columns.tolist())

results = {}  # 각 타깃별 결과 저장

for target in TARGET_NAMES:
    print(f"\n====================")
    print(f"Training AutoGluon for target: {target}")
    print("====================")

    train_df = pivot.dropna(subset=[target]).reset_index(drop=True)

    # predictor 저장 경로
    save_dir = OUTPUT_DIR / f"ag_{target}"
    save_dir.mkdir(exist_ok=True)

    predictor = MultiModalPredictor(
        label=target,
        problem_type="regression",
        eval_metric="r2",
        path=save_dir,
    )

    # 이미지 경로는 자동 인식됨 (image_path 컬럼 이름 때문)
    predictor.fit(
        train_df,
        presets="best_quality",   # 빠른 baseline
        time_limit=1200                           # 10분 제한
    )

    metrics = predictor.evaluate(train_df)
    print(metrics)

    # R² 결과 저장
    best_score = metrics["r2"]
    results[target] = best_score

# --------------------------------------------------------
# 전체 타깃 결과 요약 출력
# --------------------------------------------------------
print("\n========== AutoGluon Baseline Results ==========")
for t, s in results.items():
    print(f"{t}: R² = {s:.4f}")

print("\nSaved models in:", OUTPUT_DIR)