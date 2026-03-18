# Kaggle GitHub Bundle

이 폴더는 GitHub에 그대로 올린 뒤, Kaggle Notebook에서 바로 가져가서 실험할 수 있게 정리한 묶음입니다.

## 포함 파일

- `kaggle_kcelectra_pipeline.ipynb`
  - Kaggle에서 바로 실행 가능한 단일 노트북
  - 기본 전략: `KcELECTRA + 3-fold`
  - 옵션 전략: `K-MHaS binary pre-finetune -> Kaggle fine-tune`
- `scripts/`
  - 로컬 재현용 핵심 스크립트 모음
  - 노트북과 같은 방향의 실험을 코드로도 유지할 수 있게 넣어둠

## 권장 업로드 방식

GitHub에는 이 폴더만 올리면 됩니다.

추천 구조:

```text
your-repo/
  kaggle_github_bundle/
    README.md
    kaggle_kcelectra_pipeline.ipynb
    scripts/
```

## Kaggle에서 필요한 Input

노트북은 아래 두 input을 기준으로 작성되어 있습니다.

1. competition dataset
2. `K-MHaS` processed dataset

노트북 상단에서 이 경로를 실제 Kaggle input 이름으로 바꿔야 합니다.

```python
COMP_DIR = Path('/kaggle/input/your-competition-dataset')
KMHAS_DIR = Path('/kaggle/input/your-kmhas-processed-dataset')
```

중요:

- 현재 에러가 났던 이유는 `COMP_DIR`가 실제 Kaggle input 폴더명과 달랐기 때문입니다.
- 먼저 Kaggle에서 `/kaggle/input` 아래 폴더명을 확인한 뒤 수정해야 합니다.

확인용 셀:

```python
!ls -la /kaggle/input
!find /kaggle/input -maxdepth 2 -type f | head -100
```

## Kaggle GPU 설정

반드시 Kaggle Notebook 설정에서 `Accelerator = GPU`로 바꿔야 합니다.

확인 셀:

```python
import torch
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "no gpu")
```

`device: cpu`가 나오면 현재 설정으로는 사실상 학습이 너무 느립니다.

## 노트북에서 먼저 수정할 값

상단 설정 셀에서 우선 이 값들만 조정하면 됩니다.

- `MODEL_ID`
- `USE_KMHAS_PRETRAIN`
- `USE_TITLE`
- `FOLDS`
- `EPOCHS`
- `BATCH_SIZE`
- `EVAL_BATCH_SIZE`
- `MAX_LENGTH`
- `GRAD_ACCUM`
- `USE_FP16`
- `USE_BF16`

## 현재 추천 시작 설정

```python
MODEL_ID = 'beomi/KcELECTRA-base-v2022'
USE_KMHAS_PRETRAIN = True
USE_TITLE = True

FOLDS = 3
EPOCHS = 4
BATCH_SIZE = 8
EVAL_BATCH_SIZE = 16
MAX_LENGTH = 128
GRAD_ACCUM = 1
USE_FP16 = True
USE_BF16 = False
```

## 출력 파일

노트북 실행이 끝나면 `/kaggle/working`에 아래 파일이 생성됩니다.

- `submission.csv`
- `submission_comments_label.csv`
- `submission_comments_label_numeric.csv`
- `metrics.json`

Kaggle 제출에는 보통 `submission.csv`를 쓰면 됩니다.

## scripts 폴더 설명

- `runtime_profiles.py`
  - 로컬 실행용 런타임 프로파일 정의
- `transformer_baseline.py`
  - 단일 split transformer baseline
- `transformer_kfold_ensemble.py`
  - 현재 핵심 k-fold 학습 스크립트
- `pretrain_kmhas_binary.py`
  - `K-MHaS` binary pre-finetune
- `two_stage_hierarchical.py`
  - `none/toxic -> offensive/hate` 2-stage 실험
- `pseudo_label_self_training.py`
  - pseudo-label 실험용
- `augment_weak_korean_comments.py`
  - 약한 text augmentation 생성기
- `transformer_kfold_train_only_aug.py`
  - train fold에만 augmentation 적용하는 누수 없는 버전

## 현재 판단

지금까지 실험 기준으로는:

- `K-MHaS + KcELECTRA`는 유효
- 약한 text augmentation은 비추천
- seed/epoch/blend만으로는 한계가 있었음
- 다음 큰 전략은 `다른 backbone` 또는 `pseudo-label` 쪽이 더 유력

그래서 Kaggle에서는 우선:

1. `KcELECTRA + K-MHaS + USE_TITLE=True`
2. backbone 교체 실험
3. 필요하면 pseudo-label

순서로 가는 걸 추천합니다.
