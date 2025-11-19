# CSIRO Image2Biomass Prediction

## 대회 소개
CSIRO Image2Biomass는 호주 전역의 위성/항공 이미지와 현장 조사 데이터를 이용해 지상 바이오매스를 예측하는 회귀 대회입니다. 각 샘플은 여러 스펙트럼 밴드와 관측 지점을 포함하며, 목표는 Root Mean Squared Error(RMSE)를 최소화하는 것입니다.

## 폴더 구성
- `csiro-rent18.ipynb` : EDA, 피처 엔지니어링, LightGBM/TabNet 비교 실험을 포함한 통합 노트북입니다.

## 접근 방법
1. **피처 엔지니어링** : NDVI, EVI 등 식생 지수와 그림자 대비 지수를 추가로 계산했습니다.
2. **모델링** : Kaggle Notebook 환경에서 LightGBM을 기본 모델로 사용하고, TabNet과의 Blending으로 스코어를 개선했습니다.
3. **검증 전략** : 지리적 편향을 줄이기 위해 Region 기반 GroupKFold(5 folds)를 적용했습니다.
4. **제출** : Fold별 예측을 평균해 최종 CSV를 생성했습니다.

## 실행 방법
노트북을 열어 상단 셀부터 순서대로 실행하면 데이터 로딩 → 전처리 → 학습 → 제출 파일 생성까지 재현할 수 있습니다. 외부 의존성은 Kaggle 기본 환경에서 제공되지만, 로컬 실행 시에는 `pip install lightgbm pytorch-tabnet`을 미리 수행하면 됩니다.
