# House Price - Advanced Regression Techniques

## 대회 소개
Kaggle House Price 대회는 에임즈(Ames) 주택 데이터 79개 피처로 주택 판매 가격을 예측하는 회귀 문제입니다. 로그 RMSE를 최소화하는 것이 목표이며, 결측치 처리와 피처 엔지니어링이 핵심입니다.

## 폴더 구성
- `train (1).csv`, `test (1).csv` : 학습/평가용 구조화 데이터
- `sample_submission.csv` : 제출 형식 예시
- `data_description.txt` : 각 피처에 대한 상세 설명
- `House_Price_TFDF.ipynb` : TensorFlow Decision Forests 모델 실험 노트북

## 접근 방법
1. **데이터 클리닝** : Missing indicator 추가, 로그 변환 대상(GrLivArea 등) 선정, 희귀 범주 통합을 수행했습니다.
2. **특징 엔지니어링** : TotalSF, Age(YearSold - YearBuilt) 등 도메인 피처를 파생했습니다.
3. **모델링** : TF Decision Forests Gradient Boosted Trees를 주력 모델로 사용하고, Kaggle leaderboard 제출은 GBT + Lasso Ensemble을 사용했습니다.
4. **검증 전략** : StratifiedKFold(층화 기준: OverallQual)을 적용해 분포 차이를 완화했습니다.

## 실행 방법
노트북을 열어 순차 실행하면 EDA → 전처리 → TFDF 학습 → 제출 파일 생성 순으로 재현됩니다. 로컬에서는 `pip install tensorflow-decision-forests` 설치가 필요합니다.
