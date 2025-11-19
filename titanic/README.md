# Titanic - Machine Learning from Disaster

## 대회 소개
Kaggle Titanic 대회는 승객의 신상/선내 정보를 기반으로 생존 여부를 예측하는 이진 분류 문제입니다. 평가 지표는 정확도이며, 초보자가 머신러닝 파이프라인을 연습하기 좋은 대표적인 대회입니다.

## 폴더 구성
- `train.csv`, `test.csv`, `gender_submission.csv` : 원본 데이터 및 제출 예시
- `titanic.ipynb` : 기초 EDA와 베이스라인 모델(Logistic Regression) 구축 노트북
- `titanic-ml.ipynb` : 특성 공학과 다양한 모델(XGBoost, RandomForest 등) 실험 노트북

## 접근 방법
1. **EDA** : 성별, 선실 등급, 승선 항구에 따른 생존률을 시각화했습니다.
2. **데이터 처리** : 이름/티켓에서 타이틀 추출, 가족 규모 파생 변수, Age/SibSp/Embarked 결측 보간을 수행했습니다.
3. **모델링** : 기본 Logistic Regression → RandomForest → XGBoost 순으로 성능을 개선했고, 최종 제출은 Soft Voting Ensemble을 사용했습니다.
4. **평가** : StratifiedKFold(5 folds)로 검증한 뒤, 교차 검증 평균 정확도를 리더보드 점수와 비교했습니다.

## 실행 방법
두 노트북 모두 Kaggle Notebook 환경에서 재현 가능하며, 로컬 실행 시에는 `pip install scikit-learn xgboost seaborn`을 권장합니다. 데이터 파일은 같은 폴더에 두고 셀을 순서대로 실행하면 됩니다.
