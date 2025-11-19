# Kaggle

이 저장소는 Kaggle에서 참가한 대회의 데이터와 노트북을 모아 놓은 곳입니다. 폴더 이름이 곧 참가한 Competition 이름이며, 각 폴더의 README에서 대회 개요와 실험 노트북을 간략히 정리했습니다.

## 폴더 구성

- **CAFA_6_Protein_Function_Prediction**
  - `prepare_data.py`, `dataset.py` : 유전자 서열과 Gene Ontology 라벨을 불러와 전처리
  - `train_baseline.py`, `model.py` : 1D CNN 기반 베이스라인 학습 코드
  - `cafa6_1DCNN.ipynb` : 실험 설정과 결과 정리
  - `predcit_make_submission.py` : 제출 파일 생성 스크립트

- **CSIRO-Image2Biomass Prediction**
  - `csiro-rent18.ipynb` : 위성 이미지와 현장 바이오매스 측정을 활용한 회귀 실험 노트북

- **Digit_Recognizer**
  - `train.csv.zip`, `test.csv.zip` : 손글씨 숫자 이미지 데이터를 압축한 파일
  - `sample_submission.csv` : 제출 파일 형식 예시
  - `PCA_3D_Digit.ipynb` : PCA를 사용한 3차원 시각화 노트북

- **House_Price**
  - `train (1).csv`, `test (1).csv` : 집 값 예측을 위한 학습/평가 데이터
  - `sample_submission.csv` : 제출 파일 예시
  - `data_description.txt` : 각 특성에 대한 설명
  - `House_Price_TFDF.ipynb` : TensorFlow Decision Forests 모델 실습 노트북

- **titanic**
  - `train.csv`, `test.csv`, `gender_submission.csv` : 타이타닉 생존 예측용 데이터
  - `titanic.ipynb`, `titanic-ml.ipynb` : EDA와 모델링 과정을 담은 노트북

## 사용 방법

각 Competition 폴더의 README에 대회 설명, 사용한 특징량 및 모델 접근법을 정리했습니다. 노트북 파일은 Kaggle 환경에서 실행하던 형태 그대로 제공되므로 데이터 압축을 풀고 실행하면 됩니다. 대회 참가 경험을 기록하고 공유하기 위한 용도로 구성되어 있으니 학습이나 참고에 활용하시기 바랍니다.
