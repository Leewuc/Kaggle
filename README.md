# Kaggle

이 저장소는 Kaggle 대회에서 사용했던 데이터와 노트북을 모아 놓은 곳입니다. 각 폴더에는 대회별 데이터 파일과 실험 노트북이 들어 있어 다른 분들도 참고할 수 있습니다.

## 폴더 구성

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

노트북 파일은 Kaggle 환경에서 실행하던 형태 그대로 제공됩니다. 필요한 경우 데이터 압축을 풀고 실행하면 됩니다. 대회 참가 경험을 기록하고 공유하기 위한 용도로 구성되어 있으니 학습이나 참고에 활용하시기 바랍니다.
