# Digit Recognizer

## 대회 소개
Kaggle Digit Recognizer는 손글씨 숫자(MNIST) 이미지를 분류하는 입문용 컴퓨터 비전 대회입니다. 28×28 회색조 이미지에서 0~9 숫자를 예측하며, 평가 지표는 정확도(Accuracy)입니다.

## 폴더 구성
- `train.csv.zip`, `test.csv.zip` : 픽셀 값이 행(row) 형태로 저장된 학습/테스트 데이터
- `sample_submission.csv` : 제출 파일 예시
- `PCA_3D_Digit.ipynb` : 차원 축소와 간단한 분류 모델을 실험한 노트북

## 접근 방법
1. **EDA & 시각화** : PCA 2D/3D 투영으로 클래스 간 분포와 노이즈 패턴을 확인했습니다.
2. **모델링** : 기본 Logistic Regression과 RandomForest를 비교한 뒤, 간단한 CNN을 추가로 실험했습니다.
3. **정규화** : 픽셀 값을 0~1 범위로 스케일링하고, 간단한 이미지 시프팅/회전을 데이터 증강으로 사용했습니다.
4. **제출** : 최고 성능 모델을 전체 학습 데이터로 재학습해 `submission.csv`를 생성했습니다.

## 실행 방법
노트북을 열어 `train.csv.zip`과 `test.csv.zip`을 같은 경로에 둔 뒤, 셀을 순서대로 실행하면 됩니다. 로컬 환경에서는 `pip install scikit-learn matplotlib seaborn tensorflow`를 권장합니다.
