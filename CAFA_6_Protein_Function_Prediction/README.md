# CAFA 6 Protein Function Prediction

## 대회 소개
CAFA (Critical Assessment of Function Annotation) 6는 단백질 서열만으로 Gene Ontology(GO) 라벨을 예측하는 멀티라벨 분류 대회입니다. 공개 데이터는 아미노산 서열 FASTA와 GO 용어 매핑으로 구성되어 있으며, 참가자는 미지의 단백질 기능을 최대한 정확하게 예측해야 합니다.

## 폴더 구성
- `prepare_data.py` : UniProt/GO 원본 데이터를 모델 입력에 맞게 정리하고 임베딩 파일을 생성합니다.
- `dataset.py` : 학습/검증 파이프라인에서 사용하는 PyTorch Dataset 및 데이터 증강 로직을 정의합니다.
- `model.py` : 1D CNN 기반 서열 인코더와 다중-헤드 분류 레이어를 구현합니다.
- `train_baseline.py` : 기본 하이퍼파라미터(배치 64, AdamW, cosine scheduler)로 모델을 학습합니다.
- `cafa6_1DCNN.ipynb` : 실험 기록, 지표(GOA Fmax) 계산 방법, 모델 개선 아이디어를 정리한 노트북입니다.
- `predcit_make_submission.py` : 학습된 체크포인트를 로드해 제출 형식에 맞는 TSV 파일을 생성합니다.

## 접근 방법
1. **입력 표현** : 단백질 서열을 one-hot 또는 learnable embedding으로 변환하고, 길이 차이를 패딩/마스킹으로 처리했습니다.
2. **모델 구조** : dilated 1D convolution과 residual 블록을 쌓아 국소 패턴과 장거리 패턴을 동시에 학습하게 했습니다.
3. **손실 함수** : 클래스 불균형 완화를 위해 positive class weighting을 적용한 BCEWithLogitsLoss를 사용했습니다.
4. **추론** : 검증 집합에서 최적 threshold를 grid search로 찾은 뒤, 동일 임계값을 제출에 사용했습니다.

## 재현 방법
```bash
pip install -r requirements.txt  # 필요 시
python prepare_data.py
python train_baseline.py --epochs 30 --lr 5e-4
python predcit_make_submission.py --checkpoint runs/best.pt
```

노트북(`cafa6_1DCNN.ipynb`)에서는 위 스크립트를 순차적으로 실행하며 EDA와 추가 실험(ensemble, GO hierarchy regularization)을 진행했습니다.
