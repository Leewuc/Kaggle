# Distracted Driving Risk Detection Challenge

차량 IoT/테레매틱스 로그와 운전자 생체 신호를 활용해 사고 위험도를 1~4등급으로 분류하는 Kaggle 대회입니다. 속도·가속도·날씨·도로 사고 이력 등 다양한 센서 값을 조합해 위험 패턴을 찾고, 스태킹/반지도 학습을 통해 일반화를 강화했습니다.

## 주요 파일
- `config.py` : 데이터 경로, 난수 시드, 어그레시브 옵션 등 공통 설정과 라이브러리 로딩
- `features.py` : 속도 위반 비율, 가속도 이벤트, 엔진 부하·RPM, 기상 위험도, 운전자 심박수 지표, 구간별 집계 특징을 생성
- `data.py` : 원본 CSV(`kaggle_train.csv`, `kaggle_test.csv`)를 불러와 특성을 만들고, 표준화/결측치 처리와 샘플 가중치 계산 수행
- `models.py` : XGBoost, LightGBM, CatBoost의 OOF 학습·스태킹, 온도 보정, 전체 데이터 재학습 및 제출 파일 생성 로직
- `ssl_pl.py` : Mean Teacher 기반 반지도 학습과 안정성 조건을 활용한 퓨도 라벨링
- `train.py` : 전처리 → 교차 검증 모델 학습 → 스태킹·보정 → 반지도 학습/퓨도 라벨 → 최종 제출 파일 생성까지의 엔드 투 엔드 파이프라인

## 실행 방법
1. Kaggle 데이터셋을 `DATA_DIR` 경로에 위치시키고 필요 시 환경변수로 경로/시드/검증 전략을 설정합니다.
2. 의존성(Scikit-learn, XGBoost, LightGBM, CatBoost, PyTorch 등)을 설치한 뒤 `python train.py`를 실행하면 학습부터 제출 파일(`submission.csv`) 생성까지 진행됩니다.

