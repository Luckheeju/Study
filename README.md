# Transformer 기반 합성 데이터 증강을 통한 Split Conformal Prediction 분위수 안정화
(Stabilizing Split Conformal Prediction Quantiles via Transformer-based Synthetic Data Augmentation)

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

본 레포지토리는 동국대학교 통계학과 석사 학위 논문 "Transformer 기반 합성 데이터 증강을 통한 Split Conformal Prediction 분위수 안정화"의 공식 구현 코드를 포함하고 있습니다.

## 📌 연구 개요
이분산성(Heteroscedasticity)이 존재하는 환경, 특히 소표본(Small-sample) 데이터 환경에서는 Split Conformal Prediction(CP)의 캘리브레이션 분위수(`q_hat`)가 매우 불안정해지는 문제가 발생합니다. 

본 연구는 생성 모델들 간의 단순한 성능 비교가 아닌, 캘리브레이션 분위수(`q_hat`)의 안정화**에 핵심 목적을 두고 있습니다. 이를 위해 Tabular Twin Variational Autoencoders(TTVAE)와 CTGAN을 활용하여 캘리브레이션 세트를 직접적으로 증강하는 파이프라인을 구축하였으며, 과적합(Overfitting) 없이 더 안정적이고 좁은 예측 구간(Prediction Interval)을 확보하는 방법론을 제시합니다.

## ✨ 주요 특징
* **분위수 안정화 (Quantile Stabilization):** Conformal Prediction 내 `q_hat`의 분산을 직접적으로 제어하고 안정시킵니다.
* **캘리브레이션 데이터 증강:** 정형 데이터(Tabular Data)의 비선형적 패턴을 포착하기 위해 설계된 TTVAE 및 CTGAN 모델을 활용합니다.
* **이분산성 제어:** 변동성이 큰 시나리오에서 CP의 커버리지(Coverage) 및 구간 너비(Width) 성능 지표 개선을 시뮬레이션을 통해 검증합니다.

## 📂 디렉토리 구조
```text
├── data/                   # 데이터셋 폴더 
│   │                       # (주의: 데이터셋 내 'year' 변수는 시계열 피처가 아닌 파일 식별 라벨로 사용됨)
│   ├── raw/                # 원본 데이터
│   └── synthetic/          # TTVAE/CTGAN을 통해 생성된 증강 데이터
├── models/                 
│   ├── generators.py       # TTVAE 및 CTGAN 모델 아키텍처 구현
│   └── conformal.py        # Split CP 및 AWCP(Adaptive Weighted CP) 비순응도 점수 계산 로직
├── notebooks/              # 탐색적 데이터 분석(EDA) 및 튜토리얼용 Jupyter Notebook
├── scripts/                
│   ├── 01_generate_data.py # 합성 캘리브레이션 데이터 생성용 실행 스크립트
│   └── 02_run_cp_sim.py    # 커버리지 및 구간 너비 성능 평가를 위한 메인 시뮬레이션 스크립트
├── requirements.txt        # 필요 패키지 목록
└── README.md
