# 🎓 트랜스포머 기반 생성 모형을 활용한 컨포멀 보정 분위수의 안정화 연구
(Stabilizing Conformal Prediction Calibration Quantiles Using Transformer-Based Generative Models)

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

본 레포지토리는 동국대학교 대학원 통계데이터사이언스학과 석사 학위 논문의 공식 구현 코드를 포함하고 있습니다.

## 📌 연구 개요
불확실성 정량화 기법인 분할 컨포멀 예측(Split Conformal Prediction, SCP)은 계산 효율성과 단순성 덕분에 널리 활용되지만, 데이터를 분할하는 구조적 특성으로 인해 소표본(Small-sample) 환경에서는 캘리브레이션(보정) 분위수($\hat{q}$)의 통계적 변동성이 크게 증가하는 한계가 있습니다. 이는 반복 실험 간 예측 구간(Prediction Interval)의 길이가 일관되지 않게 요동치는 결과를 초래합니다.

본 연구는 **보정 분위수 자체를 안정화**하기 위해 예측 모형의 재학습 없이 보정 집합(Calibration set)만을 표적 증강하는 방법론을 제안합니다. 정형 데이터의 비선형적 의존 구조를 학습할 수 있는 **TTVAE (Transformer-based Tabular Variational AutoEncoder)**를 활용하여 고품질의 합성 데이터를 생성함으로써, 예측의 타당성을 유지한 채 예측 구간의 강건성을 실질적으로 향상시킵니다.

## ✨ 주요 특징
* **보정 분위수 안정화 (Quantile Stabilization):** 분할 컨포멀 예측 내 $\hat{q}$의 분산을 직접적으로 제어하여 예측 구간의 신뢰성을 확보합니다.
* **표적 데이터 증강 (Targeted Augmentation):** 전체 학습 데이터가 아닌, 보정 집합만을 대상으로 TTVAE를 학습 및 증강하여 원본 데이터의 교환성(Exchangeability)을 보존합니다.
* **사후적 적용 가능성 (Model-Agnostic Post-hoc):** 기존 예측 모형(예: 선형 회귀)의 구조 변경이나 재학습 없이 독립적으로 보정 단계에만 개입하는 유연한 아키텍처를 가집니다.
* **최적 증강 배수 도출:** 6개의 실증 데이터셋 실험을 통해 원본 표본 크기에 따른 적정 증강 비율 가이드라인을 제시합니다.

## 📊 실험 데이터 (Datasets)
실험에는 규모와 변수 구성이 상이한 6개의 공개 데이터셋이 사용되었습니다. 데이터 분할 비율은 **학습(Train) 60% : 보정(Calibration) 20% : 테스트(Test) 20%** 로 설정되었습니다.
*(주의: 각 데이터셋의 식별자(ID)나 라벨로 사용되는 변수(예: `year` 등)는 시계열 피처가 아니므로 전처리 과정에서 학습 변수에서 제외됩니다.)*

| 데이터셋 | $N$ | 연속형 변수 | 범주형 변수 | 종속 변수(Target) |
| :--- | :--- | :--- | :--- | :--- |
| **번아웃 (Burnout)** | 18,590 | 4 | 3 | 번아웃 지수 |
| **보험비 (Medical Cost)** | 1,338 | 4 | 3 | 의료비 청구액 |
| **수면 건강 (Sleep Health)** | 400 | 11 | 3 | 수면 시간 |
| **수학 성적 (Math Students)** | 399 | 16 | 17 | 수학 최종 성적 |
| **자동차 연비 (Auto MPG)** | 392 | 7 | 1 | 연비 |
| **팁 (Tips)** | 244 | 3 | 4 | 팁 금액 |

## ⚙️ TTVAE 하이퍼파라미터 설정
수치형 변수의 변분 가우시안 혼합모형(VGM)과 다층 트랜스포머 인코더/디코더를 결합한 TTVAE는 다음의 파라미터로 학습되었습니다.
* **Epochs:** 800 / **Batch size:** 64
* **Latent dimension:** 32 / **Embedding dimension:** 128
* **Transformer layers:** 2 / **Dropout:** 0.1 / **Optimizer:** Adam

## 📂 디렉토리 구조
```text
.
├── 📂 data/                    # 6개의 실증 데이터셋 원본 및 합성 데이터
│
├── 🧠 models/                  
│   ├── ttvae.py                # TTVAE 아키텍처 및 학습/샘플링 로직 구현
│   └── conformal.py            # Split Conformal Prediction 비순응 점수 및 구간 산출
│
├── 🚀 scripts/                 
│   ├── 01_train_ttvae.py       # 보정 집합 기반 TTVAE 학습 및 데이터 증강 스크립트
│   └── 02_run_simulation.py    # 반복 무작위 분할(10회)에 따른 CP 성능 평가 시뮬레이션
│
├── 📊 notebooks/               # 실험 결과 시각화 (보정 분위수 추이 그래프 등)
├── requirements.txt            # 필요 패키지 목록
└── README.md

@mastersthesis{Kim2026,
  author       = {Heeju Kim},
  title        = {Stabilizing Conformal Prediction Calibration Quantiles Using Transformer-Based Generative Models},
  school       = {Dongguk University},
  year         = {2026},
}

