# 🎓 트랜스포머 기반 생성 모형을 활용한 컨포멀 보정 분위수의 안정화 연구
(Stabilizing Conformal Prediction Calibration Quantiles Using Transformer-Based Generative Models)

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Status: 1st Draft](https://img.shields.io/badge/Status-1st_Draft_(Abstract)-orange.svg)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **💡 Notice:** 본 레포지토리는 동국대학교 대학원 통계데이터사이언스학과 **석사 학위 1차 심사(초록) 제출을 위한 예비 구현 코드 및 실험 결과**를 담고 있습니다. 전체 논문 심사 일정에 맞춰 코드가 지속적으로 업데이트될 예정입니다.

---

## 📌 연구 개요

불확실성 정량화 기법인 분할 컨포멀 예측(Split Conformal Prediction, SCP)은 계산 효율성과 단순성 덕분에 널리 활용되지만, 데이터를 분할하는 구조적 특성으로 인해 소표본(Small-sample) 환경에서는 캘리브레이션(보정) 분위수($\hat{q}$)의 통계적 변동성이 크게 증가하는 한계가 있습니다. 이는 반복 실험 간 예측 구간(Prediction Interval)의 길이가 일관되지 않게 요동치는 결과를 초래합니다.

본 연구는 **보정 분위수 자체를 안정화**하기 위해 예측 모형의 재학습 없이 보정 집합(Calibration set)만을 표적 증강하는 방법론을 제안합니다. 정형 데이터의 비선형적 의존 구조를 학습할 수 있는 **TTVAE (Transformer-based Tabular Variational AutoEncoder)**를 활용하여 고품질의 합성 데이터를 생성함으로써, 예측의 타당성을 유지한 채 예측 구간의 강건성을 실질적으로 향상시킵니다.

---

## ✨ 주요 특징

- **보정 분위수 안정화 (Quantile Stabilization):** 분할 컨포멀 예측 내 $\hat{q}$의 분산을 직접적으로 제어하여 예측 구간의 신뢰성을 확보합니다.
- **표적 데이터 증강 (Targeted Augmentation):** 전체 학습 데이터가 아닌, 보정 집합만을 대상으로 TTVAE를 학습 및 증강하여 원본 데이터의 교환성(Exchangeability)을 보존합니다.
- **사후적 적용 가능성 (Model-Agnostic Post-hoc):** 기존 예측 모형(예: 선형 회귀)의 구조 변경이나 재학습 없이 독립적으로 보정 단계에만 개입하는 유연한 아키텍처를 가집니다.
- **최적 증강 배수 도출:** 6개의 실증 데이터셋 실험을 통해 원본 표본 크기에 따른 적정 증강 비율 가이드라인을 제시합니다.

---

## 📊 실험 데이터 (Datasets)

실험에는 규모와 변수 구성이 상이한 6개의 공개 데이터셋이 사용되었습니다. 데이터 분할 비율은 **학습(Train) 60% : 보정(Calibration) 20% : 테스트(Test) 20%** 로 설정되었습니다.

> ⚠️ *주의: 각 데이터셋의 식별자(ID)나 라벨로 사용되는 변수(예: `year` 등)는 시계열 피처가 아니므로 전처리 과정에서 학습 변수에서 철저히 제외됩니다.*

| 데이터셋 | $N$ | 연속형 변수 | 범주형 변수 | 종속 변수(Target) |
| :--- | :---: | :---: | :---: | :--- |
| **번아웃 (Burnout)** | 18,590 | 4 | 3 | 번아웃 지수 |
| **보험비 (Medical Cost)** | 1,338 | 4 | 3 | 의료비 청구액 |
| **수면 건강 (Sleep Health)** | 400 | 11 | 3 | 수면 시간 |
| **수학 성적 (Math Students)** | 399 | 16 | 17 | 수학 최종 성적 |
| **자동차 연비 (Auto MPG)** | 392 | 7 | 1 | 연비 |
| **팁 (Tips)** | 244 | 3 | 4 | 팁 금액 |

---

## ⚙️ TTVAE 하이퍼파라미터 설정

본 연구의 데이터 증강에 사용된 TTVAE 아키텍처는 다음의 파라미터로 학습되었습니다.

| 파라미터 | 값 |
| :--- | :---: |
| Epochs | 800 |
| Batch size | 64 |
| Latent dimension | 32 |
| Embedding dimension | 128 |
| Transformer layers | 2 |
| Dropout | 0.1 |
| Optimizer | Adam |

---

## 📂 Project Structure

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
├── 📊 notebooks/               # 1차 심사용 실험 결과 시각화 (보정 분위수 추이 그래프 등)
├── requirements.txt            # 필요 패키지 목록
└── README.md
```

---

## 📈 주요 실험 결과 (Evaluation Metrics)

1차 심사(초록) 단계의 모의 실험을 통해 다음 지표를 산출하였습니다.

| 지표 | 설명 |
| :--- | :--- |
| **Coverage Rate** | 예측 구간이 실제 값을 포함하는 비율 (목표: $1 - \alpha$) |
| **Interval Width** | 예측 구간의 평균 길이 (짧을수록 효율적) |
| **$\hat{q}$ Variance** | 반복 실험 간 보정 분위수의 분산 (낮을수록 안정적) |

> 📋 상세 결과 및 시각화는 `notebooks/` 폴더를 참고하세요.

---

## 🔭 향후 연구 계획 (Future Work)

1차 심사 이후 지도교수 피드백을 반영하여 아래의 추가 실험 및 개선 작업을 진행할 예정입니다.

---

### 1. 이분산성 환경에서의 모의 실험 추가

현재 실험은 등분산 가정 하의 합성 데이터를 중심으로 수행되었습니다. 향후에는 **이분산성(Heteroscedasticity)을 명시적으로 부여한 합성 데이터 환경**에서 모의 실험을 추가로 진행하여, 제안 방법론이 오차 분산이 입력 변수에 따라 달라지는 현실적 데이터 구조에서도 보정 분위수 안정화 효과를 유지하는지 검증합니다.

---

### 2. 기저 예측 모형 확장 실험 (선형 회귀 → Random Forest)

현재 연구는 선형 회귀(Linear Regression)를 기저 예측 모형으로 사용하여 예측 구간(Prediction Interval, PI)을 산출하였습니다. 향후에는 **Random Forest**로 기저 모형을 교체한 실험을 추가로 수행하여, 제안된 보정 집합 증강 방법론이 비선형 모형 환경에서도 동일한 안정화 효과를 발휘하는지 비교 분석합니다.

> 💬 *비고: 기저 모형 변경 시 방법론의 Model-Agnostic 특성이 유지되는지 여부를 함께 확인합니다.*

---

### 3. 실증 데이터 재선정 (이분산성 데이터 중심)

실증 분석의 현실적 타당성을 높이기 위해 **이분산성을 내재한 데이터셋**을 중심으로 실험을 재설계합니다. 현재 후보 데이터셋은 다음과 같습니다.

- **의료(Medical) 데이터**
- **보험비(Medical Cost) 데이터**

> 💬 *비고: 데이터 규모에 따라 계층적 샘플링(Stratified Sampling) 적용 가능 여부를 추가로 검토할 예정입니다.*

---

### 4. 소표본 환경의 증강 한계에 대한 이론적 근거 보완

현재 결론에서는 보정 집합 크기가 50개 미만인 초소표본 환경에서 1배 초과 증강 시 보정 분위수의 변동성이 오히려 재상승하는 현상이 관찰되었음을 기술하고 있습니다. 그러나 이 현상의 **통계적·이론적 원인 규명**이 충분하지 않은 상태입니다.

향후에는 다음 관점에서 이를 설명할 근거를 마련합니다.

- TTVAE의 생성 품질이 원본 표본 크기에 따라 열화되는 메커니즘 분석
- 소표본에서 생성 모형이 학습 분포를 과적합(Overfitting)하거나 분포를 왜곡할 가능성 검토
- 증강 배수와 보정 분위수 분산 간의 이론적 상한(Upper Bound) 도출 시도

> 💬 *구체적으로: 보정 집합이 80개 수준인 경우 2~4배 증강에서 안정성이 극대화된 반면, 50개 미만의 경우 1배 증강만이 유효하며 초과 시 변동성이 재상승하는 이유에 대한 이론적 설명을 논문 결론 및 논의 섹션에 보완합니다.*

---

### 5. Train + Calibration 동시 증강 실험

현재 연구는 **보정 집합(Calibration set)만을 단독 증강**하는 방식을 채택하였습니다. 향후에는 **학습 집합(Train set)과 보정 집합을 동시에 증강**하는 실험을 추가로 수행하여, 단독 증강 대비 성능 차이를 비교합니다.

- 실험 설계는 랩미팅 논의 내용을 바탕으로 구체화할 예정입니다.
- 동시 증강이 교환성(Exchangeability) 가정에 미치는 영향을 이론적으로 점검합니다.
- 두 증강 전략 간의 Coverage Rate, Interval Width, $\hat{q}$ Variance 변화를 체계적으로 비교합니다.

---

### 6. 일반화 가능성 검증을 위한 데이터셋 확장

본 연구는 6개의 데이터셋을 대상으로 실험을 수행하였으나, 방법론의 일반화 가능성을 보다 엄밀히 검증하기 위해 **다양한 도메인 및 표본 규모의 추가 데이터셋**으로 실험을 확장합니다.

---

### 7. TTVAE 하이퍼파라미터 최적화

현재 TTVAE는 선행 연구의 기본 설정값(Default configuration)을 적용하였습니다. 향후에는 **데이터셋별 표본 규모를 고려한 하이퍼파라미터 최적화(Hyperparameter Tuning)**를 수행하여, 생성 품질과 보정 분위수 안정화 성능을 함께 개선합니다.

---

## 📦 설치 방법 (Installation)

```bash
git clone https://github.com/your-repo/ttvae-conformal.git
cd ttvae-conformal
pip install -r requirements.txt
```

---

## 🚀 실행 방법 (Quick Start)

```bash
# Step 1: 보정 집합 기반 TTVAE 학습 및 데이터 증강
python scripts/01_train_ttvae.py --dataset burnout --augment_ratio 2

# Step 2: CP 성능 평가 시뮬레이션 (10회 반복)
python scripts/02_run_simulation.py --dataset burnout --n_trials 10
```

---

## 📄 인용 (Citation)

```bibtex
@mastersthesis{kim2024ttvae_conformal,
  title     = {트랜스포머 기반 생성 모형을 활용한 컨포멀 보정 분위수의 안정화 연구},
  author    = {김희주},
  school    = {동국대학교 대학원 통계데이터사이언스학과},
  year      = {2024},
  note      = {석사 학위 논문 (1차 심사 단계)}
}
```

---

## 📬 문의 (Contact)

- 소속: 동국대학교 대학원 통계데이터사이언스학과
- 이메일: [your-email@dongguk.edu](mailto:your-email@dongguk.edu)
