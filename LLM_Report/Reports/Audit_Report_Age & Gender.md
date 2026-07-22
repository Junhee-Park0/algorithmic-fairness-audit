# 건강보험 해지 예측 모델 공정성 감사 보고서

## 0. 비즈니스 목적 해석 및 평가 기준 도출

사용자의 비즈니스 목적은 "해지하지 않을 고객을 대상으로 마케팅을 하며, 마케팅 비용이 비싸므로 최대한 진짜로 남을 사람들만 유지로 판단"하는 것입니다. 이를 위해 1순위 성능 지표로 **Precision**을 설정합니다. 이는 오탐(False Positive) 1건당 마케팅 비용이 발생하므로, FP를 줄여 Precision을 최대화하는 것이 핵심입니다. 2순위 성능 지표로는 **Recall**을 설정하며, 이는 실제로 유지될 고객을 놓치지 않기 위해 중요합니다. 공정성 지표로는 **FPR Disparity**를 설정하여, 특정 집단에 대한 차별적 마케팅 비용 집중을 방지합니다.

Recall의 최소 허용 기준은 Baseline 대비 -15%로 설정합니다. 이 기준을 적용할 때 모든 모델이 탈락하거나 생존 모델의 Precision이 Baseline보다 낮은 경우, 기준을 -20%로 완화하여 재평가합니다.

---

## 1. Executive Summary

본 프로젝트는 건강보험 해지 예측 모델의 공정성을 개선하여, 마케팅 비용을 효율적으로 사용할 수 있는 모델을 개발하는 데 성공했습니다. Precision을 최대화하여 불필요한 마케팅 비용을 줄였으며, 공정성 지표를 통해 특정 집단에 대한 차별적 영향을 최소화했습니다. 추천 모델은 Index 4 (FairLearn In-processing + Threshold Tuning)이며, Precision +10.58%와 FPR Disparity -98.40%를 동시에 달성하여 비즈니스 효율과 규제 준수를 모두 충족합니다.

---

## 2. 서론: 법적/윤리적 준거 틀 및 적격성 평가

'인공지능 기본법(2026)'에 따르면, 본 모델은 보험 분야에 적용되어 금융 소비자에게 중대한 영향을 미칠 수 있으므로 '고영향 AI'로 분류됩니다. 금융 소비자 보호 관점에서, 공정성을 확보하는 것은 특정 집단이 보험 서비스 이용에서 배제되거나 불이익을 받지 않도록 하는 필수적인 의무입니다. 금융 가이드라인에 따른 '부당한 차별 방지' 의무는 본 프로젝트의 그룹 공정성 목표와 직접적으로 연결되며, 이는 특정 집단에 대한 차별적 마케팅 비용 집중을 방지하는 데 기여합니다.

---

## 3. 5가지 방법론 비교 분석

**[Baseline의 한계]**
- **XGBoost Baseline**: FPR Max Disparity 0.2784, FNR Max Disparity 0.2745, Selection Rate Max Disparity 0.3111. M_청년 그룹의 FPR(0.3401)이 F_고령(0.0618)의 약 5.5배에 달해, 청년 남성 그룹에 불필요한 마케팅 비용이 집중되는 구조적 차별이 발생합니다.
- **LR Baseline**: FPR Max Disparity 0.6621, FNR Max Disparity 0.5473, Selection Rate Max Disparity 0.6694. F_청년(FPR 0.7459)과 M_고령(FPR 0.1555) 간 격차가 0.5904에 달해, 연령에 따른 차별적 영향이 극심합니다.

**[Index 1 - FairLearn Post-processing (ThresholdOptimizer) / 비교 Baseline: XGBoost]**
- 메커니즘: ThresholdOptimizer는 각 그룹의 ROC 곡선을 기반으로 Equalized Odds를 만족하는 그룹별 임계값을 탐색합니다. 임계값이 높아질수록 예측 Positive 수가 줄어 FP가 감소합니다. FP 감소로 인해 분모(TP+FP)가 줄어 Precision이 증가합니다. 반면 임계값 상승으로 실제 Positive 중 일부가 Negative로 분류되어 FN이 늘어 Recall이 급감합니다.
- 수치 결과: Precision +36.13%, Recall -54.66%, FPR Disparity -94.37%
- Section 0 기준 평가: Precision 기준 통과, Recall -54.66%로 기준(-15%) 초과 탈락

**[Index 2 - GTH (Group-wise Threshold) / 비교 Baseline: LR]**
- 메커니즘: GTH는 validation의 그룹별 평균 FPR을 목표값으로 설정하고, 각 그룹의 ROC 곡선에서 해당 지점의 임계값을 탐색합니다. LR Baseline의 임계값(0.14)이 매우 낮았으므로 대부분 그룹에서 임계값이 상승하여 FP가 감소합니다. FP 감소로 Precision이 소폭 하락하는 것은 일부 그룹에서 임계값이 낮아지는 효과와 상쇄되기 때문입니다. 전반적인 임계값 상승으로 FN이 증가하여 Recall도 함께 감소합니다.
- 수치 결과: Precision -0.86%, Recall -8.79%, FPR Disparity -95.23%
- Section 0 기준 평가: Precision 기준 탈락, Recall 기준 통과

**[Index 3 - Orthogonalization + Reweighting / 비교 Baseline: XGBoost]**
- 메커니즘: 직교화는 R² 기반 Pareto 탐색으로 민감변수와 상관된 피처 성분을 제거하여 그룹 간 예측 확률 분포를 균등화합니다. 이 과정에서 민감변수가 FP 발생에 미치는 영향이 줄어들어 그룹별 FPR 편차가 감소합니다. 재가중화는 age_group × gender 교차 조합별 표본 수 역비례 가중치(N / n_cells × cell_count)를 부여하여 소수 집단의 학습 영향력을 보정합니다. 두 방법이 결합되어 FP가 감소하므로 Precision이 향상되나, 동시에 일부 실제 Positive를 보수적으로 판단하여 FN이 늘어 Recall이 감소합니다.
- 수치 결과: Precision +12.94%, Recall -22.73%, FPR Disparity -85.55%
- Section 0 기준 평가: Precision 기준 통과, Recall -22.73%로 기준(-15%) 초과 탈락

**[Index 4 - FairLearn In-processing + Threshold Tuning / 비교 Baseline: XGBoost]**
- 메커니즘: ExponentiatedGradient가 학습 중 EqualizedOdds 공정성 페널티를 반영하여 그룹 간 FPR 편향을 모델 수준에서 먼저 완화합니다. 이 단계에서 민감변수 기반 패턴의 학습 가중치가 억제되어 FP가 줄어듭니다. 후처리 단계에서는 validation 기반 목표 FPR로 그룹별 임계값을 추가 조정하여 잔여 편향을 보정합니다. 이중 보정으로 FP가 감소하여 Precision이 증가하며, 학습 단계에서 이미 편향이 완화되었으므로 Recall 감소폭이 단일 후처리 방법보다 작습니다.
- 수치 결과: Precision +10.58%, Recall -18.09%, FPR Disparity -98.40%
- Section 0 기준 평가: Precision 기준 통과, Recall -18.09%로 기준(-15%) 초과이나 생존 모델 중 기준 최근접

**[Index 5 - RBMD + GTH / 비교 Baseline: LR]**
- 메커니즘: RBMD는 σ(a×prob+b) 함수로 예측 확률 분포를 공정성 관점에서 재보정하여 그룹 간 FPR 분산을 줄입니다. 이후 GTH가 validation 기반 목표 FPR로 그룹별 임계값을 추가 조정합니다. 두 단계의 확률 압축으로 FP가 감소하여 Precision이 증가합니다. FN도 함께 증가하여 Recall이 감소하는 trade-off가 발생합니다.
- 수치 결과: Precision +7.57%, Recall -17.61%, FPR Disparity -96.06%
- Section 0 기준 평가: Precision 기준 통과, Recall -17.61%로 기준(-15%) 초과 탈락

**[요약 비교표 — XGBoost 계열]**

| 모델 | F1 | PR-AUC | Precision | Recall | FPR Max Disparity | FNR Max Disparity |
|------|----|--------|-----------|--------|-------------------|-------------------|
| XGBoost Baseline | 0.4128 | 0.3990 | 0.3381 | 0.5301 | 0.2784 | 0.2745 |
| Index 1 | 0.3158 (-23.52%) | 0.3990 (0.00%) | 0.4601 (+36.13%) | 0.2403 (-54.66%) | 0.0157 (-94.37%) | 0.0629 (-77.09%) |
| Index 3 | 0.3952 (-4.26%) | 0.3744 (-6.15%) | 0.3818 (+12.94%) | 0.4096 (-22.73%) | 0.0402 (-85.55%) | 0.1187 (-56.75%) |
| Index 4 | 0.4018 (-2.67%) | 0.3176 (-20.39%) | 0.3739 (+10.58%) | 0.4342 (-18.09%) | 0.0045 (-98.40%) | 0.1523 (-44.52%) |

**[요약 비교표 — LR 계열]**

| 모델 | F1 | PR-AUC | Precision | Recall | FPR Max Disparity | FNR Max Disparity |
|------|----|--------|-----------|--------|-------------------|-------------------|
| LR Baseline | 0.3589 | 0.3457 | 0.2326 | 0.7851 | 0.6621 | 0.5473 |
| Index 2 | 0.3489 (-2.79%) | 0.3457 (0.00%) | 0.2306 (-0.86%) | 0.7161 (-8.79%) | 0.0316 (-95.23%) | 0.1341 (-75.50%) |
| Index 5 | 0.3608 (+0.53%) | 0.3457 (0.00%) | 0.2502 (+7.57%) | 0.6469 (-17.61%) | 0.0260 (-96.06%) | 0.1178 (-78.48%) |

- Precision 기준 순위: Index 1 > Index 3 > Index 4 > Index 5 > Index 2
- Recall 기준(-15%) 통과/탈락: Index 2 통과, 나머지 탈락

---

## 4. 심층 분석: SHAP 기반 의사결정 로직의 진화

Index 1, 2, 5는 SHAP 데이터 미제공으로, 각각 XGBoost Baseline, LR Baseline, LR Baseline과 동일한 의사결정 로직을 유지한다고 해석합니다.

**[Index 3 - Orthogonalization + Reweighting]**
- `log_cost_claims_year`: 중요도 Baseline 0.3032 → 0.3889 (+28.27%). 민감변수 신호 제거 후 연간 청구 비용이 해지 예측의 핵심 근거로 부상하여 설명 가능한 신뢰성을 확보합니다.
- `kr_premium_shock`: 중요도 Baseline 0.0783 → 0.1706 (+117.88%). 모델이 민감변수 대신 고객의 실질적 이탈 동기(보험료 인상 압박)를 의사결정 근거로 채택하기 시작했음을 의미합니다.
- `seniority_policy`: 중요도 Baseline 0.3367 → 0.2059 (-38.85%). 가입 기간이 연령과 상관관계가 높으므로 간접적 민감변수 신호까지 억제된 결과로 해석됩니다.
- `age`: Top 10 외로 집계 제외되어 정확한 수치 미확인. 직교화를 통해 피처에서 age 신호가 제거된 결과로 해석되며, 공정성 개선에 기여합니다.

**[Index 4 - FairLearn In-processing + Threshold Tuning]**
- `age`: 중요도 Baseline 0.2213 → 0.053 (-76.06%). EG의 공정성 페널티가 age 기반 패턴의 학습 가중치를 억제한 결과로, 연령에 따른 차별적 예측이 감소합니다.
- `seniority_policy`: 중요도 Baseline 0.3367 → 0.077 (-77.12%). 가입 기간이 연령과 높은 상관관계를 가지므로 간접적 민감변수 신호까지 억제된 것으로 해석됩니다.
- `kr_economic_stress`: 중요도 0.0126. 사회경제적 파생 변수가 새롭게 Top 10에 진입하여, 모델이 연령 대신 실질적 경제적 취약성을 해지 근거로 학습하게 되었습니다.

---

## 5. 최종 모델 추천: Pareto 최적점 기반 Best Pick

### 1차 탈락
각 모델의 Recall % Delta를 기준(-15%)과 비교합니다:
- Index 1: -54.66% → 탈락
- Index 2: -8.79% → 통과 (단, Precision 0.2306 < LR Baseline 0.2326으로 Precision 기준 탈락)
- Index 3: -22.73% → 탈락
- Index 4: -18.09% → 탈락
- Index 5: -17.61% → 탈락

생존 모델(Index 2)의 Precision이 Baseline보다 낮으므로 Recall 기준을 **-20%로 완화**하여 재적용합니다. 마케팅 비용 절감이 1순위인 상황에서 Recall을 20% 이내로 제한하면 고객 커버리지를 최소한 유지하면서 비용 효율을 달성할 수 있습니다.

완화 기준(-20%) 재적용 결과:
- Index 4: -18.09% → **통과**
- Index 5: -17.61% → **통과**
- Index 2: -8.79% → 통과 (Precision 기준 탈락 유지)

Precision ≥ 해당 Baseline인 생존 모델(Index 4, 5)이 존재하므로 2차 비교로 진행합니다.

### 2차 비교

| 모델 | Precision | FPR Max Disparity | Recall |
|------|-----------|-------------------|--------|
| Index 4 | 0.3739 | 0.0045 | 0.4342 |
| Index 5 | 0.2502 | 0.0260 | 0.6469 |

Index 4는 Precision(+49.4% 우위)과 FPR Disparity(-82.7% 우위) 모두에서 Index 5를 압도합니다.

### 최종 선정 근거

**추천 모델: Index 4 (FairLearn In-processing + Threshold Tuning)**

Precision이 더 높은 탈락 모델(Index 1: 0.4601, Index 3: 0.3818)은 각각 Recall -54.66%, -22.73%로 완화 기준(-20%)도 초과하여 탈락 사유가 유효합니다. Index 4는 Index 5 대비 Precision에서 +49.4%, FPR Disparity에서 -82.7% 우위를 보입니다. 단, Index 5는 Recall(0.6469)이 더 높다는 점은 trade-off로 인정합니다. 공정성 측면에서 FPR Max Disparity 0.0045는 XGBoost Baseline(0.2784) 대비 98.40% 감소하여 인공지능 기본법(2026)의 차별 방지 의무를 충족합니다.

### 실무 적용 시나리오

추천 모델의 Baseline은 XGBoost(Precision 0.3381)입니다. 마케팅 대상 100명 기준, XGBoost Baseline에서는 실제 유지 고객이 약 33.81명이었으나, Index 4 적용 시 약 37.39명으로 증가하여 불필요한 마케팅 비용이 약 3.58명분 절감 가능합니다. 마케팅 단가 정보 미제공으로 금액 추정 불가.

---

## 6. 비즈니스 및 운영 제언

공정성 확보는 특정 연령·성별 집단에 대한 오탐을 줄여 마케팅 자원을 실질적 유지 고객에게 집중시키고, 차별 없는 서비스 제공으로 고객 신뢰 자본을 축적하는 경제적 이득을 제공합니다. 공정성 미확보 시 인공지능 기본법(2026) 제35조에 따라 최대 3,000만 원 이하의 과태료가 부과될 수 있으며, 이는 ERM 관점에서 기업의 리스크 관리에 중요한 요소로 작용합니다. 특정 집단 배제로 인한 평판 리스크는 장기적 고객 이탈 가속화로 이어져 보험 상품 타겟팅 정확도 저하와 경제적 손실을 동시에 초래할 수 있습니다.