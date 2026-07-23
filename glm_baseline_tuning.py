import pandas as pd
import numpy as np
from datetime import datetime
import os
import pickle
import json
import warnings
import torch
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import optuna
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.exceptions import ConvergenceWarning
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.utils.class_weight import compute_sample_weight

# 데이터 불러오기
data=pd.read_pickle('/Health_Insurance_final.pkl')
# lapse 값 변환: 1, 3 → 1 / 2 → 0
data['lapse'] = data['lapse'].map({1: 1, 3: 1, 2: 0})

# target 변수와 사용 변수 정의
target = 'is_churn'
x_cols = [
    # 기본
    'premium', 'seniority_policy', 'type_policy_dg', 'type_product', 'new_business',
    'log_cost_claims_year', 'distribution_channel',
    # 나이 관련
    'age',
    # 성별 관련
    'gender',
    # 지역 관련
    'IICIMUN_capped', 'IICIPROV', 'C_C', 'C_H_num', 'C_GI', 'C_IE_T',
]

# 파생변수
der_cols = [
    'missing_geo_cxt',       # 지역 결측 신호
    'high_loss',                  # 보험사 손해율
    'relative_poverty',        # 지역 내 상대적 빈곤
    'kr_premium_shock',   # 가격 인상 압박(비율 단위)
    'kr_economic_stress',  # 소득 대비 체감 부담
    # 'kr_retention_years',    # 장기 유지 혜택
    'kr_early_laps',             #신계약 위험 구간
    # 'kr_direct_channel',     # 가입 채널 영향
    'kr_medical_desert'     # 인프라 취약성
]
print(f"사용 변수 개수(기본) : {len(x_cols)}")
print(f"사용 변수 개수(파생) : {len(der_cols)}")

# 기본변수+파생변수 (x_cols)
x_cols += der_cols
print(f"최종 변수 개수 : {len(x_cols)}개")

device = "cuda" if torch.cuda.is_available() else "cpu"

# 범주형 변수
cat_cols = ['type_policy_dg', 'type_product', 'new_business', 'distribution_channel', 'gender', 'C_C', 'kr_early_laps']

# 수치형 변수
num_cols = [col for col in x_cols if col not in cat_cols]

# 범주형 변수 인코딩 위해 모든 범주 type str로 변경 (C_C변수 포함)
data[cat_cols] = data[cat_cols].astype(str)


#데이터셋 분리 (시계열 데이터임을 고려해 연도별 분리)
train = data[data['period'] == 2017]
val   = data[data['period'] == 2018]
test  = data[data['period'] == 2019]

X_train, y_train = train[x_cols], train[target]
X_val, y_val     = val[x_cols], val[target]
X_test, y_test   = test[x_cols], test[target]

# 전처리 (표준화, 인코딩)
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), num_cols),   #수치형 표준화 
        ('cat', OneHotEncoder(drop='first'), cat_cols)  #범주형 표준화 
    ])

# SCAD용 skglm
try:
    from skglm.datafits import Logistic
    from skglm.penalties import SCAD
    from skglm import GeneralizedLinearEstimator
    SKGLM_AVAILABLE = True
except ImportError:
    SKGLM_AVAILABLE = False
    print("skglm 미설치 → pip install skglm")

# Adaptive Lasso / Adaptive Ridge
class AdaptiveLogisticRegression(BaseEstimator, ClassifierMixin):
    def __init__(self, C=1.0, gamma=1.0, penalty_type='adaptive_lasso',
                 class_weight=None, max_iter=3000, tol=1e-4, random_state=42):
        self.C = C
        self.gamma = gamma
        self.penalty_type = penalty_type
        self.class_weight = class_weight
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state

    def fit(self, X, y, sample_weight=None):
        X = np.asarray(X)
        # 초기 계수 추정 (Ridge - 모든 변수 유지하기 위해)
        init = LogisticRegression(
            l1_ratio=0.0, C=1.0, solver='saga',
            class_weight=self.class_weight,
            max_iter=self.max_iter, random_state=self.random_state
        )
        init.fit(X, y, sample_weight=sample_weight)
        coef_abs = np.abs(init.coef_[0])

        # adaptive 가중치 (계수 작을수록 가중치 큼 → 더 강하게 제거)
        self.weights_ = 1.0 / (coef_abs ** self.gamma + 1e-8)

        # penalty 종류별 스케일링
        if self.penalty_type == 'adaptive_lasso':
            X_scaled = X / self.weights_       # L1 (solver='saga'는 모든 계수에 동일한 가중치 벌점으로 줌. 따라서 X_scaled 사용해 adaptive lasso penalty 구현)
            l1r = 1.0
        else:  # adaptive_ridge
            X_scaled = X / np.sqrt(self.weights_)  # L2는 sqrt (마찬가지로 X_scaled 사용해 adaptive ridge penalty 구현)
            l1r = 0.0

        self.model_ = LogisticRegression(
            l1_ratio=l1r, C=self.C, solver='saga',
            class_weight=self.class_weight,
            max_iter=self.max_iter, tol=self.tol, random_state=self.random_state
        )
        self.model_.fit(X_scaled, y, sample_weight=sample_weight)
        self.classes_ = self.model_.classes_
        return self

    # 예측 단계에서 입력 데이터를 학습 때와 동일한 방식으로 스케일링해주는 메서드 (fit()에서 모델은 스케일된 X에 대해 학습함. 따라서 새로운 데이터에 대한 예측도 같은 스케일의 X 적용 후 해야됨.)
    def _scale(self, X):
        X = np.asarray(X)
        if self.penalty_type == 'adaptive_lasso':
            return X / self.weights_
        return X / np.sqrt(self.weights_)  # 'adaptive_ridge'인 경우

    def predict_proba(self, X):
        return self.model_.predict_proba(self._scale(X))

    def predict(self, X):
        return self.model_.predict(self._scale(X))

# SCAD (skglm)
class SCADLogisticRegression(BaseEstimator, ClassifierMixin):
    def __init__(self, alpha=0.1, scad_gamma=3.7, class_weight=None):
        self.alpha = alpha
        self.scad_gamma = scad_gamma
        self.class_weight = class_weight

    def _sample_weight(self, y):
        if self.class_weight is None:
            return None
        if self.class_weight == 'balanced':
            return compute_sample_weight('balanced', y)
        return np.array([self.class_weight[yi] for yi in y])  # class_weight == 'custom'

    def fit(self, X, y):
        X = np.asarray(X)
        penalty = SCAD(alpha=self.alpha, gamma=self.scad_gamma)

        self.model_ = GeneralizedLinearEstimator(
            datafit=Logistic(),
            penalty=penalty
        )

        sw = self._sample_weight(y)
        try:
            if sw is not None:
                self.model_.fit(X, y, sample_weight=sw)
            else:
                self.model_.fit(X, y)
        except TypeError:
            # sample_weight 미지원 버전 대응
            self.model_.fit(X, y)

        self.classes_ = np.unique(y)
        return self

    def predict_proba(self, X):
        X = np.asarray(X)
        coef = np.asarray(self.model_.coef_).flatten()

        # intercept가 있으면 더하고, 없으면 0
        intercept = getattr(self.model_, 'intercept_', 0.0)
        if intercept is None:
            intercept = 0.0
        intercept = np.asarray(intercept).flatten()
        intercept = intercept[0] if intercept.size > 0 else 0.0

        logits = X @ coef + intercept
        p1 = 1 / (1 + np.exp(-logits))
        return np.column_stack([1 - p1, p1])

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]

# 튜닝할 하이퍼파라미터 및 범위 설정
def objective(trial):
    # 1. penalty
    penalty = trial.suggest_categorical('penalty', [
        'lasso', 'ridge', 'elasticnet',
        'adaptive_lasso', 'adaptive_ridge', 'scad'
    ])

    # 2. weight_type
    weight_type = trial.suggest_categorical('weight_type', ['none', 'balanced', 'custom'])
    if weight_type == 'none':
        class_weight = None
    elif weight_type == 'balanced':
        class_weight = 'balanced'
    else:
        churn_weight = trial.suggest_float('churn_weight', 1.0, 8.0)
        class_weight = {0: 1.0, 1: churn_weight}
    
    # 3. C(정규화 강도, lambda 역수), l1_ratio
    if penalty in ['lasso', 'ridge', 'elasticnet']:
        C = trial.suggest_float('C', 1e-4, 10.0, log=True)
        if penalty == 'ridge':
            l1_ratio=0.0
        elif penalty == 'lasso':
            l1_ratio=1.0
        else:
            l1_ratio = trial.suggest_float('l1_ratio', 0.0, 1.0)
        kwargs = dict(
            C=C, l1_ratio=l1_ratio, class_weight=class_weight,
            solver='saga', max_iter=3000, tol=1e-4, random_state=42
        )
        if penalty == 'elasticnet':
            kwargs['l1_ratio'] = trial.suggest_float('l1_ratio', 0.0, 1.0)
        model = LogisticRegression(**kwargs)
    # 4. C, gamma
    elif penalty in ['adaptive_lasso', 'adaptive_ridge']:
        C = trial.suggest_float('C', 1e-4, 10.0, log=True)
        gamma = trial.suggest_float('gamma', 0.5, 2.0)   # 보통 
        model = AdaptiveLogisticRegression(
            C=C, gamma=gamma, penalty_type=penalty,
            class_weight=class_weight, max_iter=3000, tol=1e-4, random_state=42
        )

    # 5. alpha(정규화 강도, lambda와 동일), scad_gamma
    elif penalty == 'scad':
        if not SKGLM_AVAILABLE:
            raise optuna.exceptions.TrialPruned()
        alpha = trial.suggest_float('scad_alpha', 1e-4, 1.0, log=True)
        scad_gamma = trial.suggest_float('scad_gamma', 2.01, 5.0)   #보통 3.7
        model = SCADLogisticRegression(
            alpha=alpha, scad_gamma=scad_gamma, class_weight=class_weight
        )

    optuna_pipeline = Pipeline([
        ('preprocessor', clone(preprocessor)),
        ('model', model)
    ])

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        optuna_pipeline.fit(X_train, y_train)
        preds_proba = optuna_pipeline.predict_proba(X_val)[:, 1]

    return average_precision_score(y_val, preds_proba)

# Study 실행
study = optuna.create_study(
    direction='maximize',  # PR-AUC 극대화
    sampler=optuna.samplers.TPESampler(seed=42)  #과거 실험 바탕으로 하이퍼파라미터 조합 시도하는 베이지안 최적화 알고리즘 사용
)
study.optimize(objective, n_trials=1000)

print("\n" + "="*40)
print(f"최적 Val PR-AUC : {study.best_value:.4f}")
print(f"최적 파라미터 : {study.best_params}")
print("="*40)

# penalty별 최고 PR-AUC 비교
import pandas as pd
results = [
    {'penalty': t.params.get('penalty'), 'prauc': t.value}
    for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE
]
df = pd.DataFrame(results)
print("\n[penalty별 최고 PR-AUC]")
print(df.groupby('penalty')['prauc'].max().sort_values(ascending=False))

# 선택된 하이퍼파라미터로 모델 학습
def build_model_from_params(params):
    penalty = params['penalty']

    # class_weight 복원
    weight_type = params['weight_type']
    if weight_type == 'none':
        class_weight = None
    elif weight_type == 'balanced':
        class_weight = 'balanced'
    else:
        class_weight = {0: 1.0, 1: params['churn_weight']}

    # penalty별 모델 구성
    if penalty in ['lasso', 'ridge', 'elasticnet']:
        C = params['C']
        if penalty == 'ridge':
            l1_ratio = 0.0
        elif penalty == 'lasso':
            l1_ratio = 1.0
        else:  # elasticnet
            l1_ratio = params['l1_ratio']
        model = LogisticRegression(
            C=C, l1_ratio=l1_ratio, class_weight=class_weight,
            solver='saga', max_iter=3000, tol=1e-4, random_state=42
        )

    elif penalty in ['adaptive_lasso', 'adaptive_ridge']:
        model = AdaptiveLogisticRegression(
            C=params['C'], gamma=params['gamma'], penalty_type=penalty,
            class_weight=class_weight, max_iter=3000, tol=1e-4, random_state=42
        )

    elif penalty == 'scad':
        model = SCADLogisticRegression(
            alpha=params['scad_alpha'],
            scad_gamma=params['scad_gamma'],
            class_weight=class_weight
        )

    return model

best_params = {
    'penalty': 'adaptive_lasso',
    'weight_type': 'none',
    'gamma': 0.5010232110393945,
    **study.best_params          # C 값 덮어쓰기
}

# 최적 파라미터로 모델 재구성
best_model = build_model_from_params(best_params)

# Pipeline으로 감싸서 재학습
best_pipeline = Pipeline([
    ('preprocessor', clone(preprocessor)),
    ('model', best_model)
])

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    best_pipeline.fit(X_train, y_train)

print("best_pipeline 학습 완료")

# 모델 pickle 파일로 저장
file_name = 'glm.pkl'
with open(file_name, 'wb') as f:
    pickle.dump(best_pipeline, f)

# best_params json 파일로 저장 
best_params_file_name ='glm_best_params.json'
with open(best_params_file_name, "w", encoding="utf-8") as f:
    json.dump(best_params, f, ensure_ascii=False, indent=2)
