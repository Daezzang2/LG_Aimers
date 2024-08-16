import os
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
from tqdm import tqdm
import warnings

# 경고 메시지 숨기기
warnings.filterwarnings("ignore", category=UserWarning, module="xgboost")

# 데이터 읽어오기
ROOT_DIR = "/Users/hyeonggeun_kim/Documents/LG Aimers/v2/"
RANDOM_STATE = 110

# Load data
train_data = pd.read_csv(os.path.join(ROOT_DIR, "dataset/train.csv"))

# under sampling
normal_ratio = 1.0  # 1.0 means 1:1 ratio

df_normal = train_data[train_data["target"] == "Normal"]
df_abnormal = train_data[train_data["target"] == "AbNormal"]

num_normal = len(df_normal)
num_abnormal = len(df_abnormal)
print(f"  Total: Normal: {num_normal}, AbNormal: {num_abnormal}")

df_normal = df_normal.sample(n=int(num_abnormal * normal_ratio), replace=False, random_state=RANDOM_STATE)
df_concat = pd.concat([df_normal, df_abnormal], axis=0).reset_index(drop=True)
df_concat.value_counts("target")

# 레이블 인코딩
label_encoder = LabelEncoder()
df_concat["target"] = label_encoder.fit_transform(df_concat["target"])

# 데이터 분할
df_train, df_val = train_test_split(
    df_concat,
    test_size=0.3,
    stratify=df_concat["target"],
    random_state=RANDOM_STATE,
)

def print_stats(df: pd.DataFrame):
    num_normal = len(df[df["target"] == 0])
    num_abnormal = len(df[df["target"] == 1])
    if num_normal == 0:
        ratio = "Infinity (No Normal samples)"
    else:
        ratio = num_abnormal / num_normal

    print(f"  Total: Normal: {num_normal}, AbNormal: {num_abnormal}" + f" ratio: {ratio}")

# Print statistics
print(f"  \tAbnormal\tNormal")
print_stats(df_train)
print_stats(df_val)

# 피처 스케일링 및 변환
df_train = df_train.apply(pd.to_numeric, errors='coerce').fillna(0)
df_val = df_val.apply(pd.to_numeric, errors='coerce').fillna(0)

# 오버샘플링 적용
features = [col for col in df_train.columns if col != "target"]
train_x = df_train[features]
train_y = df_train["target"]

smote = SMOTE(random_state=RANDOM_STATE)
train_x, train_y = smote.fit_resample(train_x, train_y)

# 모델 정의
model = XGBClassifier(random_state=RANDOM_STATE, eval_metric='logloss')

# 하이퍼파라미터 튜닝을 위한 그리드 서치 설정
param_grid = {
    'n_estimators': [50, 100],  # 적당히 조정된 범위
    'learning_rate': [0.1, 0.2],
    'max_depth': [3, 5],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0]
}

grid_search = GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    scoring='f1_macro',
    cv=5,
    verbose=2,
    n_jobs=-1,
    error_score='raise'  # 디버깅 모드 활성화
)

# 하이퍼파라미터 튜닝 및 모델 학습
grid_search.fit(train_x, train_y)

# 최적의 하이퍼파라미터 출력
print(f"Best parameters found: {grid_search.best_params_}")

# 최적 모델로 검증 데이터 예측 및 평가
val_pred = grid_search.best_estimator_.predict(df_val[features])
print(classification_report(df_val['target'], val_pred))
print(confusion_matrix(df_val['target'], val_pred))

# 테스트 데이터 예측
test_data = pd.read_csv(os.path.join(ROOT_DIR, "testset/test.csv"))
df_test_x = test_data[features]

# 피처 스케일링 및 변환
df_test_x = df_test_x.apply(pd.to_numeric, errors='coerce').fillna(0)

test_pred = grid_search.best_estimator_.predict(df_test_x)

# 제출 파일 작성
df_sub = pd.read_csv("/Users/hyeonggeun_kim/Documents/LG Aimers/v2/testset/submission.csv")
df_sub["target"] = label_encoder.inverse_transform(test_pred)

# 제출 파일 저장
df_sub.to_csv(r"/Users/hyeonggeun_kim/Documents/LG Aimers/v2/result/submission_xgboost.csv", index=False)
print("finished programming")