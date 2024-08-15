import os
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from tqdm import tqdm

# 데이터 읽어오기
ROOT_DIR = "C:/Users/hyunj/OneDrive/바탕 화면/LG Aimers"
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

# 모델 정의
model = XGBClassifier(random_state=RANDOM_STATE, use_label_encoder=False, eval_metric='logloss')

# 하이퍼파라미터 튜닝을 위한 그리드 서치 설정
param_grid = {
    'n_estimators': [100, 200],
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 5, 7],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0]
}

grid_search = GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    scoring='f1_macro',  # 또는 적합한 평가 지표를 선택할 수 있습니다.
    cv=5,
    verbose=2,
    n_jobs=-1
)

# 학습 데이터 준비
features = []

for col in df_train.columns:
    try:
        df_train[col] = df_train[col].astype(int)
        features.append(col)
    except:
        continue

train_x = df_train[features]
train_y = df_train["target"]

# 하이퍼파라미터 튜닝 및 모델 학습
grid_search.fit(train_x, train_y)

# 최적의 하이퍼파라미터 출력
print(f"Best parameters found: {grid_search.best_params_}")

# 최적 모델로 예측
best_model = grid_search.best_estimator_

# 테스트 데이터 예측
test_data = pd.read_csv(os.path.join(ROOT_DIR, "testset/test.csv"))
df_test_x = test_data[features]

for col in df_test_x.columns:
    try:
        df_test_x.loc[:, col] = df_test_x[col].astype(int)
    except:
        continue

test_pred = best_model.predict(df_test_x)


# 제출 파일 작성
df_sub = pd.read_csv("C:/Users/hyunj/OneDrive/바탕 화면/LG Aimers/testset/submission.csv")
df_sub["target"] = test_pred

# 제출 파일 저장
df_sub.to_csv(r"C:/Users/hyunj/OneDrive/바탕 화면/LG Aimers/result/submission_xgboost.csv", index=False)
print("finished programming")
