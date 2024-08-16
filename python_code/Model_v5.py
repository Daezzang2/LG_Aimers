import os
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split

# 설정
ROOT_DIR = "/Users/hyeonggeun_kim/Documents/LG Aimers/v2/"
RANDOM_STATE = 110

# 데이터 읽어오기
train_data = pd.read_csv(os.path.join(ROOT_DIR, "dataset/train.csv"))

# 언더 샘플링
normal_ratio = 1.0  # 1:1 비율
df_normal = train_data[train_data["target"] == "Normal"]
df_abnormal = train_data[train_data["target"] == "AbNormal"]

# 샘플링 및 병합
df_normal = df_normal.sample(n=int(len(df_abnormal) * normal_ratio), replace=False, random_state=RANDOM_STATE)
df_concat = pd.concat([df_normal, df_abnormal], axis=0).reset_index(drop=True)

# 데이터 분할
df_train, df_val = train_test_split(
    df_concat,
    test_size=0.3,
    stratify=df_concat["target"],
    random_state=RANDOM_STATE,
)

# 통계 정보 출력 함수
def print_stats(df: pd.DataFrame):
    num_normal = len(df[df["target"] == "Normal"])
    num_abnormal = len(df[df["target"] == "AbNormal"])
    print(f"  Total: Normal: {num_normal}, AbNormal: {num_abnormal}, ratio: {num_abnormal / num_normal}")

# 학습 및 검증 데이터 통계 출력
print("Train Data Stats:")
print_stats(df_train)
print("Validation Data Stats:")
print_stats(df_val)

# 사용할 피처 선택
features = [
    'Head Zero Position Z Collect Result_Dam',
    '1st Pressure 1st Pressure Unit Time_AutoClave',
    '3rd Pressure Unit Time_AutoClave',
    'Chamber Temp. Collect Result_AutoClave',
    'Chamber Temp. Unit Time_AutoClave',
    'Stage2 Line3 Distance Speed Collect Result_Dam',
    'Stage3 Circle1 Distance Speed Collect Result_Dam',
    'WorkMode Collect Result_Fill1',
    'WorkMode Collect Result_Fill2'
]

# 피처를 학습 데이터에서 선택
train_x = df_train[features].astype(int)
train_y = df_train["target"]

# 모델 정의 (SVM, RBF 커널 사용)
model = SVC(kernel='rbf', random_state=RANDOM_STATE)

# 모델 학습
model.fit(train_x, train_y)

# 검증 데이터로 예측
val_x = df_val[features].astype(int)
val_y = df_val["target"]
val_pred = model.predict(val_x)

# 검증 데이터 결과 출력
print("\nValidation Results:")
print(classification_report(val_y, val_pred))
print(confusion_matrix(val_y, val_pred))

# 테스트 데이터 예측
test_data = pd.read_csv(os.path.join(ROOT_DIR, "testset/test.csv"))
df_test_x = test_data[features].astype(int)

# 예측 수행
test_pred = model.predict(df_test_x)

# 제출 파일 작성
df_sub = pd.read_csv(os.path.join(ROOT_DIR, "testset/submission.csv"))
df_sub["target"] = test_pred

# 결과 저장
submission_path = os.path.join(ROOT_DIR, "result/submission.csv")
df_sub.to_csv(submission_path, index=False)
print(f"Submission file saved to {submission_path}")
