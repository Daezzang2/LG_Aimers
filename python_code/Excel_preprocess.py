import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# 데이터 로드
df = pd.read_excel('/Users/hyeonggeun_kim/Documents/LG Aimers/v2/dataset/undersampled_data.xlsx')

# 타겟 변수 레이블 인코딩 (Normal -> 0, AbNormal -> 1)
label_encoder = LabelEncoder()
df['target'] = label_encoder.fit_transform(df['target'])

# 숫자형 데이터만 선택 (int, float 타입)
df_numeric = df.select_dtypes(include=[np.number])

# 피처와 타겟 변수 분리
X = df_numeric.drop(columns=['target'])  # 'target' 열을 제외한 피처 데이터
y = df_numeric['target']  # 타겟 변수

# 데이터 분할
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=42)

# 모델 학습
model = LGBMClassifier(random_state=42)
model.fit(X_train, y_train)

# 피처 중요도 가져오기
feature_importances = model.feature_importances_
feature_names = X_train.columns

# 중요도와 피처 이름을 DataFrame으로 정리
importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': feature_importances
})

# 중요도 순으로 정렬
importance_df = importance_df.sort_values(by='Importance', ascending=False)

# 상위 20개 피처 시각화
plt.figure(figsize=(10, 8))
plt.barh(importance_df['Feature'][:20], importance_df['Importance'][:20])
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.title('Top 20 Important Features')
plt.gca().invert_yaxis()
plt.show()

# 상위 20개 피처의 데이터 타입 출력 (로그 출력)
top_20_features = importance_df['Feature'][:20]
for feature in top_20_features:
    print(f"Feature: {feature}, Data Type: {df[feature].dtype}")

# 중요도 DataFrame 출력
importance_df.head(20)
