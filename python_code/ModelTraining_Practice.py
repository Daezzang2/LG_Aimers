import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# 1. 데이터 로드
df = pd.read_excel('/Users/hyeonggeun_kim/Documents/LG Aimers/v2/dataset/undersampled_data.xlsx')

# 2. 타겟 변수 생성
# 예시로 각 열에서 비정상 범위를 정의합니다. 필요에 따라 조건을 수정하세요.
df['target'] = (
    (df['Head Zero Position Z Collect Result_Dam'] > 300) | 
    ((df['1st Pressure 1st Pressure Unit Time_AutoClave'] > 50) &  (df['1st Pressure 1st Pressure Unit Time_AutoClave'] < 100)) | 
    ((df['3rd Pressure Unit Time_AutoClave'] > 50) & (df['3rd Pressure Unit Time_AutoClave'] < 60)) | 
    ((df['Chamber Temp. Collect Result_AutoClave'] > 30) &  (df['Chamber Temp. Collect Result_AutoClave'] < 35)) | 
    ((df['Chamber Temp. Unit Time_AutoClave'] > 50) & (df['Chamber Temp. Unit Time_AutoClave'] < 300)) |
    ((df['Stage2 Line3 Distance Speed Collect Result_Dam'] >= 5800) & (df['Stage2 Line3 Distance Speed Collect Result_Dam'] <= 6000)) |
    ((df['Stage3 Circle1 Distance Speed Collect Result_Dam'] >= 5800) & (df['Stage3 Circle1 Distance Speed Collect Result_Dam'] <= 6000)) |
    ((df['WorkMode Collect Result_Fill1'] > 1) & (df['WorkMode Collect Result_Fill1'] < 6)) |
    (df['WorkMode Collect Result_Fill2'] == 3)
).astype(int)


# 3. 특징과 타겟 분리
X = df.drop(['target'], axis=1)  # 'target' 열을 제외한 나머지 열들이 특징이 됩니다.
y = df['target']

# 4. 학습 및 테스트 데이터셋 분리
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. 모델 학습
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 6. 예측
y_pred = model.predict(X_test)

# 7. 평가
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

print(f'Accuracy: {accuracy}')
print('Confusion Matrix:')
print(conf_matrix)

# 8. 모델 저장 (선택 사항)
import joblib
joblib.dump(model, 'anomaly_detection_model.pkl')
