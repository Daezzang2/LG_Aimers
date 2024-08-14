import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import joblib

# 1. 데이터 로드
df = pd.read_excel('/Users/hyeonggeun_kim/Documents/LG Aimers/v2/dataset/undersampled_data.xlsx')

# 2. 타겟 변수 생성
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

X = df[features]
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
joblib.dump(model, 'anomaly_detection_model.pkl')

# 9. 테스트 데이터에 대해 예측 수행 (새로운 test 데이터)
test_data = pd.read_csv('/Users/hyeonggeun_kim/Documents/LG Aimers/v2/testset/test.csv')

# 테스트 데이터에서 Set ID와 사용하고자 하는 9개의 열만 선택합니다.
Set_IDs = test_data['Set ID']
X_test_new = test_data[features]

# 예측 수행
test_predictions = model.predict(X_test_new)

# 10. 예측 결과를 Normal과 AbNormal로 변환
# 0은 'Normal', 1은 'AbNormal'로 변환
test_predictions_str = ['Normal' if pred == 0 else 'AbNormal' for pred in test_predictions]

# 11. 예측 결과를 엑셀 파일로 저장
# 예측 결과와 Set ID를 합쳐서 새로운 데이터프레임 생성
results_df = pd.DataFrame({
    'Set ID': Set_IDs,
    'target': test_predictions_str
})

# 결과를 엑셀 파일로 저장
output_path = '/Users/hyeonggeun_kim/Documents/LG Aimers/v2/result/test_predictions.xlsx'
results_df.to_excel(output_path, index=False)

print(f"Predictions saved to {output_path}")
