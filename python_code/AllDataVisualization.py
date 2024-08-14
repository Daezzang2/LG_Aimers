import pandas as pd
import matplotlib.pyplot as plt
import os

# 엑셀 파일 읽기
df = pd.read_excel('/Users/hyeonggeun_kim/Documents/LG Aimers/v2/dataset/undersampled_data.xlsx')

# 'target' 값이 맨 끝 열에 있으므로 마지막 열을 가져오기
target_column = df.columns[-1]
colors = df[target_column].map({'Normal': 'green', 'AbNormal': 'red'})

# 이미지 저장 경로 설정
save_dir = os.path.expanduser('/Users/hyeonggeun_kim/Documents/LG Aimers/v2/dataset/data_visualization')
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# 모든 열에 대해 시각화 수행 ('target' 열 제외)
columns_to_plot = df.columns[:-1]

for column in columns_to_plot:
    # 열의 데이터가 숫자 타입인지 확인하고 NaN 값이 있는지 확인
    if not pd.api.types.is_numeric_dtype(df[column]) or df[column].isnull().all():
        print("건너뜁니다.")
        continue
    
    # NaN 값을 가진 행을 제외하고 시각화
    plt.figure(figsize=(10, 6))
    plt.scatter(df.index, df[column].dropna(), c=colors[df[column].dropna().index], label=column, alpha=0.6, s=5)
    plt.xlabel('Index')
    plt.ylabel('Values')
    plt.title(f'Data Visualization for {column}')
    plt.legend()
    # 이미지 파일로 저장
    plt.savefig(os.path.join(save_dir, f'{column}.png'))
    plt.close()  # 플롯을 닫아 다음 플롯에 영향을 주지 않도록 함
