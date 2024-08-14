import pandas as pd
import matplotlib.pyplot as plt

# 엑셀 파일 읽기
df = pd.read_excel('/Users/hyeonggeun_kim/Documents/LG Aimers/v2/dataset/undersampled_data.xlsx')

# 'target' 값이 맨 끝 열에 있으므로 마지막 열을 가져오기
target_column = df.columns[-1]
colors = df[target_column].map({'Normal': 'green', 'AbNormal': 'red'})

# 원하는 열의 데이터를 시각화
columns_to_plot = ['CURE START POSITION X Collect Result_Dam']

plt.figure(figsize=(10, 6))

for column in columns_to_plot:
    plt.scatter(df.index, df[column], c=colors, label=column, alpha=0.6, s=5)

plt.xlabel('Index')
plt.ylabel('Values')
plt.title('Data Visualization for Selected Columns')
plt.legend()
plt.show()
