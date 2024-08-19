import pandas as pd

# 엑셀 파일 경로
file_path = '/Users/hyeonggeun_kim/Documents/LG Aimers/v2/dataset/undersampled_data.xlsx'
df = pd.read_excel(file_path)

# 결측치가 있는 열 제거
df_cleaned = df.dropna(axis=1)

# 숫자형 열만 선택
df_numeric = df_cleaned.select_dtypes(include=[float, int])

# 모든 값이 동일한 열 제거
columns_to_drop = [col for col in df_numeric.columns if df_numeric[col].nunique() == 1]
df_numeric = df_numeric.drop(columns=columns_to_drop)

# 새로운 엑셀 파일로 저장
output_file_path = '/Users/hyeonggeun_kim/Documents/LG Aimers/v2/dataset/preprocessed_data_v2.xlsx'
df_numeric.to_excel(output_file_path, index=False)

print(f"데이터가 {output_file_path}에 성공적으로 저장되었습니다.")
