import pandas as pd

#file_path = '/Users/hyeonggeun_kim/Documents/LG Aimers/v2/dataset/undersampled_data.xlsx'  # 엑셀 파일 경로
file_path = "C:/Users/hyunj/OneDrive/바탕 화면/LG Aimers/dataset/undersampled_data.xlsx"
df = pd.read_excel(file_path)

# 추출할 열 이름 목록
columns_to_extract = ['Process Desc._Dam','Head Zero Position Z Collect Result_Dam','Stage2 Line3 Distance Speed Collect Result_Dam','Stage3 Circle1 Distance Speed Collect Result_Dam',
                      'Process Desc._AutoClave','Chamber Temp. Collect Result_AutoClave','Chamber Temp. Unit Time_AutoClave','1st Pressure 1st Pressure Unit Time_AutoClave','3rd Pressure Unit Time_AutoClave',
                      'Process Desc._Fill1','WorkMode Collect Result_Fill1',
                      'Process Desc._Fill2','WorkMode Collect Result_Fill2','target'
]


# 특정 열 추출
df_extracted = df[columns_to_extract]

# 새로운 엑셀 파일로 저장
output_file_path = 'C:/Users/hyunj/OneDrive/바탕 화면/LG Aimers/dataset/preprocessed_data.xlsx'  # 저장할 새 엑셀 파일 경로
df_extracted.to_excel(output_file_path, index=False)