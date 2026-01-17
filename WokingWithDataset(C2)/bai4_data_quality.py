import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from matplotlib import pyplot
from scipy.stats import zscore

print('Bài tập 4: Kiểm tra và xử lý chất lượng dữ liệu \n')
print('1. Tạo dữ liệu thiếu (NaN) giả lập hoặc dùng dataset có missing.\n')
print('2. Kiểm tra missing: isna().sum()\n')
print('3. Xử lý missing bằng 2 cách: Forward-fill và Interpolation.\n')
print('4. Phát hiện outliers bằng: Rolling mean/variance; Z-score hoặc IQR \n')
print('5. Vẽ trước và sau khi xử lý. \n')

path = r'C:\DOHOANGPHUC\UTH\PhanTichChuoiThoiGianVaDuBao\BaiTap\Practice\Data\daily-min-temperatures.csv'
df = pd.read_csv(path, parse_dates=["Date"],index_col="Date")
print('5 dòng đầu dữ liệu\n')
print(df.head(5))
print('\n')

# Tạo dữ liệu thiếu (NaN) giả lập
df_miss = df.copy()

np.random.seed(42)
df_miss.loc[np.random.choice(df_miss.index,1000,replace=False),'Temp'] = np.nan # không chọn trùng 1000 giá trị 
print(df_miss) 
print('\n')

print('Kiểm tra missing: isna().sum()\n')
print(df_miss.isna().sum())
print('\n')

print('Xử lý missing bằng Forward-fill\n')
df_ffill = df_miss.copy()
df_ffill["Temp"] = df_ffill["Temp"].ffill()
print('\n')

print('Xử lý missing bằng Interpolation\n')
df_interp = df_miss.copy()
df_interp["Temp"] = df_interp["Temp"].interpolate()
print('\n')

print('Kiểm tra lại missing trước khi xử lý\n')
print(df_miss.isna().sum())
print('\n')
print('Kiểm tra lại missing sau khi xử lý bằng Forward-fill\n')
print(df_ffill.isna().sum())
print('\n')
print('Kiểm tra lại missing sau khi xử lý bằng Interpolation\n')
print(df_interp.isna().sum())
print('\n')

print('Vẽ trước và sau khi xử lý missing\n')
fig, ax = plt.subplots(3, 1, figsize=(12, 10))
ax[0].plot(df_miss.index, df_miss['Temp'], label='Missing Data', color='red')
ax[0].set_title('Data giả lập Missing')
ax[1].plot(df_ffill.index, df_ffill['Temp'], label='Forward-fill', color='blue')
ax[1].set_title('Xử lý bằng Forward-fill')
ax[2].plot(df_interp.index, df_interp['Temp'], label='Interpolation', color='green')
ax[2].set_title('Xử lý bằng Interpolation')
plt.xlabel('Date')
plt.tight_layout()
plt.show()
print('\n')



print('Phát hiện outliers bằng Z-score\n')
df_out = df_interp.copy()   # dùng dữ liệu đã xử lý missing interp
df_out['zscore'] = zscore(df_out['Temp'])

outliers = df_out[np.abs(df_out['zscore']) > 3]

print('Số lượng outliers phát hiện:', outliers.shape[0])
print('\n')

df_clean = df_out.copy()
df_clean.loc[np.abs(df_clean['zscore']) > 3, 'Temp'] = np.nan
df_clean['Temp'] = df_clean['Temp'].interpolate()

print('Vẽ trước và sau khi xử lý outliers\n')
fig, ax = plt.subplots(2, 1, figsize=(12, 8))
ax[0].plot(df_interp.index, df_interp['Temp'], label='Trước xử lý outliers', color='orange')
ax[0].set_title('Trước khi xử lý outliers')
ax[0].legend()
ax[1].plot(df_clean.index, df_clean['Temp'], label='Sau xử lý outliers', color='green')
ax[1].set_title('Sau khi xử lý outliers')
ax[1].legend()

plt.tight_layout()
plt.show()

