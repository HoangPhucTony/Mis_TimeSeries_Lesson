import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from matplotlib import pyplot
from pandas.plotting import autocorrelation_plot
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

print('Bài tập 2: Kiểm tra White Noise \n')
print('1. Sinh dữ liệu white noise bằng NumPy (1000 điểm).\n')
print('2. Vẽ ACF → kiểm tra xem các lag có nằm trong ngưỡng tin cậy 95% không?.\n')
print('3. So sánh ACF của white noise với ACF của chuỗi AirPassengers.\n')

path = r'C:\DOHOANGPHUC\UTH\PhanTichChuoiThoiGianVaDuBao\BaiTap\Practice\Data\daily-min-temperatures.csv'
df = pd.read_csv(path, parse_dates=["Date"], index_col="Date") # Chuyển thành datetimes 
print('5 dòng đầu dữ liệu\n')
print(df.head(5))
print('\n')

print('Sinh dữ liệu white noise bằng NumPy (1000 điểm)\n')
np.random.seed(42)
white_noise = np.random.normal(loc=0, scale=1, size = 1000)

print('Vẽ ACF của white noise\n')
plot_acf(white_noise, lags=40)
plt.title('ACF - WhiteNoise(1000 điểm)')
plt.show()
print('\n')

print('So sánh ACF của white noise với ACF của chuỗi AirPassengers.\n')
df_air = pd.read_csv(r'C:\DOHOANGPHUC\UTH\PhanTichChuoiThoiGianVaDuBao\BaiTap\Practice\Data\AirPassengers.csv', parse_dates=['Month'], index_col='Month')

fig,ax = plt.subplots(2,1, figsize=(12, 8))
plot_acf(white_noise, lags=40, ax=ax[0])
ax[0].set_title('ACF - WhiteNoise(1000 điểm)')
plot_acf(df_air, lags=40, ax=ax[1])
ax[1].set_title('ACF - AirPassengers')
plt.show()
print('\n')

