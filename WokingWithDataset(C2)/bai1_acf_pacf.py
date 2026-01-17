import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from matplotlib import pyplot
from pandas.plotting import autocorrelation_plot
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

print('Bài tập 1: Tính và phân tích ACF/PACF \n')
print('1. Tải và xem trước dữ liệu (head, info, plot).\n')
print('2. Tính ACF, PACF đến 40 lags.\n')
print('3. Kết luận: \n')

path = r'C:\DOHOANGPHUC\UTH\PhanTichChuoiThoiGianVaDuBao\BaiTap\Practice\Data\daily-min-temperatures.csv'
df = pd.read_csv(path, parse_dates=["Date"], index_col="Date")

print('5 dòng đầu dữ liệu\n')
print(df.head(5))
print('\n')

print('Thông tin dữ liệu\n')
print(df.info())
print('\n')

print('Mô tả dữ liệu\n')
print(df.describe())
print('\n')

print('Vẽ Time Series Line plot\n')
df.plot()
plt.xlabel=('Date')
plt.ylabel=('Temperature')
plt.title('Daily Minimum Temperatures')
plt.show()
print('\n')

# print('Tính ACF và PACF đến 40 lags\n')
# dc_acf = plot_acf(df, lags=40)
# plt.title('ACF')
# plt.show()
# dc_pacf = plot_pacf(df, lags=40)
# plt.title('PACF')
# plt.show()
# print('\n')

print('Tính ACF và PACF đến 40 lags\n')
fig,ax = plt.subplots(2,1, figsize=(12, 8))
plot_acf(df, lags=40, ax=ax[0])
plot_pacf(df, lags=40, ax=ax[1])
ax[0].set_title('ACF')
ax[1].set_title('PACF')
plt.show()

