import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.arima.model import ARIMA

from sklearn.metrics import mean_absolute_error, mean_squared_error



path = r'C:\DOHOANGPHUC\UTH\PhanTichChuoiThoiGianVaDuBao\BaiTap\Practice\Data\AirPassengers.csv'
df = pd.read_csv(path, parse_dates=['Month'], index_col='Month')

print(df.head(5))

df.plot()
plt.title('Monthly Airline Passengers')
plt.xlabel('Month')
plt.ylabel('Number of Passengers')
plt.show()


ADF_ts = df['Passengers']
result = adfuller(ADF_ts)
print('ADF test Statistic:', result[0]) # Test statistic
print('p-value:', result[1])  # p-value: Xác suất bác bỏ H₀
print('Critical Values:')
for key, value in result[4].items(): # Critical values
    print(f'   {key}: {value}')

if result[1] <= 0.05:
    print(f'p-value = {result[1]} <= 0.05 => Chuỗi có tính dừng => Bác bỏ H0 ')
else:
    print(f'p-value = {result[1]} > 0.05 => Chuỗi không có tính dừng => Chấp nhận H0')
print('\n')

df_diff1 = df['Passengers'].diff() # sai phân bậc 1
df_diff1 = df_diff1.dropna()
df_seasonal = df_diff1.diff(12) # sai phân mùa vụ
df_seasonal = df_seasonal.dropna()
df_seasonal.plot()
plt.title('Monthly Airline Passengers - Sai phân bậc 1 & Sai phân mùa vụ')
plt.xlabel('Month')
plt.ylabel('Differenced Value')
plt.show()


plot_acf(df_seasonal, lags=40)
plt.title('ACF sau sai phân bậc 1 & sai phân mùa vụ')
plt.show()
plot_pacf(df_seasonal, lags=40)
plt.title('PACF sau sai phân bậc 1 & sai phân mùa vụ')
plt.show()

df_sarima = df['Passengers']
n = int(len(df_sarima) * 0.8)
train = df_sarima[:n]
test = df_sarima[n:]

p, d, q = 1, 1, 1
P, D, Q, s = 1, 1, 1, 12

print(f"\n--- Đang chạy mô hình SARIMAX({p},{d},{q})x({P},{D},{Q},{s}) ---")
model = SARIMAX(train, order=(p,d,q), seasonal_order=(P,D,Q,s))
results = model.fit()

print("AIC của mô hình SARIMA:", results.aic)
print("BIC của mô hình SARIMA:", results.bic)

residuals = results.resid
print("\n--- Biểu đồ chẩn đoán phần dư SARIMA ---")
residuals.plot(title='Residuals (SARIMA) theo thời gian')
plt.show()
plot_acf(residuals, lags=40)
plt.title('ACF của Residuals (SARIMA)')
plt.show()

forecast = results.forecast(steps=len(test))

plt.plot(train, label='Train')
plt.plot(test, label='Test')
plt.plot(test.index, forecast, label='Forecast')
plt.legend()
plt.title('SARIMA Forecast')
plt.show()

mae = mean_absolute_error(test, forecast)
rmse = np.sqrt(mean_squared_error(test, forecast))
mape = np.mean(np.abs((test - forecast) / test)) * 100
print("\n--- Đánh giá mô hình SARIMA ---")
print("MAE:", mae)
print("RMSE:", rmse)
print("MAPE (%):", mape)


model_arima = ARIMA(train, order=(1,1,1))
result_arima = model_arima.fit()
forecast_arima = result_arima.forecast(steps=len(test))
plt.plot(train, label='Train')
plt.plot(test, label='Test')
plt.plot(test.index, forecast_arima, label='Forecast')
plt.legend()
plt.title('ARIMA Forecast')
plt.show()

mae_arima = mean_absolute_error(test, forecast_arima)
rmse_arima = np.sqrt(mean_squared_error(test, forecast_arima))

print("ARIMA MAE:", mae_arima)
print("ARIMA RMSE:", rmse_arima)




