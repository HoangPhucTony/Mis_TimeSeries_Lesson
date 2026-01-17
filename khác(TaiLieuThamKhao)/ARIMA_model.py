import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX 
from sklearn.metrics import mean_absolute_error, mean_squared_error

print('\n')
# Bài 1: Khám phá dữ liệu (EDA) 
print('Bài 1: Khám phá dữ liệu (EDA) ')
path = r"C:\DOHOANGPHUC\UTH\PhanTichChuoiThoiGianVaDuBao\BaiTap\Practice\Data\Monthly_Retail_Sales.csv"
df = pd.read_csv(path,parse_dates=['date'], index_col='date')

print(df.info())

df.plot()
plt.title('Monthly Retail Sales')
plt.xlabel('Date')
plt.ylabel('Sales Value')
plt.show()

print('Kết luận: Chuỗi có xu hướng tăng dần vì giá trị values tăng dần \n')
print('Có dấu hiệu mùa vụ, mùa vụ theo tháng, kiểu mùa vụ multiplicative (nhân) \n')
print('Chuỗi có khả năng không dừng vì có xu hướng và mùa vụ, mean và variance không ổn định \n')

print('\n')
# Bài 2: Kiểm định ADF trên chuỗi gốc 
print('Bài 2: Kiểm định ADF trên chuỗi gốc')
# adfuller() là hàm thực hiện kiểm định Augmented Dickey–Fuller (ADF) -> Chuỗi có dừng hay ko 
ADF_ts = df['value']
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

# Bài 3: Sai phân & kiểm định lại # biến chuỗi không dừng → chuỗi dừng
# SAI PHÂN (DIFFERENCING)
print('Bài 3: Sai phân & kiểm định lại')
df_diff1 = df['value'].diff() # sai phân bậc 1
df_diff1 = df_diff1.dropna()

df_diff1.plot()
plt.title('Monthly Retail Sales - Sai phân bậc 1')
plt.xlabel('Date')
plt.ylabel('Differenced Value')
plt.show()


result_diff1 = adfuller(df_diff1)
print('ADF test Statistic, sai phân bậc 1:', result_diff1[0]) # Test statistic
print('p-value, sai phân bậc 1:', result_diff1[1])
print('Critical Values, sai phân bậc 1:')
for key, value in result_diff1[4].items(): # Critical values
    print(f'   {key}: {value}')

if result_diff1[1] <= 0.05:
    print(f'p-value = {result_diff1[1]} <= 0.05 => Chuỗi DỪNG sau sai phân \n')
else:
    print(f'p-value = {result_diff1[1]} > 0.05 => Chuỗi CHƯA DỪNG sau sai phân \n')


print('Kết luận: Sau sai phân bậc 1, chuỗi chưa dừng vì kiểm định ADF cho p-value > 0.05 \n')
print('Giá trị d trong ARIMA, d = 1 ') # d = số lần sai phân KHÔNG MÙA VỤ để loại trend
print('Phần chưa dừng là do mùa vụ, không phải do trend \n')

# Bài 4: Phân tích ACF & PACF 
print('Bài 4: Phân tích ACF & PACF')

plot_acf(df_diff1, lags=40)
plt.title('ACF - Sai phân bậc 1 (Còn seasonal)')
plt.show()

plot_pacf(df_diff1, lags=40, method='ywm')
plt.title('PACF - Sai phân bậc 1 (Còn seasonal)')
plt.show()

print('Kết luận: ACF thể hiện các spike dương mạnh tại các lag 12, 24, 36,... (dấu hiệu mùa vụ) \n')
print('PACF xuất hiện các gai lớn tại các lag 11–13  \n')
print('Sau sai phân bậc 1, chuỗi vẫn chưa dừng do còn yếu tố mùa vụ \n')
print('Nếu bỏ qua yếu tố mùa vụ, thì ta có thể chọn mô hình hóa ARIMA (1,1,1) hoặc (1,1,0)')

#########
print('4.5 Sai phân mùa + Kiểm định lại \n')

df_seasonal_diff = df_diff1.diff(12) # Sai phân mùa vụ (seasonal differencing)
df_seasonal_diff = df_seasonal_diff.dropna()
df_seasonal_diff.plot()
plt.title('Monthly Retail Sales - Sai phân bậc 1 và sai phân mùa vụ (12)')
plt.xlabel('Date')
plt.ylabel('Seasonally Differenced Value')
plt.show()

result_seasonal_diff = adfuller(df_seasonal_diff)
print('ADF test Statistic, sai phân mùa:', result_seasonal_diff[0]) #
print('p-value, sai phân mùa:', result_seasonal_diff[1])
print('Critical Values, sai phân mùa:')
for key, value in result_seasonal_diff[4].items(): # Critical values
    print(f'   {key}: {value}')
if result_seasonal_diff[1] <= 0.05:
    print(f'p-value = {result_seasonal_diff[1]} <= 0.05 => Chuỗi DỪNG sau sai phân mùa \n')
else:
    print(f'p-value = {result_seasonal_diff[1]} > 0.05 => Chuỗi CHƯA DỪNG sau sai phân mùa \n')


plot_acf(df_seasonal_diff, lags=40)
plt.title('ACF - Sai phân bậc 1 + Sai phân mùa (Chuỗi đã dừng)')
plt.show()

plot_pacf(df_seasonal_diff, lags=40, method='ywm')
plt.title('PACF - Sai phân bậc 1 + Sai phân mùa (Chuỗi đã dừng)')
plt.show()


# Bài 5: Thiết lập ARIMA / SARIMA – Dự đoán – Đánh giá
print('Bài 5: Thiết lập ARIMA ' )
df_sarima = df['value']
print('5.1 Chia dữ liêu thành: train(80%), test(20%). Giữ nguyên thứ tự thời gian (không shuffle) ')
n = int(len(df_sarima) * 0.8)
train = df_sarima[:n]
test = df_sarima[n:]

print("\nTrain size:", len(train))
print("Test size:", len(test))

# 2. Thiết lập mô hình ARIMA  
# – Dựa trên ACF/PACF, đề xuất 1–2 mô hình ARIMA. 
# Ví dụ: ARIMA(1,1,1), 
# ARIMA(2,1,1) 
# – Fit mô hình trên tập train. 
# Ghi lại AIC và BIC.
p, d, q = 1, 1, 1
print(f"\n--- Đang chạy mô hình ARIMA({p},{d},{q}) ---")

model = ARIMA(train, order=(p, d, q))
results = model.fit()

print("AIC:", results.aic)
print("BIC:", results.bic)

# 3. Chẩn đoán phần dư (Residual Diagnostics): 
# – Vẽ Residuals theo thời gian, ACF của residuals 
# – Kiểm tra: Residuals có mean ≈ 0? Có tự tương quan không?
# ARIMA không xử lý seasonality nên residuals còn autocorrelation.
residuals = results.resid
print("\n--- Biểu đồ chẩn đoán phần dư ARIMA ---")
residuals.plot(title='Residuals (ARIMA) theo thời gian')
plt.show()

plot_acf(residuals, lags=30)
plt.title('ACF của Residuals')
plt.show()

# 4. Dự đoán (Forecasting): 
# – Dự đoán trên tập test 
# – Vẽ đồ thị: Train, Test (thực tế), Forecast
forecast = results.forecast(steps=len(test))
plt.plot(train, label='Train')
plt.plot(test, label='Test (Thực tế)')
plt.plot(test.index, forecast, label='Forecast')
plt.legend()
plt.title('ARIMA Forecast')
plt.show()


# 5. Đánh giá ARIMA. Chỉ số đánh giá bắt buộc: MAE, RMSE, MAPE 
mae = mean_absolute_error(test, forecast)
rmse = np.sqrt(mean_squared_error(test, forecast))
mape = np.mean(np.abs((test - forecast) / test)) * 100

print("\n--- Đánh giá mô hình ARIMA ---")
print("MAE:", mae)
print("RMSE:", rmse)
print("MAPE (%):", mape)

# Bài 5.5 : Thiết lập SARIMA – Dự đoán – Đánh giá
print('Bài 5.5: Thiết lập SARIMA')

df_sarima = df['value']

print('5.1 Chia dữ liêu thành: train(80%), test(20%). Giữ nguyên thứ tự thời gian (không shuffle)')
n = int(len(df_sarima) * 0.8)
train = df_sarima[:n]
test = df_sarima[n:]

print("\nTrain size:", len(train))
print("Test size:", len(test))


p, d, q = 1, 1, 1
P, D, Q, s = 1, 1, 1, 12

print(f"\n--- Đang chạy mô hình SARIMA({p},{d},{q})x({P},{D},{Q},{s}) ---")

model = SARIMAX(train,order=(p, d, q),seasonal_order=(P, D, Q, s))

results = model.fit()

print("AIC:", results.aic)
print("BIC:", results.bic)

residuals = results.resid

print("\n--- Biểu đồ chẩn đoán phần dư SARIMA ---")
residuals.plot(title='Residuals (SARIMA) theo thời gian')
plt.show()

plot_acf(residuals, lags=30)
plt.title('ACF của Residuals')
plt.show()

forecast = results.forecast(steps=len(test))

plt.plot(train, label='Train')
plt.plot(test, label='Test (Thực tế)')
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

