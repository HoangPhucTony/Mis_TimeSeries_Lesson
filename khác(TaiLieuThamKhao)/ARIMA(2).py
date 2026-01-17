# ======================================================
# IMPORT
# ======================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_squared_error

plt.style.use('seaborn-v0_8')

# ======================================================
# HÀM TIỆN ÍCH
# ======================================================
def adf_test(series, title=''):
    print(f'ADF Test: {title}')
    result = adfuller(series)
    print('ADF Statistic:', result[0])
    print('p-value:', result[1])
    print('Critical Values:')
    for k, v in result[4].items():
        print(f'   {k}: {v}')
    if result[1] <= 0.05:
        print('=> Chuỗi DỪNG (Bác bỏ H0)\n')
    else:
        print('=> Chuỗi KHÔNG DỪNG (Chấp nhận H0)\n')


def evaluate_model(test, forecast, name='MODEL'):
    mae = mean_absolute_error(test, forecast)
    rmse = np.sqrt(mean_squared_error(test, forecast))
    mape = np.mean(np.abs((test - forecast) / test)) * 100

    print(f'\n--- Đánh giá mô hình {name} ---')
    print('MAE:', mae)
    print('RMSE:', rmse)
    print('MAPE (%):', mape)


# ======================================================
# BÀI 1: EDA
# ======================================================
print('\nBài 1: Khám phá dữ liệu (EDA)')
path = r"C:\DOHOANGPHUC\UTH\PhanTichChuoiThoiGianVaDuBao\BaiTap\Practice\Data\Monthly_Retail_Sales.csv"
df = pd.read_csv(path, parse_dates=['date'], index_col='date')

print(df.info())

plt.figure(figsize=(10, 4))
plt.plot(df['value'])
plt.title('Monthly Retail Sales')
plt.xlabel('Date')
plt.ylabel('Sales Value')
plt.show()

print('Kết luận:')
print('- Chuỗi có xu hướng tăng')
print('- Có mùa vụ theo tháng (multiplicative)')
print('- Chuỗi có khả năng không dừng\n')


# ======================================================
# BÀI 2: ADF CHUỖI GỐC
# ======================================================
print('Bài 2: Kiểm định ADF chuỗi gốc')
adf_test(df['value'], 'Chuỗi gốc')


# ======================================================
# BÀI 3: SAI PHÂN BẬC 1
# ======================================================
print('Bài 3: Sai phân & kiểm định lại')

df_diff1 = df['value'].diff().dropna()

plt.figure(figsize=(10, 4))
plt.plot(df_diff1)
plt.title('Sai phân bậc 1')
plt.show()

adf_test(df_diff1, 'Sai phân bậc 1')

print('Kết luận:')
print('- d = 1')
print('- Chuỗi chưa dừng hoàn toàn do còn mùa vụ\n')


# ======================================================
# BÀI 4: ACF & PACF (VẼ CHUNG ĐỂ SO SÁNH)
# ======================================================
print('Bài 4: Phân tích ACF & PACF')

fig, ax = plt.subplots(2, 1, figsize=(10, 6))
plot_acf(df_diff1, lags=40, ax=ax[0])
plot_pacf(df_diff1, lags=40, method='ywm', ax=ax[1])

ax[0].set_title('ACF – Sai phân bậc 1')
ax[1].set_title('PACF – Sai phân bậc 1')
plt.tight_layout()
plt.show()

print('Kết luận:')
print('- ACF có spike tại 12, 24 → mùa vụ')
print('- Gợi ý ARIMA(1,1,1) nếu bỏ mùa vụ\n')


# ======================================================
# BÀI 5: ARIMA
# ======================================================
print('Bài 5: ARIMA')

n = int(len(df) * 0.8)
train, test = df['value'][:n], df['value'][n:]

model_arima = ARIMA(train, order=(1, 1, 1))
res_arima = model_arima.fit()

print('AIC:', res_arima.aic)
print('BIC:', res_arima.bic)

# Residual diagnostics
fig, ax = plt.subplots(2, 1, figsize=(10, 6))
res_arima.resid.plot(ax=ax[0], title='Residuals ARIMA')
plot_acf(res_arima.resid, lags=30, ax=ax[1])
plt.tight_layout()
plt.show()

forecast_arima = res_arima.forecast(len(test))

plt.figure(figsize=(10, 4))
plt.plot(train, label='Train')
plt.plot(test, label='Test')
plt.plot(test.index, forecast_arima, label='Forecast')
plt.legend()
plt.title('ARIMA Forecast')
plt.show()

evaluate_model(test, forecast_arima, 'ARIMA')


# ======================================================
# BÀI 5.5: SARIMA
# ======================================================
print('\nBài 5.5: SARIMA')

model_sarima = SARIMAX(
    train,
    order=(1, 1, 1),
    seasonal_order=(1, 1, 1, 12)
)
res_sarima = model_sarima.fit()

print('AIC:', res_sarima.aic)
print('BIC:', res_sarima.bic)

fig, ax = plt.subplots(2, 1, figsize=(10, 6))
res_sarima.resid.plot(ax=ax[0], title='Residuals SARIMA')
plot_acf(res_sarima.resid, lags=30, ax=ax[1])
plt.tight_layout()
plt.show()

forecast_sarima = res_sarima.forecast(len(test))

plt.figure(figsize=(10, 4))
plt.plot(train, label='Train')
plt.plot(test, label='Test')
plt.plot(test.index, forecast_sarima, label='Forecast')
plt.legend()
plt.title('SARIMA Forecast')
plt.show()

evaluate_model(test, forecast_sarima, 'SARIMA')
