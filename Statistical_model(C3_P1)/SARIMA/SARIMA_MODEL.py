import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging
import os

from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error

LOG_DIR = r'C:\DOHOANGPHUC\UTH\PhanTichChuoiThoiGianVaDuBao\BaiTap\Practice\Statistical_model(C3_P1)\SARIMA'
os.makedirs(LOG_DIR, exist_ok=True)
LOG_PATH = os.path.join(LOG_DIR, 'sarima.log')

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[
        logging.FileHandler(LOG_PATH, mode="w", encoding="utf-8"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger()

# Ảnh 
PLOT_DIR = r'C:\DOHOANGPHUC\UTH\PhanTichChuoiThoiGianVaDuBao\BaiTap\Practice\Statistical_model(C3_P1)\Plot\SARIMA'
os.makedirs(PLOT_DIR, exist_ok=True)

#dataset 
path = r'C:\DOHOANGPHUC\UTH\PhanTichChuoiThoiGianVaDuBao\BaiTap\Practice\Data\AirPassengers.csv'
df = pd.read_csv(path, parse_dates=['Month'], index_col='Month')

logger.info("="*60)
logger.info("DATASET: AIRPASSENGERS – MONTHLY AIRLINE PASSENGERS")
logger.info("="*60)
logger.info(f"Số quan sát: {len(df)}")
logger.info(f"Thời gian: {df.index.min()} → {df.index.max()}\n")

# BÀI 1
logger.info("="*60)
logger.info("BÀI 1: VẼ CHUỖI THỜI GIAN – NHẬN DIỆN MÙA VỤ")
logger.info("="*60)

df.plot()
plt.title("Monthly Airline Passengers")
plt.xlabel("Time")
plt.ylabel("Passengers")
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, "Bai1_TimeSeries.png"))
plt.show()

logger.info(
    "- Chuỗi có xu hướng tăng rõ rệt theo thời gian.\n"
    "- Mùa vụ lặp lại theo chu kỳ 12 tháng.\n"
    "- Biên độ dao động tăng dần → phương sai không ổn định.\n"
    "- Chuỗi KHÔNG dừng.\n"
)

# BÀI 2
logger.info("="*60)
logger.info("BÀI 2: KIỂM ĐỊNH ADF CHUỖI GỐC")
logger.info("="*60)

adf_result = adfuller(df['Passengers'])

logger.info(f"ADF Statistic : {adf_result[0]:.6f}")
logger.info(f"p-value       : {adf_result[1]:.6f}")
logger.info("Critical Values:")
for k, v in adf_result[4].items():
    logger.info(f"  {k}: {v:.6f}")

if adf_result[1] < 0.05:
    logger.info("⇒ Bác bỏ H0 → Chuỗi dừng\n")
else:
    logger.info("⇒ Không bác bỏ H0 → Chuỗi KHÔNG dừng\n")

# BÀI 3
logger.info("="*60)
logger.info("BÀI 3: SAI PHÂN THƯỜNG + SAI PHÂN MÙA VỤ")
logger.info("="*60)

# d = 1
df_diff = df['Passengers'].diff().dropna()

# D = 1, s = 12
df_diff_seasonal = df_diff.diff(12).dropna()

df_diff_seasonal.plot()
plt.title("Differenced Series (d=1, D=1, s=12)")
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, "Bai3_Differenced_d1_D1.png"))
plt.show()

logger.info(f"Số quan sát sau sai phân: {len(df_diff_seasonal)}")

# KIỂM ĐỊNH ADF SAU SAI PHÂN (d=1, D=1)
logger.info("KIỂM ĐỊNH ADF SAU SAI PHÂN (d=1, D=1)")

adf_diff_seasonal = adfuller(df_diff_seasonal)

logger.info(f"- ADF Statistic : {adf_diff_seasonal[0]:.6f}")
logger.info(f"- p-value       : {adf_diff_seasonal[1]:.6f}")
logger.info("- Critical Values:")
for level, value in adf_diff_seasonal[4].items():
    logger.info(f"   + {level} : {value:.6f}")

if adf_diff_seasonal[1] < 0.05:
    logger.info(
        "- KẾT LUẬN ADF: p-value < 0.05 "
        "⇒ Bác bỏ H0 "
        "⇒ Chuỗi ĐÃ dừng sau sai phân thường + mùa vụ.\n"
    )
else:
    logger.info(
        "- KẾT LUẬN ADF: p-value ≥ 0.05 "
        "⇒ Chưa bác bỏ H0 "
        "⇒ Chuỗi CHƯA dừng.\n"
    )


# BÀI 4
logger.info("="*60)
logger.info("BÀI 4: ACF / PACF → XÁC ĐỊNH (p, q, P, Q)")
logger.info("="*60)

n = len(df_diff_seasonal)
threshold = 1.96 / np.sqrt(n)

logger.info(f"Số quan sát (n): {n}")
logger.info(f"Ngưỡng ý nghĩa 95%: ±{threshold:.4f}\n")

# ACF
logger.info("4.1 PHÂN TÍCH ACF (lag 1 → 10)")

acf_values = acf(df_diff_seasonal, nlags=10)

for lag in range(1, 11):
    value = acf_values[lag]
    significance = "CÓ ý nghĩa" if abs(value) > threshold else "KHÔNG ý nghĩa"
    logger.info(
        f"  Lag {lag:2d}: ACF = {value:+.4f} → {significance}"
    )

plt.figure()
plot_acf(df_diff_seasonal, lags=40)
plt.title("ACF sau sai phân (d=1, D=1)")
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, "Bai4_ACF.png"))
plt.show()

# PACF
logger.info("\n4.2 PHÂN TÍCH PACF (lag 1 → 10)")

pacf_values = pacf(df_diff_seasonal, nlags=10, method="ywm")

for lag in range(1, 11):
    value = pacf_values[lag]
    significance = "CÓ ý nghĩa" if abs(value) > threshold else "KHÔNG ý nghĩa"
    logger.info(
        f"  Lag {lag:2d}: PACF = {value:+.4f} → {significance}"
    )

plt.figure()
plot_pacf(df_diff_seasonal, lags=40, method="ywm")
plt.title("PACF sau sai phân (d=1, D=1)")
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, "Bai4_PACF.png"))
plt.show()

# KẾT LUẬN
logger.info("\n4.3 KẾT LUẬN ACF & PACF")

logger.info(
    "- ACF có spike vượt ngưỡng tại lag 1 → gợi ý MA(1).\n"
    "- PACF có spike vượt ngưỡng tại lag 1 → gợi ý AR(1).\n"
    "- ACF xuất hiện spike rõ tại lag 12 → MA mùa vụ (Q = 1).\n"
    "- PACF xuất hiện spike tại lag 12 → AR mùa vụ (P = 1).\n"
)

logger.info(
    "→ Chọn mô hình: SARIMA(1,1,1)(1,1,1,12)\n"
)


# BÀI 5
logger.info("="*60)
logger.info("BÀI 5: FIT SARIMA")
logger.info("="*60)

y = df['Passengers']
train_size = int(len(y) * 0.8)

train = y.iloc[:train_size]
test  = y.iloc[train_size:]

model_sarima = SARIMAX(
    train,
    order=(1,1,1),
    seasonal_order=(1,1,1,12),
    enforce_stationarity=False,
    enforce_invertibility=False
)

sarima_fit = model_sarima.fit()

logger.info("Huấn luyện SARIMA hoàn tất")
logger.info(f"AIC: {sarima_fit.aic:.4f}")
logger.info(f"BIC: {sarima_fit.bic:.4f}\n")

# BÀI 6
logger.info("="*60)
logger.info("BÀI 6: FORECAST & ĐÁNH GIÁ")
logger.info("="*60)

forecast_sarima = sarima_fit.forecast(steps=len(test))
forecast_sarima.index = test.index

plt.figure(figsize=(10,5))
plt.plot(train, label="Train")
plt.plot(test, label="Test")
plt.plot(forecast_sarima, label="SARIMA Forecast", linestyle="--")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, "Bai6_SARIMA_Forecast.png"))
plt.show()

mae_sarima = mean_absolute_error(test, forecast_sarima)
rmse_sarima = np.sqrt(mean_squared_error(test, forecast_sarima))

logger.info(f"MAE  : {mae_sarima:.4f}")
logger.info(f"RMSE : {rmse_sarima:.4f}\n")

# BÀI 7
logger.info("="*60)
logger.info("BÀI 7: SO SÁNH SARIMA VỚI ARIMA")
logger.info("="*60)

arima_model = ARIMA(train, order=(1,1,1))
arima_fit = arima_model.fit()

forecast_arima = arima_fit.forecast(steps=len(test))
forecast_arima.index = test.index

mae_arima = mean_absolute_error(test, forecast_arima)
rmse_arima = np.sqrt(mean_squared_error(test, forecast_arima))

logger.info("ARIMA(1,1,1):")
logger.info(f"MAE  : {mae_arima:.4f}")
logger.info(f"RMSE : {rmse_arima:.4f}\n")

logger.info("KẾT LUẬN:")
logger.info(
    "- ARIMA không xử lý được mùa vụ.\n"
    "- SARIMA mô hình hóa tốt cả xu hướng và mùa vụ.\n"
    "- SARIMA cho sai số MAE, RMSE thấp hơn.\n"
    "- SARIMA phù hợp hơn với AirPassengers.\n"
)
