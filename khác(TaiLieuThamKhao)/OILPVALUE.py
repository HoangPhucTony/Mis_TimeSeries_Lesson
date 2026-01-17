import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox
from scipy import stats
from scipy.stats import jarque_bera
from sklearn.metrics import mean_absolute_error, mean_squared_error

# ---------------------------------------------------------
# PHẦN 1: TẢI DỮ LIỆU VÀ XÂY DỰNG MÔ HÌNH
# ---------------------------------------------------------
# Đọc dữ liệu
df = pd.read_excel(r"C:\Users\HP\Downloads\Oil.xlsx", sheet_name="Data 1")
Production_data = df[['Order date', 'Production']]
Production_data['Order date'] = pd.to_datetime(Production_data['Order date'])
production_ts = Production_data.set_index('Order date')

# Hiển thị dữ liệu gốc
print("Dữ liệu gốc (5 dòng đầu):")
print(production_ts.head())

# Kiểm tra tính dừng (Stationarity)
def check_stationarity(timeseries):
    result = adfuller(timeseries, autolag='AIC')
    p_value = result[1]
    print(f'ADF Statistic: {result[0]}')
    print(f'p-value: {p_value}')
    print('Stationary' if p_value < 0.05 else 'Non-Stationary')

print("\n--- Kiểm tra tính dừng ---")
check_stationarity(production_ts['Production'])

# Thiết lập tham số (Theo yêu cầu của bạn)
p, d, q = 2, 1, 0
P, D, Q, s = 1, 1, 1, 12

print(f"\n--- Đang chạy mô hình SARIMAX({p},{d},{q})x({P},{D},{Q},{s}) ---")
model = SARIMAX(production_ts, order=(p, d, q), seasonal_order=(P, D, Q, s))
results = model.fit()

# Dự báo 12 kỳ tiếp theo
forecast_periods = 12
forecast = results.forecast(steps=forecast_periods)

# ---------------------------------------------------------
# PHẦN 2: CHẨN ĐOÁN PHẦN DƯ (RESIDUAL DIAGNOSTICS)
# Kiểm tra xem mô hình "học" quá khứ có tốt không?
# ---------------------------------------------------------
print("\n" + "="*50)
print("PHẦN 2: CHẨN ĐOÁN PHẦN DƯ (TRAINING DATA)")
print("="*50)

residuals = results.resid

# 1. Vẽ biểu đồ chẩn đoán
print("Đang vẽ biểu đồ chẩn đoán (Diagnostic Plots)...")
results.plot_diagnostics(figsize=(12, 8))
plt.tight_layout()
plt.show()

# 2. Kiểm định Ljung-Box (Kiểm tra tự tương quan)
# H0: Phần dư là nhiễu trắng (Tốt)
lags_to_test = [12, 24] 
lb_test = acorr_ljungbox(residuals, lags=lags_to_test, return_df=True)

print("\n>>> Kết quả Ljung-Box Test:")
print(lb_test)

p_value_lb = lb_test.loc[12, 'lb_pvalue']
print(f"\nKết luận tại Lag 12 (p-value = {p_value_lb:.4f}):")
if p_value_lb < 0.05:
    print("👉 BÁC BỎ H0: Phần dư KHÔNG PHẢI nhiễu trắng (Mô hình chưa tối ưu).")
else:
    print("👉 CHẤP NHẬN H0: Phần dư LÀ nhiễu trắng (Mô hình tốt).")

# 3. Kiểm định Jarque-Bera (Kiểm tra phân phối chuẩn)
score, p_value_jb = jarque_bera(residuals)
print("\n>>> Kết quả Jarque-Bera Test:")
print(f"P-value: {p_value_jb:.4f}")
if p_value_jb < 0.05:
    print("👉 Phần dư KHÔNG tuân theo phân phối chuẩn.")
else:
    print("👉 Phần dư tuân theo phân phối chuẩn.")

# ---------------------------------------------------------
# PHẦN 3: ĐÁNH GIÁ VỚI DỮ LIỆU THỰC TẾ (FORECAST EVALUATION)
# So sánh kết quả dự báo với 12 tháng thực tế bạn cung cấp
# ---------------------------------------------------------
print("\n" + "="*50)
print("PHẦN 3: ĐÁNH GIÁ ĐỘ CHÍNH XÁC (FORECAST VS ACTUAL)")
print("="*50)

# 1. Nhập dữ liệu thực tế (Hardcoded data)
df1 = pd.read_excel(r"C:\Users\HP\Downloads\Oil.xlsx", sheet_name="Data2")
actual_data = df1[['Date','Actual_Production']]

df_actual = pd.DataFrame(actual_data)
df_actual['Date'] = pd.to_datetime(df_actual['Date'])
df_actual.set_index('Date', inplace=True)

# Lấy giá trị dự báo để so sánh
predicted_values = forecast.values

# 2. Tính toán sai số (Forecast Errors)
forecast_errors = df_actual['Actual_Production'] - predicted_values
mae = mean_absolute_error(df_actual['Actual_Production'], predicted_values)
rmse = np.sqrt(mean_squared_error(df_actual['Actual_Production'], predicted_values))
mape = np.mean(np.abs(forecast_errors / df_actual['Actual_Production'])) * 100

print(f"1. MAE  (Mean Absolute Error): {mae:,.2f}")
print(f"2. RMSE (Root Mean Squared Error): {rmse:,.2f}")
print(f"3. MAPE (Mean Absolute Percentage Error): {mape:.2f}%")

if mape < 10:
    print("   -> Đánh giá: Dự báo TỐT (Sai số < 10%)")
else:
    print("   -> Đánh giá: Cần cải thiện mô hình.")

# 3. Kiểm định độ chệch (Bias Test)
# H0: Trung bình sai số dự báo = 0 (Không bị chệch)
t_stat, p_value_bias = stats.ttest_1samp(forecast_errors, popmean=0)
print(f"\n4. Kiểm định độ chệch (Bias T-test) p-value: {p_value_bias:.4f}")
if p_value_bias > 0.05:
    print("   👉 Mô hình KHÔNG BỊ CHỆCH (Unbiased).")
else:
    print("   👉 Mô hình BỊ CHỆCH (Biased - Dự báo quá cao hoặc quá thấp).")

# 4. Kiểm định tính ngẫu nhiên của sai số dự báo (Ljung-Box on Forecast Errors)
# Chỉ test lag nhỏ (ví dụ 6) vì chỉ có 12 mẫu
lb_test_forecast = acorr_ljungbox(forecast_errors, lags=[6], return_df=True)
p_val_forecast = lb_test_forecast.loc[6, 'lb_pvalue']
print(f"\n5. Kiểm định Ljung-Box trên Sai số Dự báo (Lag 6) p-value: {p_val_forecast:.4f}")
if p_val_forecast > 0.05:
    print("   👉 Sai số dự báo là NGẪU NHIÊN (Tốt).")
else:
    print("   👉 Sai số dự báo CÒN QUY LUẬT (Chưa tốt).")

# 5. Vẽ biểu đồ so sánh cuối cùng
plt.figure(figsize=(12, 6))
# Vẽ dữ liệu lịch sử (lấy 24 tháng cuối để nối vào cho đẹp)
plt.plot(production_ts.index[-24:], production_ts['Production'][-24:], label='Lịch sử (Historical)', color='gray', alpha=0.5)
# Vẽ thực tế tương lai
plt.plot(df_actual.index, df_actual['Actual_Production'], label='Thực tế (Actual)', marker='o', color='green', linewidth=2)
# Vẽ dự báo
plt.plot(df_actual.index, predicted_values, label='Dự báo (Forecast)', marker='x', color='red', linestyle='--', linewidth=2)

plt.title(f"So sánh Thực tế vs Dự báo (MAPE: {mape:.2f}%)")
plt.xlabel("Thời gian")
plt.ylabel("Sản lượng")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# In bảng so sánh
comparison_df = df_actual.copy()
comparison_df['Forecast'] = predicted_values
comparison_df['Diff'] = comparison_df['Actual_Production'] - comparison_df['Forecast']
comparison_df['Diff (%)'] = (comparison_df['Diff'] / comparison_df['Actual_Production']) * 100
print("\n--- BẢNG SO SÁNH CHI TIẾT ---")
print(comparison_df)