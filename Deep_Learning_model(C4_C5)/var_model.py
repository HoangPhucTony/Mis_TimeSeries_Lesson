from utils import *
import pandas as pd
import numpy as np
import logging
import matplotlib.pyplot as plt

from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import mean_absolute_error, mean_squared_error

BASE_DIR = r'C:\DOHOANGPHUC\UTH\PhanTichChuoiThoiGianVaDuBao\BaiTap\Practice\HW_TimeSeries\Deep_Learning_model(C4_C5)\logs'
log_path("VAR", BASE_DIR)

plot_dir = r'C:\DOHOANGPHUC\UTH\PhanTichChuoiThoiGianVaDuBao\BaiTap\Practice\HW_TimeSeries\Deep_Learning_model(C4_C5)\plots\var'

data_path = r'C:\DOHOANGPHUC\UTH\PhanTichChuoiThoiGianVaDuBao\BaiTap\Practice\Data\data_lab4\weather and load dataset.csv'
df = load_dataset(data_path)

VAR_COLS = ["load_MW", "temperature", "humidity", "wind_speed"]
df = df[VAR_COLS].asfreq("H") # Đặt tần số về Hour

logging.info("PRACTICE 4 – MULTIVARIATE TIME SERIES FORECASTING")
logging.info("MODEL: VAR (Statistical Baseline)")
logging.info(f"Số dòng: {df.shape[0]} | Số biến: {df.shape[1]}")
logging.info("Các biến sử dụng cho VAR:")
for c in VAR_COLS:
    logging.info(f" - {c}")

logging.info("\n5 dòng dữ liệu đầu:")
logging.info(df.head(5).to_string())

# 3. Time split (70 / 15 / 15)
train_df, val_df, test_df = time_split(df)

logging.info("\n CHIA DỮ LIỆU THEO THỜI GIAN (70 / 15 / 15)")
logging.info(f"Train set      : {train_df.shape}")
logging.info(f"Validation set : {val_df.shape}")
logging.info(f"Test set       : {test_df.shape}")

# 4. ADF Test (CHỈ DÙNG TRAIN)
logging.info("\n KIỂM TRA TÍNH DỪNG (ADF TEST – TRAIN SET)")

def adf_test(series, name):
    result = adfuller(series, autolag="AIC")
    logging.info(
        f"{name}: ADF={result[0]:.4f}, p-value={result[1]:.4f}"
    )
    return result[1] <= 0.05

stationary = {}
for col in train_df.columns:
    stationary[col] = adf_test(train_df[col], col)

# 5. Differencing (nếu cần) – CHỈ DỰA TRÊN TRAIN
use_diff = not all(stationary.values())

if use_diff:
    logging.info("\n XỬ LÝ CHUỖI KHÔNG DỪNG")
    logging.info("→ Ít nhất một biến KHÔNG dừng")
    logging.info("→ Thực hiện sai phân bậc 1 (d = 1)")

    train_diff = train_df.diff().dropna()

    # Test bắt đầu ngay sau VALIDATION nên mốc inverse differencing là giá trị cuối VAL
    last_known = val_df.iloc[-1]

    # nối val + test để diff LIÊN TỤC
    future = pd.concat([train_df.iloc[-1:], val_df, test_df])
    future_diff = future.diff().dropna()

    val_diff  = future_diff.iloc[:len(val_df)]
    test_diff = future_diff.iloc[len(val_df):]

else:
    logging.info("\n XỬ LÝ CHUỖI DỪNG")
    logging.info("→ Tất cả biến đều dừng")
    logging.info("→ KHÔNG thực hiện sai phân")

    train_diff = train_df.copy()
    val_diff   = val_df.copy()
    test_diff  = test_df.copy()

# 6. Chọn lag VAR bằng AIC (TRAIN)
logging.info("\n CHỌN BẬC TRỄ VAR (DỰA TRÊN AIC)")

model = VAR(train_diff)
lag_order = model.select_order(maxlags=6)

logging.info("Bảng chọn lag:")
logging.info(lag_order.summary().as_text())

#LẤY LAG AIC NHỎ NHẤT
best_lag = lag_order.selected_orders["aic"]

logging.info(f"→ Lag VAR được chọn (AIC nhỏ nhất): p = {best_lag}")

# 7. Walk-forward Validation (CHUẨN ĐỀ)
logging.info("\n WALK-FORWARD VALIDATION")

# History gồm TRAIN + VAL để bước dự báo đầu tiên của TEST dùng đúng các lag liền trước
history = pd.concat([train_diff, val_diff])
predictions_diff = []
actuals_diff = []

load_idx = history.columns.get_loc("load_MW")

for t in range(len(test_diff)):
    model = VAR(history)
    model_fit = model.fit(best_lag)

    forecast = model_fit.forecast(
        history.values[-best_lag:], steps=1
    )

    y_hat = forecast[0, load_idx]
    y_true = test_diff.iloc[t]["load_MW"]

    predictions_diff.append(y_hat)
    actuals_diff.append(y_true)

    # cập nhật history bằng GIÁ TRỊ THẬT
    history = pd.concat([history, test_diff.iloc[t:t+1]])

logging.info(f"Số bước dự báo walk-forward: {len(predictions_diff)}")
predictions_diff = np.array(predictions_diff)
actuals_diff = np.array(actuals_diff)

# 8. Inverse differencing (QUAY VỀ THANG GỐC)
if use_diff:
    logging.info("\n INVERSE DIFFERENCING – QUAY VỀ THANG ĐO GỐC")

    preds = []
    trues = []

    prev = last_known["load_MW"]

    for y_hat_d, y_true_d in zip(predictions_diff, actuals_diff):
        y_hat = prev + y_hat_d
        y_true = prev + y_true_d

        preds.append(y_hat)
        trues.append(y_true)

        prev = y_true

    preds = np.array(preds)
    trues = np.array(trues)

else:
    preds = predictions_diff
    trues = actuals_diff

# 9. Đánh giá mô hình
rmse = np.sqrt(mean_squared_error(trues, preds))
mae  = mean_absolute_error(trues, preds)
mask = trues != 0  # có giờ load = 0 MW, loại ra để MAPE không bị chia cho 0
mape = np.mean(np.abs((trues[mask] - preds[mask]) / trues[mask])) * 100

logging.info("\nĐÁNH GIÁ MÔ HÌNH VAR (TEST SET)")
logging.info(f"RMSE : {rmse:.4f}")
logging.info(f"MAE  : {mae:.4f}")
logging.info(f"MAPE : {mape:.2f}%")

logging.info("\nLƯU BIỂU ĐỒ KẾT QUẢ")
# 10. Plot: Actual vs VAR Prediction (TEST SET)
plt.figure(figsize=(12, 5))
plt.plot(test_df.index[:len(trues)],trues,
    label="Actual Load",
    color="black",
    linewidth=2
)
plt.plot(test_df.index[:len(preds)],preds,
    label="VAR Prediction",
    linestyle="--",
    color="red"
)
plt.title("VAR – Actual vs Prediction (Test Set)")
plt.xlabel("Time")
plt.ylabel("Electricity Load (MW)")
plt.legend()
plt.grid(True)
save_plot("1_VAR_Actual_vs_Predicted_Load.png", plot_dir)
plt.tight_layout()
plt.close()

# Giá trị metric (đã tính)
metrics = ['RMSE', 'MAE', 'MAPE']
values  = [rmse, mae, mape]

plt.figure(figsize=(6, 4))
plt.bar(metrics, values)

plt.title('2_Model Evaluation Metrics (VAR - Test Set)')
plt.ylabel('Value')
plt.grid(axis='y', linestyle='--', alpha=0.7)
# Hiển thị giá trị trên cột
for i, v in enumerate(values):
    plt.text(i, v, f'{v:.2f}', ha='center', va='bottom')
plt.tight_layout()
save_plot("2_VAR_Evaluation_Metrics.png", plot_dir)
plt.close()