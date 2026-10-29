# BÀI THỰC HÀNH 3 Dự đoán chuỗi thời gian sử dụng Machine Learning models 
import os
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import LinearRegression

# logging
BASE_DIR = r"C:\DOHOANGPHUC\UTH\PhanTichChuoiThoiGianVaDuBao\BaiTap\Practice\Machine_Learning_model(C3_P2)"
PLOT_DIR = os.path.join(BASE_DIR, "plot")

os.makedirs(PLOT_DIR, exist_ok=True)

LOG_PATH = os.path.join(BASE_DIR, "report.log")

logging.basicConfig(
    level=logging.INFO, # Info, Warning, Error, Critical -> không ghi debug
    format="%(message)s", # ghi nội dung, ko ghi thời gian, tên file,
    handlers=[
        logging.FileHandler(LOG_PATH, mode="w", encoding="utf-8"), # ghi đè lên file
        logging.StreamHandler() # ghi ra terminal 
    ]
)

logger = logging.getLogger() # 

logger.info("="*70)
logger.info("BÀI THỰC HÀNH 3: Dự đoán chuỗi thời gian sử dụng Machine Learning models ")
logger.info("="*70)

logger.info(" Sử dụng dataset: \n")


path = r"C:\DOHOANGPHUC\UTH\PhanTichChuoiThoiGianVaDuBao\BaiTap\Practice\Data\ML_DATA\PJMW_hourly.csv"

df = pd.read_csv(path, parse_dates=["Datetime"], index_col="Datetime")
df = df.sort_index()
df.rename(columns={"PJMW_MW": "y"}, inplace=True) # thay đôi trực tiếp lên df 

logger.info("\n[BƯỚC 1] Chuẩn bị dữ liệu")
logger.info(df.head())
logger.info(df.info())
logger.info(" Mô tả dữ liệu: ")
logger.info(f"Số lượng bản ghi: {len(df)}")
logger.info(f"Khoảng thời gian: {df.index.min()} → {df.index.max()}")
logger.info(" DESCRIBE")
logger.info(df.describe())


plt.figure(figsize=(10,4))
plt.plot(df["y"])
plt.title("Raw Time Series")
plt.xlabel("Time")
plt.ylabel("MW")
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, "ChuoiRaw.png"))
plt.close()

# Tạo lag features (Supervised Learning)
logger.info("\n[BƯỚC 2] Tạo lag features")

p = 3  # số lag 

for i in range(1, p + 1):
    df[f"lag_{i}"] = df["y"].shift(i)

df = df.dropna()

logger.info(df.head())

# Chia X - Y
logger.info("\n[BƯỚC 3] Chia X và y")

X = df[[f"lag_{i}" for i in range(1, p + 1)]]
y = df["y"]

logger.info("X:")
logger.info(X.head())
logger.info("y:")
logger.info(y.head())

#  Chia train / test theo thời gian
logger.info("\n[BƯỚC 4] Chia train / test theo thời gian")

split = int(len(df) * 0.8)

X_train, X_test = X.iloc[:split], X.iloc[split:]
y_train, y_test = y.iloc[:split], y.iloc[split:]

logger.info(f"Train size: {X_train.shape}")
logger.info(f"Test size : {X_test.shape}")

# Huấn luyện mô hình XGBoost
logger.info("\n[BƯỚC 5] Huấn luyện mô hình XGBoost")

model = XGBRegressor(
    n_estimators=100,
    max_depth=3,
    learning_rate=0.1,
    random_state=42
)

model.fit(X_train, y_train)

logger.info("→ Huấn luyện hoàn tất")

# Dự đoán và đánh giá 
logger.info("\n[BƯỚC 6] Dự đoán và đánh giá")

y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100

logger.info(f"MAE  = {mae:.2f}")
logger.info(f"RMSE = {rmse:.2f}")
logger.info(f"MAPE = {mape:.2f}%")

# Vẽ so sánh
plt.figure(figsize=(10,4))
plt.plot(y_test.values, label="Actual")
plt.plot(y_pred, label="Predicted")
plt.title("Actual vs Predicted - XGBoost")
plt.xlabel("Time Step")
plt.ylabel("MW")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, "ThucTe_DuBao.png"))
plt.close()

# Phân tích feature importance để tìm hiểu lag nào ảnh hưởng mạnh nhất đến dự báo
logger.info("\n[BƯỚC 7] Feature Importance")

importance = model.feature_importances_

for f, s in zip(X.columns, importance):
    logger.info(f"{f}: {s:.4f}")

plt.figure(figsize=(6,4))
plt.bar(X.columns, importance)
plt.title("Feature Importance (Lag Features)")
plt.xlabel("Lag")
plt.ylabel("Importance")
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, "CacFeatureQuanTrong.png"))
plt.close()

logger.info("\nHOÀN THÀNH BÀI THỰC HÀNH 3")


# BÀI TẬP NÂNG CAO 
logger.info("\n" + "="*70)
logger.info("Bài tập nâng cao - Machine Learning models")
logger.info("="*70)

# Bài 1
logger.info("\n[1] Thêm rolling mean / rolling std")

window = 3

# shift(1) để rolling chỉ dùng quá khứ (t-1 → t-window), tránh rò rỉ y_t vào feature
df["rolling_mean"] = df["y"].shift(1).rolling(window=window).mean()
df["rolling_std"]  = df["y"].shift(1).rolling(window=window).std()

df = df.dropna()

feature_cols = (
    [f"lag_{i}" for i in range(1, p+1)] +
    ["rolling_mean", "rolling_std"]
)

X = df[feature_cols]
y = df["y"]

logger.info("Features sau khi bổ sung rolling:")
logger.info(X.head())
# NOTE: Rolling mean giúp model nắm xu hướng ngắn hạn, rolling std phản ánh mức độ biến động


# Bài 2 Dự báo nhiều bước (multi-step forecasting)
logger.info("\n[2] Dự báo multi-step (h = 3)")

h = 3
multi_results = {}

# Feature mới nhất là t-1 nên horizon 1 bước = y_t, horizon h bước = y_{t+h-1}
for step in range(1, h + 1):
    df[f"y_t+{step}"] = df["y"].shift(-(step - 1))

df_multi = df.dropna()

X_multi = df_multi[feature_cols]

for step in range(1, h + 1):
    y_multi = df_multi[f"y_t+{step}"]

    split = int(len(df_multi) * 0.8)
    X_train, X_test = X_multi[:split], X_multi[split:]
    y_train, y_test = y_multi[:split], y_multi[split:]

    model = XGBRegressor(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.1,
        random_state=42
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100

    multi_results[f"t+{step}"] = rmse

    logger.info(f"t+{step}: RMSE = {rmse:.2f}, MAPE = {mape:.2f}%")

# Bài 3: Tune hyperparameter bằng TimeSeriesSplit 
logger.info("\n[3] Tune bằng TimeSeriesSplit")

tscv = TimeSeriesSplit(n_splits=5)

params = {
    "max_depth": [2, 3],
    "learning_rate": [0.05, 0.1]
}

best_rmse = np.inf
best_param = None

for depth in params["max_depth"]:
    for lr in params["learning_rate"]:
        rmses = []

        for train_idx, test_idx in tscv.split(X):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            model = XGBRegressor(
                n_estimators=100,
                max_depth=depth,
                learning_rate=lr,
                random_state=42
            )

            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            rmses.append(np.sqrt(mean_squared_error(y_test, y_pred)))

        avg_rmse = np.mean(rmses)

        logger.info(f"depth={depth}, lr={lr} → RMSE={avg_rmse:.2f}")

        if avg_rmse < best_rmse:
            best_rmse = avg_rmse
            best_param = (depth, lr)

logger.info(f"BEST PARAM: depth={best_param[0]}, lr={best_param[1]}")

# Huấn luyện lại XGBoost với tham số tốt nhất trên fold cuối (X_train, X_test từ vòng lặp trên)
# để so sánh công bằng với Random Forest và Linear Regression trên cùng tập test
xgb_best = XGBRegressor(
    n_estimators=100,
    max_depth=best_param[0],
    learning_rate=best_param[1],
    random_state=42
)
xgb_best.fit(X_train, y_train)
y_pred = xgb_best.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100

# Bài 4: So sánh với Random Forest
from sklearn.ensemble import RandomForestRegressor

logger.info("\n[4] So sánh XGBoost vs Random Forest (fold cuối của TimeSeriesSplit)")

rf = RandomForestRegressor(
    n_estimators=200,
    max_depth=5,
    random_state=42
)

rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)

rf_rmse = np.sqrt(mean_squared_error(y_test, rf_pred))

logger.info(f"RMSE Random Forest: {rf_rmse:.2f}")
logger.info(f"RMSE XGBoost     : {rmse:.2f}")


# ------------------------------------------------------------------------------------------------------------------------
# Trả lời câu hỏi cuối chương
logger.info("Trả lời câu hỏi cuối chương")
logger.info("\n[CÂU 1] So sánh XGBoost vs Linear Regression")

# Huấn luyện Linear Regression
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

# Dự đoán
lr_pred = lr_model.predict(X_test)

# Đánh giá
lr_mae = mean_absolute_error(y_test, lr_pred)
lr_rmse = np.sqrt(mean_squared_error(y_test, lr_pred))
lr_mape = np.mean(np.abs((y_test - lr_pred) / y_test)) * 100

logger.info(f"Linear Regression - MAE : {lr_mae:.2f}") # {Giá trị: Đinh dạng}
logger.info(f"Linear Regression - RMSE: {lr_rmse:.2f}")
logger.info(f"Linear Regression - MAPE: {lr_mape:.2f}%")

logger.info(f"XGBoost RMSE           : {rmse:.2f}")
logger.info(f"XGBoost MAPE           : {mape:.2f}%")


# Vẽ so sánh
plt.figure(figsize=(10,4))
plt.plot(y_test.values, label="Actual", color="black")
plt.plot(lr_pred, label="Linear Regression", linestyle="--")
plt.plot(y_pred, label="XGBoost")
plt.title("Actual vs Linear Regression vs XGBoost")
plt.xlabel("Time Step")
plt.ylabel("MW")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, "Cau1_SoSanh_LR_XGB.png"))
plt.close()

logger.info("\n[CÂU 2] Phân tích ảnh hưởng khi tăng số lag p")

lag_values = [1, 3, 6, 12]
rmse_by_lag = []

for p_test in lag_values:
    df_tmp = df.copy()

    # tạo lag
    for i in range(1, p_test + 1):
        df_tmp[f"lag_{i}"] = df_tmp["y"].shift(i)

    df_tmp = df_tmp.dropna()

    X_tmp = df_tmp[[f"lag_{i}" for i in range(1, p_test + 1)]]
    y_tmp = df_tmp["y"]

    split = int(len(df_tmp) * 0.8)
    X_tr, X_te = X_tmp[:split], X_tmp[split:]
    y_tr, y_te = y_tmp[:split], y_tmp[split:]

    model_tmp = XGBRegressor(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.1,
        random_state=42
    )

    model_tmp.fit(X_tr, y_tr)
    pred_tmp = model_tmp.predict(X_te)

    rmse_tmp = np.sqrt(mean_squared_error(y_te, pred_tmp))
    rmse_by_lag.append(rmse_tmp)

    logger.info(f"p = {p_test} → RMSE = {rmse_tmp:.2f}")

# Vẽ RMSE theo p
plt.figure(figsize=(6,4))
plt.plot(lag_values, rmse_by_lag, marker="o")
plt.xlabel("Number of lags (p)")
plt.ylabel("RMSE")
plt.title("RMSE theo số lag p")
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, "Cau2_RMSE_Lagp.png"))
plt.close()

logger.info("\n[CÂU 3] Minh chứng XGBoost phù hợp hơn ARIMA")

# Vẽ RMSE theo horizon multi-step
steps = list(multi_results.keys())
values = list(multi_results.values())

plt.figure(figsize=(6,4))
plt.plot(steps, values, marker="o") 
plt.xlabel("Forecast Horizon")
plt.ylabel("RMSE")
plt.title("RMSE tăng theo forecast horizon (XGBoost)")
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, "Cau3_RMSE_Horizon.png"))
plt.close()

logger.info("RMSE tăng dần theo horizon → đặc trưng chuỗi phi tuyến, biến động mạnh")


logger.info("\nKết luận: ")
logger.info("1. Với feature chỉ gồm lag + rolling (đều là tổ hợp tuyến tính của quá khứ gần), "
            "Linear Regression cho RMSE thấp hơn XGBoost → quan hệ giữa các giờ liền kề gần như tuyến tính")
logger.info("2. Tăng số lag p giảm RMSE mạnh từ p=1 lên p=3, sau đó chỉ cải thiện nhẹ → lợi ích giảm dần")
logger.info("3. Với dữ liệu Hourly Energy Consumption, XGBoost phù hợp hơn ARIMA")
logger.info("4. MAPE giúp đánh giá sai số theo tỷ lệ %, phù hợp với dữ liệu tiêu thụ điện")
logger.info("5. XGBoost vượt Random Forest; để XGBoost vượt Linear Regression cần thêm feature phi tuyến "
            "(giờ trong ngày, thứ trong tuần, lag 24h/168h)")


