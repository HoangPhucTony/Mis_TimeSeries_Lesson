from utils import *
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from xgboost import plot_importance
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Đường dẫn file log
BASE_DIR = r'C:\DOHOANGPHUC\UTH\PhanTichChuoiThoiGianVaDuBao\BaiTap\Practice\HW_TimeSeries\Deep_Learning_model(C4_C5)\logs'
log_path("XGBoost", BASE_DIR)
# Đường dẫn lưu hình ảnh biểu đồ
plot_dir = r'C:\DOHOANGPHUC\UTH\PhanTichChuoiThoiGianVaDuBao\BaiTap\Practice\HW_TimeSeries\Deep_Learning_model(C4_C5)\plots\xgb'
# Load dataset
data_path = r'C:\DOHOANGPHUC\UTH\PhanTichChuoiThoiGianVaDuBao\BaiTap\Practice\Data\data_lab4\weather and load dataset.csv'
df = load_dataset(data_path)

logging.info("PRACTICE 4 – MULTIVARIATE TIME SERIES FORECASTING")
logging.info("MODEL: XGBOOST (Machine Learning)")
logging.info("DATASET: Electricity Load + Weather")
logging.info("Mục tiêu: Dự báo Load(t+1) từ lịch sử 24 giờ + biến thời tiết\n\n")

logging.info("1. DATA UNDERSTANDING")
logging.info("Dữ liệu gồm các biến:")
logging.info("- load_MW: Điện năng tiêu thụ (MW)")
logging.info("- temperature: Nhiệt độ")
logging.info("- humidity: Độ ẩm")
logging.info("- wind_speed: Tốc độ gió\n")

logging.info("5 dòng dữ liệu đầu tiên:")
logging.info(df.head(5).to_string() + "\n")

# Chia dữ liệu theo thời gian: 70% train, 15% val, 15% test
train_df, val_df, test_df = time_split(df)
logging.info("2. PREPROCESSING")
logging.info("Chia dữ liệu theo thứ tự thời gian (KHÔNG SHUFFLE)")
logging.info("Tỷ lệ: Train 70% – Validation 15% – Test 15%")

logging.info("Kích thước tập dữ liệu:")
logging.info(f"Train      : {train_df.shape}")
logging.info(f"Validation : {val_df.shape}")
logging.info(f"Test       : {test_df.shape}\n")

logging.info("Khoảng thời gian:")
logging.info(f"Train      : {train_df.index.min()} → {train_df.index.max()}")
logging.info(f"Validation : {val_df.index.min()} → {val_df.index.max()}")
logging.info(f"Test       : {test_df.index.min()} → {test_df.index.max()}\n")

logging.info("Chuẩn hóa dữ liệu:")
logging.info("- Sử dụng StandardScaler")
logging.info("- Fit trên TRAIN, transform VALIDATION và TEST")
logging.info("- Áp dụng thống nhất cho VAR, XGBoost và LSTM")
logging.info("→ Tránh data leakage và đảm bảo so sánh công bằng\n")

# Chuẩn hóa toàn bộ biến bằng StandardScaler
train_scaled, val_scaled, test_scaled, scaler = scale_features(train_df, val_df, test_df)
logging.info("Toàn bộ biến đã được chuẩn hóa bằng StandardScaler(ở cả 3 mô hình xgboost, lstm, var)\n")


# 5. Feature Engineering
FEATURES = ['load_MW', 'temperature', 'humidity', 'wind_speed']
TARGET = 'load_MW'
L = 24
logging.info("5. FEATURE ENGINEERING – XGBOOST")
logging.info(f"Lag window L = {L} (24 giờ)")
logging.info("Lag features cho tất cả biến:")
logging.info("- load_MW")
logging.info("- temperature")
logging.info("- humidity")
logging.info("- wind_speed")
logging.info(f"Target: {TARGET}\n")

logging.info("Rolling features (chỉ dùng cho XGBoost):")
logging.info("- Rolling mean 6 giờ")
logging.info("- Rolling standard deviation 24 giờ")
logging.info("- Load trend: load(t−1) − load(t−24)")
logging.info("→ Tất cả rolling feature tính trên load.shift(1) để không chứa giá trị cần dự báo\n")

logging.info("Feature Engineering cho XGBoost")
logging.info(f"Lag window: {L}")

def create_features(df):
    df_feat = pd.DataFrame(index=df.index)
    # Lag features
    for i in range(1, L + 1):
        df_feat[f'load_lag_{i}'] = df['load_MW'].shift(i)
        df_feat[f'temp_lag_{i}'] = df['temperature'].shift(i)
        df_feat[f'hum_lag_{i}']  = df['humidity'].shift(i)
        df_feat[f'wind_lag_{i}'] = df['wind_speed'].shift(i)

    # Rolling features – shift(1) để chỉ dùng quá khứ (t-1 trở về trước), tránh rò rỉ load(t)
    # và khớp với create_features_last_step khi walk-forward
    df_feat['load_rm_6']   = df['load_MW'].shift(1).rolling(6).mean()
    df_feat['load_rs_24']  = df['load_MW'].shift(1).rolling(24).std()
    df_feat['load_trend'] = df['load_MW'].shift(1) - df['load_MW'].shift(24)

    # Target
    df_feat[TARGET] = df[TARGET]
    return df_feat.dropna()


def create_features_last_step(history):
    """
    Tạo feature CHỈ cho bước tiếp theo (t+1)
    Dùng cho walk-forward – RẤT NHANH
    """
    row = {}

    # lag features
    for i in range(1, L + 1):
        row[f'load_lag_{i}'] = history['load_MW'].iloc[-i]
        row[f'temp_lag_{i}'] = history['temperature'].iloc[-i]
        row[f'hum_lag_{i}']  = history['humidity'].iloc[-i]
        row[f'wind_lag_{i}'] = history['wind_speed'].iloc[-i]

    # rolling features
    row['load_rm_6']   = history['load_MW'].iloc[-6:].mean()
    row['load_rs_24']  = history['load_MW'].iloc[-24:].std()
    row['load_trend'] = history['load_MW'].iloc[-1] - history['load_MW'].iloc[-24]

    return pd.DataFrame([row])


# 6. Chuẩn bị TRAIN + VAL (dùng để train ban đầu)
train_feat = create_features(train_scaled)
val_feat   = create_features(val_scaled)

X_train = train_feat.drop(columns=[TARGET])
y_train = train_feat[TARGET]

X_val = val_feat.drop(columns=[TARGET])
y_val = val_feat[TARGET]

logging.info("Chuyển bài toán sang supervised learning:")
logging.info("Input  : Vector lag + rolling features")
logging.info("Output : load_MW(t+1)\n")

logging.info("Kích thước dữ liệu sau feature engineering:")
logging.info(f"X_train: {X_train.shape}, y_train: {y_train.shape}")
logging.info(f"X_val  : {X_val.shape}, y_val  : {y_val.shape}\n")

# 7. Huấn luyện XGBoost
logging.info("6. HUẤN LUYỆN MÔ HÌNH XGBOOST")

logging.info("Cấu hình mô hình:")
logging.info("- n_estimators: 500")
logging.info("- max_depth: 6")
logging.info("- learning_rate: 0.05")
logging.info("- subsample: 0.8")
logging.info("- colsample_bytree: 0.8")

logging.info("Sử dụng early stopping trên validation set (RMSE)")
logging.info("→ Tránh overfitting\n")

xgb_model = XGBRegressor(
    n_estimators=500,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="reg:squarederror",
    eval_metric="rmse",
    early_stopping_rounds=50,  # XGBoost >= 2.0: khai báo trong constructor thay vì fit()
    random_state=42
)

xgb_model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    verbose=False
)

logging.info("Huấn luyện XGBoost hoàn tất (early stopping hợp lệ)\n")

logging.info("7. BACKTESTING – WALK-FORWARD VALIDATION")
logging.info("Quy trình:")
logging.info("Train → predict t+1 → thêm GIÁ TRỊ THẬT → predict t+2 → ...")
logging.info("→ Mô phỏng đúng kịch bản dự báo thực tế\n")

# train model 1 lần với best_iteration
best_n_estimators = xgb_model.best_iteration
if best_n_estimators is None:
    best_n_estimators = 500

model = XGBRegressor(
    n_estimators=best_n_estimators,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="reg:squarederror",
    random_state=42
)
logging.info(f"Số cây tối ưu (best_iteration): {best_n_estimators}\n")

# dùng toàn bộ train + val
trainval_feat = create_features(pd.concat([train_scaled, val_scaled]))
X_trainval = trainval_feat.drop(columns=[TARGET])
y_trainval = trainval_feat[TARGET]

model.fit(X_trainval, y_trainval, verbose=False)

history = pd.concat([train_scaled, val_scaled])
predictions = []
actuals = []

for t in range(len(test_scaled)):
    # CHỈ tạo feature cho 1 bước tiếp theo
    X_next = create_features_last_step(history)

    y_hat = model.predict(X_next)[0]
    y_true = test_scaled.iloc[t][TARGET]

    predictions.append(y_hat)
    actuals.append(y_true)

    # update history bằng GIÁ TRỊ THẬT (đúng walk-forward)
    history = pd.concat([history, test_scaled.iloc[t:t+1]])


# 9. Inverse Scaling (CHỈ LOAD – ĐÚNG CỘT)
load_index = train_df.columns.get_loc("load_MW")

def inverse_transform_load(values, scaler, load_index):
    dummy = np.zeros((len(values), scaler.mean_.shape[0]))
    dummy[:, load_index] = values
    return scaler.inverse_transform(dummy)[:, load_index]

pred_inv = inverse_transform_load(predictions, scaler, load_index)
true_inv = inverse_transform_load(actuals, scaler, load_index)


# 10. Đánh giá mô hình (ĐƠN VỊ MW – CHUẨN)
rmse = np.sqrt(mean_squared_error(true_inv, pred_inv))
mae  = mean_absolute_error(true_inv, pred_inv)
mask = true_inv != 0
mape = np.mean(np.abs((true_inv[mask] - pred_inv[mask]) / true_inv[mask])) * 100


logging.info("10. ĐÁNH GIÁ MÔ HÌNH XGBOOST (ĐƠN VỊ MW)")
logging.info(f"RMSE: {rmse:.2f}")
logging.info(f"MAE : {mae:.2f}")
logging.info(f"MAPE: {mape:.2f}%\n")

logging.info("So sánh một số giá trị đầu:")
logging.info("Actual (MW): " + str(true_inv[:5]))
logging.info("Predicted (MW): " + str(pred_inv[:5]) + "\n")

# 11. Feature Importance – XGBoost
logging.info("11. PHÂN TÍCH FEATURE IMPORTANCE – XGBOOST")
logging.info("Importance type: GAIN")
logging.info("→ Thể hiện mức độ đóng góp của từng feature vào mô hình")
logging.info("→ Trả lời câu hỏi: biến thời tiết nào ảnh hưởng mạnh nhất\n")

plt.figure(figsize=(10, 6))
plot_importance(model, max_num_features=15,importance_type="gain")
plt.title("Top 15 Feature Importance (XGBoost – Gain)")
plt.tight_layout()
save_plot("1_XGBoost_Feature_Importance_Top15.png", plot_dir)
plt.close()
logging.info("Đã lưu biểu đồ:")
logging.info("- Feature Importance Top 15\n")
logging.info("NHẬN XÉT FEATURE IMPORTANCE – XGBOOST (TÓM TẮT)\n")

logging.info(
    "Feature importance cho thấy XGBoost chủ yếu dựa vào load trong quá khứ gần nhất "
    "(load_lag_1, load_lag_2), phản ánh quán tính ngắn hạn của tiêu thụ điện.\n"
)

logging.info(
    "Nhóm tiếp theo là load_lag_24, load_lag_23 và load_trend, cho thấy mô hình học được "
    "chu kỳ ngày (daily seasonality): load hiện tại giống load cùng giờ ngày hôm trước.\n"
)

logging.info(
    "Các biến thời tiết chỉ đóng góp phụ trợ và phân tán trên nhiều độ trễ "
    "(wind_lag_6, temp_lag_20–22, hum_lag_4...), không có biến thời tiết nào nổi trội "
    "khi đã có lịch sử load.\n"
)

logging.info(
    "Kết luận (Câu 11): XGBoost học được rằng load phụ thuộc mạnh vào hành vi tiêu "
    "thụ 1–2 giờ trước và cùng giờ ngày hôm trước; thời tiết chỉ bổ sung thông tin nhỏ "
    "cho dự báo 1 giờ tới.\n"
)


# 12. Plot – Actual vs Predicted Load_MW
logging.info("VẼ BIỂU ĐỒ SO SÁNH LOAD THẬT VS DỰ BÁO")
plt.figure(figsize=(14, 6))
plt.plot(test_df.index,true_inv,label="Actual Load", linewidth=2)
plt.plot(test_df.index,pred_inv,label="Predicted Load (XGBoost)",linestyle="--")
plt.title("Actual vs Predicted Electricity Load – XGBoost Walk-forward")
plt.xlabel("Time")
plt.ylabel("Load (MW)")
plt.legend()
plt.grid(True)
plt.tight_layout()
save_plot("2_XGBoost_Actual_vs_Predicted_Load.png", plot_dir)
plt.close()
logging.info("Đã lưu biểu đồ:")
logging.info("- Actual vs Predicted Load (Walk-forward)")
logging.info("NHẬN XÉT BIỂU ĐỒ ACTUAL VS PREDICTED LOAD – XGBOOST (WALK-FORWARD)\n")

logging.info(
    "1. Độ bám tổng thể giữa giá trị dự báo và giá trị thực tế là rất tốt.\n"
    "→ Đường dự báo (Predicted Load) gần như chồng lên đường thực tế (Actual Load) "
    "trong hầu hết các khoảng thời gian.\n"
    "→ Điều này cho thấy XGBoost học tốt cấu trúc tổng thể của chuỗi và không bị lệch pha theo thời gian.\n"
)

logging.info(
    "2. XGBoost tái hiện rõ chu kỳ ngày (daily seasonality).\n"
    "→ Các dao động lên xuống theo chu kỳ 24 giờ được mô hình dự báo khá chính xác.\n"
    "→ Đỉnh và đáy của load trong ngày được bắt đúng thời điểm, phản ánh việc mô hình "
    "khai thác hiệu quả các lag 24 giờ và rolling features.\n"
)

logging.info(
    "3. Sai lệch chủ yếu xuất hiện tại các điểm biến động đột ngột (spikes hoặc drops).\n"
    "→ Ở một số thời điểm load giảm hoặc tăng mạnh bất thường, mô hình có xu hướng "
    "dự báo thấp hơn hoặc cao hơn thực tế.\n"
    "→ Đây là đặc điểm phổ biến của các mô hình dựa trên lịch sử, khi xảy ra sự kiện "
    "ngoại lai không lặp lại trong dữ liệu huấn luyện.\n"
)

logging.info(
    "4. Mô hình phản ứng tốt với sự thay đổi xu hướng trung hạn.\n"
    "→ Khi load có xu hướng giảm hoặc tăng kéo dài theo thời gian, đường dự báo "
    "đi theo xu hướng chung thay vì chỉ bám sát các dao động ngắn hạn.\n"
    "→ Điều này cho thấy vai trò quan trọng của rolling mean và load_trend trong mô hình.\n"
)

logging.info(
    "5. Không xuất hiện hiện tượng trễ dự báo (prediction lag).\n"
    "→ Đường dự báo không bị lệch sang trái hoặc phải so với đường thực tế.\n"
    "→ Điều này chứng tỏ quy trình walk-forward validation được triển khai đúng, "
    "không xảy ra data leakage và mô hình luôn dự báo dựa trên thông tin quá khứ.\n"
)

logging.info(
    "Kết luận: Biểu đồ Actual vs Predicted cho thấy XGBoost dự báo load khá chính xác, "
    "bắt tốt chu kỳ ngày và xu hướng ngắn hạn, trong khi sai số chủ yếu xuất hiện tại "
    "các thời điểm biến động đột ngột. Kết quả này phù hợp với các chỉ số RMSE, MAE và MAPE "
    "đã đánh giá trước đó.\n"
)

# Giá trị metric (đã tính)
metrics = ['RMSE', 'MAE', 'MAPE']
values  = [rmse, mae, mape]

plt.figure(figsize=(6, 4))
plt.bar(metrics, values)

plt.title('Model Evaluation Metrics (XGBoost - Test Set)')
plt.ylabel('Value')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
# Hiển thị giá trị trên cột
for i, v in enumerate(values):
    plt.text(i, v, f'{v:.2f}', ha='center', va='bottom')
save_plot("3_XGBoost_Evaluation_Metrics.png", plot_dir)
plt.close()
