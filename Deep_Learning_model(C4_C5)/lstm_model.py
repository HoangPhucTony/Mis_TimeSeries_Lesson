from utils import *
import pandas as pd
import numpy as np
import logging
import matplotlib.pyplot as plt

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping

from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow.keras.utils import set_random_seed

set_random_seed(42)  # cố định seed (Python, NumPy, TensorFlow) để kết quả chạy lại giống nhau

BASE_DIR = r'C:\DOHOANGPHUC\UTH\PhanTichChuoiThoiGianVaDuBao\BaiTap\Practice\HW_TimeSeries\Deep_Learning_model(C4_C5)\logs'
log_path("LSTM", BASE_DIR)

plot_dir = r'C:\DOHOANGPHUC\UTH\PhanTichChuoiThoiGianVaDuBao\BaiTap\Practice\HW_TimeSeries\Deep_Learning_model(C4_C5)\plots\lstm'

data_path = r'C:\DOHOANGPHUC\UTH\PhanTichChuoiThoiGianVaDuBao\BaiTap\Practice\Data\data_lab4\weather and load dataset.csv'
df = load_dataset(data_path)

logging.info("PRACTICE 4 – MULTIVARIATE TIME SERIES FORECASTING")
logging.info("MODEL: LSTM (DEEP LEARNING)")
logging.info("DATASET: Electricity Load + Weather")
logging.info("MỤC TIÊU: Dự báo điện năng tiêu thụ (load_MW) từ dữ liệu đa biến")\


logging.info("Hiển thị 5 dòng dữ liệu đầu tiên:")
logging.info(df.head(5).to_string())
logging.info("\nCác biến trong dataset:")
logging.info(", ".join(df.columns))


train_df, val_df, test_df = time_split(df)

logging.info("2. CHIA DỮ LIỆU THEO THỜI GIAN (KHÔNG SHUFFLE)")
logging.info("Tỷ lệ: Train 70% – Validation 15% – Test 15%")
logging.info(f"Kích thước TRAIN      : {train_df.shape}")
logging.info(f"Kích thước VALIDATION : {val_df.shape}")
logging.info(f"Kích thước TEST       : {test_df.shape}")
logging.info("→ Việc không shuffle giúp tránh rò rỉ thông tin tương lai\n")


# 4. Scaling (StandardScaler – FIT TRÊN TRAIN)
train_scaled, val_scaled, test_scaled, scaler = scale_features(
    train_df, val_df, test_df )
logging.info("3. CHUẨN HÓA DỮ LIỆU (STANDARD SCALER)")
logging.info("• Fit scaler CHỈ trên tập TRAIN")
logging.info("• Transform Validation & Test")
logging.info("• Áp dụng cho toàn bộ biến đầu vào")

logging.info("→ Chuẩn hóa giúp LSTM hội tụ nhanh và ổn định hơn\n")
logging.info("Toàn bộ biến đã được chuẩn hóa bằng StandardScaler\n")

# 5. Feature Engineering cho LSTM
FEATURES = ['load_MW', 'temperature', 'humidity', 'wind_speed']
TARGET = 'load_MW'
L = 24 # lag window

logging.info("4. FEATURE ENGINEERING CHO LSTM")
logging.info(f"• Các biến đầu vào (features): {FEATURES}")
logging.info(f"• Biến mục tiêu (target)     : {TARGET}")
logging.info(f"• Cửa sổ trễ (lag window)    : L = {L} giờ")

logging.info("→ Mỗi mẫu LSTM có dạng: (24 bước thời gian × 4 biến)")

def create_lstm_sequences(df, features, target, window):
    # (samples, timesteps, features)
    X, y = [], []
    values = df[features].values # input 
    target_values = df[target].values # output 
#sliding window 
    for i in range(window, len(df)): # mỗi thời điểm i, gồm L bước
        X.append(values[i - window:i])
        y.append(target_values[i]) # target tại thời điểm i

    return np.array(X), np.array(y)

# 6. Chuẩn bị dữ liệu TRAIN + VAL
X_train, y_train = create_lstm_sequences(train_scaled, FEATURES, TARGET, L)
X_val, y_val     = create_lstm_sequences(val_scaled,   FEATURES, TARGET, L)

logging.info("5. CHUẨN BỊ DỮ LIỆU ĐẦU VÀO CHO LSTM")
logging.info("Dạng dữ liệu: Tensor 3D (samples, timesteps, features)")

logging.info(f"X_train shape: {X_train.shape}")
logging.info(f"y_train shape: {y_train.shape}")
logging.info(f"X_val   shape: {X_val.shape}")
logging.info(f"y_val   shape: {y_val.shape}")

logging.info("→ Dữ liệu phù hợp với kiến trúc LSTM\n")


logging.info("6. KIẾN TRÚC MÔ HÌNH LSTM")
logging.info("Cấu trúc mạng:")
logging.info("LSTM(64, return_sequences=True)")
logging.info("→ LSTM(32)")
logging.info("→ Dense(1)")

logging.info("• Loss function : Mean Squared Error (MSE)")
logging.info("• Optimizer     : Adam")
logging.info("• EarlyStopping : Theo dõi val_loss (patience = 5)\n")

model = Sequential([ #các layer xếp nối tiếp, output layer trước = input layer sau
    LSTM(64, return_sequences=True, input_shape=(L, len(FEATURES))), #(24,4)
    LSTM(32),
    Dense(1)
])

model.compile(optimizer="adam", loss="mse")

model.summary(print_fn=logging.info) # in ra log
early_stopping = EarlyStopping( # chống overfitting
    monitor="val_loss",
    patience=5,
    restore_best_weights=True # sử dụng trọng số tốt nhất sau khi dừng 
)

logging.info("7. HUẤN LUYỆN MÔ HÌNH LSTM")
logging.info("• Số epoch tối đa : 50")
logging.info("• Batch size      : 32")
logging.info("• Có sử dụng EarlyStopping")

logging.info("→ Mô hình tự động dừng khi val_loss không cải thiện\n")
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=50,
    batch_size=32,
    callbacks=[early_stopping],
    verbose=0
)
logging.info("Huấn luyện LSTM hoàn tất (có EarlyStopping)\n")

# 8. WALK-FORWARD VALIDATION (CHUẨN ĐỀ)
logging.info("8. WALK-FORWARD VALIDATION (CHUẨN THỰC TẾ)")
logging.info("Quy trình:")
logging.info("• Dự báo từng bước t+1")
logging.info("• Sau mỗi bước, thêm GIÁ TRỊ THẬT vào history")
logging.info("• Mô phỏng đúng kịch bản dự báo ngoài thực tế\n")
history_data = pd.concat([train_scaled, val_scaled])
predictions = []
actuals = []

for t in range(len(test_scaled)):
    # Lấy window cuối cùng làm input
    window_data = history_data.iloc[-L:][FEATURES].values
    X_input = window_data.reshape(1, L, len(FEATURES))

    # Dự báo t+1
    y_hat = model.predict(X_input, verbose=0)[0][0]
    y_true = test_scaled.iloc[t][TARGET]

    predictions.append(y_hat)
    actuals.append(y_true)

    # Cập nhật history bằng GIÁ TRỊ THẬT (đúng walk-forward)
    history_data = pd.concat([history_data, test_scaled.iloc[t:t+1]])

predictions = np.array(predictions)
actuals = np.array(actuals)

# 9. Inverse Scaling (CHỈ CHO LOAD)
# Scaler được fit trên toàn bộ cột của df (YEAR, Month, ...) nên phải lấy đúng vị trí cột load_MW
load_index = train_df.columns.get_loc(TARGET)

def inverse_transform_load(values, scaler, feature_index):
    dummy = np.zeros((len(values), scaler.mean_.shape[0]))
    dummy[:, feature_index] = values
    return scaler.inverse_transform(dummy)[:, feature_index]

pred_inv = inverse_transform_load(predictions, scaler, load_index)
true_inv = inverse_transform_load(actuals, scaler, load_index)

# 10. Đánh giá mô hình
rmse = np.sqrt(mean_squared_error(true_inv, pred_inv))
mae  = mean_absolute_error(true_inv, pred_inv)
mask = true_inv != 0  # có giờ load = 0 MW, loại ra để MAPE không bị chia cho 0
mape = np.mean(np.abs((true_inv[mask] - pred_inv[mask]) / true_inv[mask])) * 100

logging.info("9. ĐÁNH GIÁ HIỆU SUẤT MÔ HÌNH LSTM (TRÊN TEST SET)")
logging.info(f"RMSE : {rmse}")
logging.info(f"MAE  : {mae}")
logging.info(f"MAPE : {mape}%")

logging.info("→ LSTM cho sai số thấp, dự báo bám sát xu hướng thực tế\n")
logging.info("-"*80 + "\n")


logging.info("10. TRỰC QUAN HÓA KẾT QUẢ")
logging.info("• 1. Biểu đồ Actual vs Predicted Load")
logging.info("• 2. Biểu đồ Training Loss vs Validation Loss")
logging.info("→ Dùng để đánh giá độ chính xác và overfitting\n")

# 11. Plot Actual vs Predicted Load
plt.figure(figsize=(14, 6))
plt.plot(true_inv,label="Actual Load",color="black",linewidth=2)
plt.plot(pred_inv,label="Predicted Load (LSTM)",color="red",linestyle="--")
plt.title("LSTM Forecast – Actual vs Predicted Electricity Load")
plt.xlabel("Time Steps (Test Set)")
plt.ylabel("Electricity Load (MW)")
plt.legend()
plt.grid(True)
plt.tight_layout()
save_plot("1_LSTM_Actual_vs_Predicted_Load.png", plot_dir)
plt.close()
logging.info("Đã lưu biểu đồ Actual vs Predicted Load cho LSTM")
logging.info("NHẬN XÉT KẾT QUẢ MÔ HÌNH LSTM (DEEP LEARNING)")

logging.info(
    "Biểu đồ Actual vs Predicted cho thấy mô hình LSTM dự báo điện năng tiêu thụ "
    "bám sát rất tốt chuỗi thực tế trên toàn bộ tập test. Đường dự báo gần như "
    "chồng lên đường giá trị thực, không xuất hiện hiện tượng lệch pha kéo dài "
    "hoặc trôi xu hướng theo thời gian."
)

logging.info(
    "Mô hình học rõ ràng chu kỳ ngày (24 giờ), thể hiện qua các dao động lặp lại "
    "đều đặn với khoảng cách giữa các đỉnh và đáy xấp xỉ 24 bước thời gian. "
    "Điều này cho thấy LSTM đã khai thác hiệu quả cấu trúc seasonality theo ngày "
    "của chuỗi điện năng tiêu thụ."
)

logging.info(
    "Trong các giai đoạn biến động ngắn hạn, mô hình phản ứng kịp thời với xu hướng "
    "tăng giảm của load. Tuy nhiên, tại một số đỉnh và đáy nhọn, giá trị dự báo "
    "có xu hướng thấp hơn hoặc cao hơn nhẹ so với giá trị thực, cho thấy hiện tượng "
    "làm mượt (smoothing). Đây là đặc điểm điển hình của LSTM khi tối ưu hàm mất mát MSE."
)

logging.info(
    "Tại các thời điểm xuất hiện biến động bất thường hoặc outlier (ví dụ giai đoạn "
    "đầu tập test với mức giảm load đột ngột), mô hình không phản ứng cực đoan "
    "như dữ liệu thực tế. Điều này giúp mô hình ổn định hơn nhưng làm sai số tăng cục bộ "
    "tại các điểm hiếm gặp."
)

logging.info(
    "Ở phần cuối tập test, mô hình duy trì khả năng dự báo ổn định, không xuất hiện "
    "hiện tượng tích lũy lỗi theo thời gian. Điều này cho thấy quy trình walk-forward "
    "validation đã được triển khai đúng, sử dụng giá trị thực để cập nhật history, "
    "phù hợp với kịch bản dự báo ngoài thực tế."
)

logging.info("")

logging.info("ĐÁNH GIÁ HIỆU SUẤT ĐỊNH LƯỢNG (TEST SET):")
logging.info(f"• RMSE : {rmse}  → Sai số bình phương trung bình thấp, phản ánh độ bám sát tổng thể tốt")
logging.info(f"• MAE  : {mae}  → Sai số tuyệt đối trung bình nhỏ, cho thấy dự báo ổn định")
logging.info(f"• MAPE : {mape}% → Đã loại các giờ load = 0 MW; MAPE vẫn bị khuếch đại ở các giờ load rất nhỏ")

logging.info(
    "Các chỉ số RMSE và MAE thấp phù hợp với quan sát trực quan từ biểu đồ dự báo. "
    "Sự nhất quán giữa đánh giá định lượng và trực quan cho thấy mô hình LSTM "
    "có khả năng tổng quát hóa tốt và không bị overfitting."
)

logging.info(
    "Dựa trên đường Training Loss và Validation Loss, hai đường loss giảm đồng thời "
    "và không tách xa nhau. Validation loss không tăng mạnh sau khi đạt giá trị nhỏ nhất, "
    "cho thấy mô hình không bị overfitting và quá trình EarlyStopping đã hoạt động hiệu quả."
)

logging.info(
    "Tổng hợp kết quả cho thấy mô hình LSTM phù hợp cho bài toán dự báo điện năng "
    "đa biến, học tốt quan hệ phi tuyến giữa load và các biến thời tiết, đồng thời "
    "mô hình hóa hiệu quả chu kỳ ngày của chuỗi thời gian."
)

# Plot Training Loss vs Validation Loss (CHECK OVERFITTING)
logging.info("VẼ BIỂU ĐỒ TRAIN LOSS VS VALIDATION LOSS (LSTM)")
plt.figure(figsize=(8, 5))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title("LSTM Training vs Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss (MSE)")
plt.legend()
plt.grid(True)
save_plot("2_LSTM_Train_vs_Val_Loss.png", plot_dir)
plt.close()
logging.info("Đã lưu biểu đồ Train vs Validation Loss cho LSTM")
logging.info("PHÂN TÍCH TRAINING LOSS VS VALIDATION LOSS (LSTM)")
logging.info(
    "Biểu đồ Training Loss và Validation Loss cho thấy mô hình LSTM hội tụ nhanh "
    "trong những epoch đầu tiên. Cả hai đường loss đều giảm mạnh từ epoch 0 đến epoch 2, "
    "cho thấy mô hình học hiệu quả các đặc trưng chính của dữ liệu."
)

best_epoch = int(np.argmin(history.history['val_loss']))
logging.info(
    f"Validation loss đạt giá trị nhỏ nhất tại epoch {best_epoch} "
    f"(val_loss = {min(history.history['val_loss']):.4f}), mô hình dừng sau "
    f"{len(history.history['loss'])} epoch nhờ EarlyStopping, trong khi training loss "
    "tiếp tục giảm chậm."
)

logging.info(
    "Sau epoch tối ưu, validation loss nhích lên nhẹ trong khi training loss vẫn giảm, "
    "cho thấy dấu hiệu overfitting rất nhẹ. Validation loss thấp hơn training loss "
    "trong suốt quá trình huấn luyện, nhiều khả năng do giai đoạn validation (7–9/2022) "
    "ít spike bất thường hơn tập train."
)

logging.info(
    "Việc sử dụng EarlyStopping với cơ chế restore_best_weights giúp mô hình dừng huấn luyện "
    "tại epoch có validation loss tốt nhất, đảm bảo khả năng tổng quát hóa trên dữ liệu mới."
)

logging.info(
    "Tổng hợp biểu đồ loss cho thấy mô hình LSTM không bị underfitting hoặc overfitting "
    "nghiêm trọng, quá trình huấn luyện ổn định và phù hợp với bài toán dự báo chuỗi "
    "thời gian đa biến."
)

# Giá trị metric (đã tính)
metrics = ['RMSE', 'MAE', 'MAPE']
values  = [rmse, mae, mape]

plt.figure(figsize=(6, 4))
plt.bar(metrics, values)

plt.title('Model Evaluation Metrics (LSTM - Test Set)')
plt.ylabel('Value')
plt.grid(axis='y', linestyle='--', alpha=0.7)
# Hiển thị giá trị trên cột
for i, v in enumerate(values):
    plt.text(i, v, f'{v:.2f}', ha='center', va='bottom')
plt.tight_layout()
save_plot("3_LSTM_Evaluation_Metrics.png", plot_dir)
plt.close()
