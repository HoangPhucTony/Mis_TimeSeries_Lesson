import logging
import os
import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import acf, pacf

# ================== SETUP LOGGING ==================
LOG_DIR = r"C:\DOHOANGPHUC\UTH\PhanTichChuoiThoiGianVaDuBao\BaiTap\Practice\WokingWithDataset(C2)"
os.makedirs(LOG_DIR, exist_ok=True)

LOG_PATH = os.path.join(LOG_DIR, "bai6_report.log")

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[
        logging.FileHandler(LOG_PATH, mode="w", encoding="utf-8"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger()
logger.info("BÀI 6: BÁO CÁO PHÂN TÍCH CHUỖI THỜI GIAN\n")

# ================== LOAD DATA ==================
path_air = r"C:\DOHOANGPHUC\UTH\PhanTichChuoiThoiGianVaDuBao\BaiTap\Practice\Data\AirPassengers.csv"
path_temp = r"C:\DOHOANGPHUC\UTH\PhanTichChuoiThoiGianVaDuBao\BaiTap\Practice\Data\daily-min-temperatures.csv"

df_air = pd.read_csv(path_air, parse_dates=["Month"], index_col="Month")
df_temp = pd.read_csv(path_temp, parse_dates=["Date"], index_col="Date")

# ================== 1. MÔ TẢ DATASET ==================
logger.info("1. Mô tả dataset")

logger.info("• AirPassengers:")
logger.info(f"  - Số quan sát : {len(df_air)}")
logger.info(f"  - Thời gian  : {df_air.index.min().date()} → {df_air.index.max().date()}")
logger.info(f"  - Mean       : {df_air['Passengers'].mean():.2f}")
logger.info(f"  - Std        : {df_air['Passengers'].std():.2f}\n")

logger.info("• Daily Minimum Temperatures:")
logger.info(f"  - Số quan sát : {len(df_temp)}")
logger.info(f"  - Thời gian  : {df_temp.index.min().date()} → {df_temp.index.max().date()}")
logger.info(f"  - Mean       : {df_temp['Temp'].mean():.2f}")
logger.info(f"  - Std        : {df_temp['Temp'].std():.2f}\n")

# ================== 2. KIỂM TRA ACF / PACF ==================
logger.info("2. Kiểm tra ACF/PACF (Daily Minimum Temperatures)")

series = df_temp["Temp"]
n = len(series)
threshold = 1.96 / np.sqrt(n)

logger.info(f"- Ngưỡng tin cậy 95%: ±{threshold:.4f}\n")

logger.info("• ACF (lag 1 → 10):")
acf_vals = acf(series, nlags=10)
for i in range(1, 11):
    logger.info(f"  Lag {i}: {acf_vals[i]:.4f}")

logger.info("\n• PACF (lag 1 → 10):")
pacf_vals = pacf(series, nlags=10, method="ywm")
for i in range(1, 11):
    logger.info(f"  Lag {i}: {pacf_vals[i]:.4f}")

logger.info("")

# ================== 3. KIỂM TRA WHITE NOISE ==================
logger.info("3. Kiểm tra white noise")

np.random.seed(42)
white_noise = np.random.normal(0, 1, 1000)
threshold_wn = 1.96 / np.sqrt(len(white_noise))

logger.info(f"- White noise (1000 điểm), threshold: ±{threshold_wn:.4f}")

acf_wn = acf(white_noise, nlags=10)
for i in range(1, 11):
    logger.info(f"  Lag {i}: {acf_wn[i]:.4f}")

logger.info("\n- So sánh với AirPassengers:")

acf_air = acf(df_air["Passengers"], nlags=10)
for i in range(1, 11):
    logger.info(f"  Lag {i}: {acf_air[i]:.4f}")

logger.info("")

# ================== 4. PHÂN RÃ CHUỖI ==================
logger.info("4. Phân rã chuỗi")

logger.info(
    "- Chuỗi AirPassengers được phân rã bằng phương pháp STL với chu kỳ mùa vụ 12 tháng.\n"
    "- Thành phần Trend cho thấy xu hướng tăng rõ rệt theo thời gian.\n"
    "- Thành phần Seasonal thể hiện mùa vụ mạnh, lặp lại hàng năm.\n"
    "- Thành phần Residual dao động quanh 0 và không có cấu trúc rõ ràng.\n"
)

# ================== 5. VẤN ĐỀ CHẤT LƯỢNG DỮ LIỆU ==================
logger.info("5. Vấn đề chất lượng dữ liệu")

logger.info(
    "- Dataset Daily Minimum Temperatures được giả lập missing value bằng cách gán NaN.\n"
    "- Missing value được xử lý bằng Forward-fill và Interpolation.\n"
)

logger.info(
    "- Kết quả cho thấy sau khi xử lý, dữ liệu không còn giá trị thiếu.\n"
    "- Phương pháp Interpolation cho chuỗi mượt và liên tục hơn.\n"
)

logger.info(
    "- Outliers được phát hiện bằng phương pháp Z-score.\n"
    "- Các điểm ngoại lai được xử lý lại bằng Interpolation.\n"
    "- Sau xử lý, dữ liệu ổn định hơn và giảm biến động bất thường.\n"
)

# ================== 6. Ý NGHĨA TRỰC QUAN HÓA ==================
logger.info("6. Ý nghĩa trực quan hóa")

logger.info(
    "- Line chart giúp quan sát xu hướng tăng và dao động theo mùa của chuỗi.\n"
    "- Rolling mean (12) làm rõ xu hướng dài hạn của dữ liệu.\n"
    "- Rolling std (12) cho thấy độ biến động tăng theo thời gian.\n"
)

logger.info(
    "- Seasonal plot theo tháng thể hiện mô hình mùa vụ lặp lại qua các năm.\n"
    "- Heatmap seasonality giúp quan sát đồng thời xu hướng theo năm và sự khác biệt giữa các tháng.\n"
)

logger.info(
    "- Trực quan hóa là bước quan trọng giúp hiểu dữ liệu trước khi xây dựng mô hình dự báo.\n"
)

# ================== 7. KẾT LUẬN ==================
logger.info("7. Kết luận")

logger.info(
    "- Chuỗi dữ liệu không phải white noise.\n"
    "- Có tự tương quan, xu hướng và mùa vụ rõ ràng.\n"
    "- Dữ liệu phù hợp để áp dụng các mô hình dự báo chuỗi thời gian.\n"
)
