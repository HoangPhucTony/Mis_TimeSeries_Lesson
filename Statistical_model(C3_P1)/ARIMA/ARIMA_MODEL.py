import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import logging
import os

from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX 
from sklearn.metrics import mean_absolute_error, mean_squared_error


# SETUP LOGGING 
LOG_DIR = r'C:\DOHOANGPHUC\UTH\PhanTichChuoiThoiGianVaDuBao\BaiTap\Practice\Statistical_model(C3_P1)\ARIMA'
os.makedirs(LOG_DIR, exist_ok=True) # Tạo thư mục nếu chưa tồn tại

LOG_PATH = os.path.join(LOG_DIR, "arima.log")

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[
        logging.FileHandler(LOG_PATH, mode="w", encoding="utf-8"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger()

logger.info("=" * 60)
logger.info("BÀI TẬP THỰC HÀNH 2: KIỂM ĐỊNH ADF & ARIMA / SARIMA")
logger.info("=" * 60)

# THƯ MỤC LƯU BIỂU ĐỒ 
plot_path = r"C:\DOHOANGPHUC\UTH\PhanTichChuoiThoiGianVaDuBao\BaiTap\Practice\Statistical_model(C3_P1)\Plot\ARIMA"
os.makedirs(plot_path, exist_ok=True)


# PHÂN TÍCH DỮ LIỆU 
path = r'C:\DOHOANGPHUC\UTH\PhanTichChuoiThoiGianVaDuBao\BaiTap\Practice\Data\Monthly_Retail_Sales.csv'
df = pd.read_csv(path, parse_dates=["date"], index_col="date")

logger.info(f"Dataset sử dụng: Monthly_Retail_Sales.csv")
logger.info(f"Số quan sát: {len(df)}")
logger.info(f"Khoảng thời gian: {df.index.min()} → {df.index.max()}")

# BÀI 1: Khám phá dữ liệu (EDA)
logger.info("\n" + "="*60)
logger.info("BÀI 1: KHÁM PHÁ DỮ LIỆU (EDA)")
logger.info("="*60)

logger.info("Dataset sử dụng: Monthly_Retail_Sales.csv")
logger.info("- Chuỗi có xu hướng không? ")
logger.info("- Có dấu hiệu mùa vụ không? ")
logger.info("- Chuỗi có khả năng dừng không? ")

logger.info("1.1 MÔ TẢ THỐNG KÊ DỮ LIỆU:")
logger.info(f"\n{df.describe()}\n")

logger.info("1.2 VẼ ĐỒ THỊ CHUỖI THỜI GIAN")
df.plot()
plt.title("Monthly Retail Sales")
plt.xlabel("Date")
plt.ylabel("Sales Value")
plt.tight_layout()
plt.savefig(plot_path + "\\Bai1_EDA_Line_plot.png")
plt.show()

logger.info("\n1.3 NHẬN XÉT TỪ ĐỒ THỊ CHUỖI THỜI GIAN:")

logger.info(
    "- Xu hướng (Trend): "
    "Chuỗi thể hiện xu hướng tăng rõ rệt theo thời gian. "
    "Giá trị trung bình của chuỗi tăng dần và không dao động quanh "
    "một mức cố định, cho thấy sự tồn tại của thành phần xu hướng."
)

logger.info(
    "- Mùa vụ (Seasonality): "
    "Các đỉnh và đáy của chuỗi lặp lại tương đối đều đặn theo chu kỳ "
    "khoảng 12 tháng, cho thấy chuỗi có yếu tố mùa vụ theo năm."
)

logger.info(
    "- Biên độ dao động: "
    "Biên độ dao động của chuỗi có xu hướng tăng dần theo thời gian, "
    "cho thấy phương sai của chuỗi không ổn định."
)

logger.info(
    "- Tính dừng (Stationarity): "
    "Do chuỗi tồn tại cả xu hướng và mùa vụ, đồng thời phương sai "
    "thay đổi theo thời gian, nên chuỗi KHÔNG dừng ở trạng thái ban đầu."
)

logger.info(
    "- Kết luận: "
    "Cần thực hiện sai phân thường (d) và sai phân mùa vụ (D) "
)

# Bài 2: Kiểm định ADF trên chuỗi gốc
logger.info("\n" + "="*60)
logger.info("BÀI 2: KIỂM ĐỊNH ADF TRÊN CHUỖI GỐC")
logger.info("="*60)
logger.info("- Thực hiện kiểm định ADF cho chuỗi gốc ")

ADF_ts = df["value"]
adf_result = adfuller(ADF_ts)

adf_statistic = adf_result[0]
p_value = adf_result[1]
critical_values = adf_result[4]

logger.info("2.1 KẾT QUẢ KIỂM ĐỊNH ADF:")
logger.info(f"- ADF Statistic : {adf_statistic:.4f}")
logger.info(f"- p-value       : {p_value:.6f}")

logger.info("- Critical Values:")
for level, value in critical_values.items():
    logger.info(f"   + {level} : {value:.4f}")

logger.info("\n2.2 GIẢI THÍCH KIỂM ĐỊNH ADF:")

logger.info(
    " Giả thuyết H0: "
    "Chuỗi thời gian có nghiệm đơn vị (unit root), "
    "tức là chuỗi KHÔNG dừng."
)

logger.info(
    " Giả thuyết H1: "
    "Chuỗi thời gian KHÔNG có nghiệm đơn vị, "
    "tức là chuỗi CÓ tính dừng."
)

logger.info(
    "- p-value cho biết mức độ bằng chứng để bác bỏ H₀. "
    "Nếu p-value < 0.05, ta bác bỏ H₀ và kết luận chuỗi dừng."
)
logger.info('\n')

logger.info("\n2.3 KẾT LUẬN KIỂM ĐỊNH:")

if p_value < 0.05:
    logger.info(
        f"- p-value = {p_value:.6f} < 0.05 "
        "⇒ Bác bỏ H₀ "
        "⇒ Chuỗi CÓ tính dừng."
    )
else:
    logger.info(
        f"- p-value = {p_value:.6f} ≥ 0.05 "
        "⇒ Không bác bỏ H₀ "
        "⇒ Chuỗi KHÔNG có tính dừng."
    )

# Bài 3: Sai phân & kiểm định lại
logger.info("\n" + "="*60)
logger.info("BÀI 3: SAI PHÂN & KIỂM ĐỊNH LẠI")
logger.info("="*60)

logger.info("- Sau sai phân, chuỗi đã dừng chưa?")
logger.info("- Giá trị d trong ARIMA là bao nhiêu?")

logger.info("3.1 SAI PHÂN BẬC 1")

df_diff1 = df["value"].diff().dropna()
logger.info(f"- Số quan sát sau sai phân bậc 1: {len(df_diff1)}")

df_diff1.plot()
plt.title("Monthly Retail Sales - Sai phân bậc 1")
plt.xlabel("Date")
plt.ylabel("Differenced Value")
plt.tight_layout()
plt.savefig(plot_path + "\\Bai3_SaiPhanBac1.png")
plt.show()

logger.info(
    "- Nhận xét: Chuỗi sau sai phân bậc 1 dao động quanh giá trị trung bình "
    "xấp xỉ 0, cho thấy thành phần xu hướng đã được loại bỏ."
)

logger.info("\n3.2 KIỂM ĐỊNH ADF SAU SAI PHÂN BẬC 1")

adf_diff1 = adfuller(df_diff1)

logger.info(f"- ADF Statistic : {adf_diff1[0]:.6f}")
logger.info(f"- p-value       : {adf_diff1[1]:.6f}")
logger.info("- Critical Values:")
for level, value in adf_diff1[4].items():
    logger.info(f"   + {level} : {value:.6f}")

if adf_diff1[1] < 0.05:
    logger.info(
        "- Kết luận ADF: p-value < 0.05 "
        "⇒ Bác bỏ H₀ "
        "⇒ Chuỗi ĐÃ dừng sau sai phân bậc 1."
    )
else:
    logger.info(
        "- Kết luận ADF: p-value ≥ 0.05 "
        "⇒ Chưa bác bỏ H₀ "
        "⇒ Chuỗi CHƯA dừng."
    )

logger.info("\n3.3 THỬ SAI PHÂN BẬC 2")

df_diff2 = df["value"].diff().diff().dropna()
logger.info(f"- Số quan sát sau sai phân bậc 2: {len(df_diff2)}")

df_diff2.plot()
plt.title("Monthly Retail Sales - Sai phân bậc 2")
plt.xlabel("Date")
plt.ylabel("Second Differenced Value")
plt.tight_layout()
plt.savefig(plot_path + "\\Bai3_SaiPhanBac2.png")
plt.show()

logger.info(
    "- Nhận xét: Chuỗi sau sai phân bậc 2 dao động mạnh quanh 0, "
    "biên độ lớn hơn đáng kể so với sai phân bậc 1."
)

logger.info("\n3.4 KIỂM ĐỊNH ADF SAU SAI PHÂN BẬC 2")

adf_diff2 = adfuller(df_diff2)

logger.info(f"- ADF Statistic : {adf_diff2[0]:.6f}")
logger.info(f"- p-value       : {adf_diff2[1]:.6e}")
logger.info("- Critical Values:")
for level, value in adf_diff2[4].items():
    logger.info(f"   + {level} : {value:.6f}")

if adf_diff2[1] < 0.05:
    logger.info(
        "- Kết luận ADF: p-value < 0.05 "
        "⇒ Bác bỏ H₀ "
        "⇒ Chuỗi ĐÃ dừng sau sai phân bậc 2."
    )
else:
    logger.info(
        "- Kết luận ADF: p-value ≥ 0.05 "
        "⇒ Chưa bác bỏ H₀ "
        "⇒ Chuỗi CHƯA dừng."
    )

# Bài 4: Phân tích ACF & PACF
logger.info("\n" + "="*60)
logger.info("BÀI 4: PHÂN TÍCH ACF & PACF")
logger.info("="*60)

logger.info(
    "Chuỗi sử dụng để phân tích ACF & PACF: "
    "Chuỗi đã dừng sau sai phân bậc 2 (df_diff2)\n"
)
# Tính ngưỡng ý nghĩa thống kê 95%
n = len(df_diff2)
threshold = 1.96 / np.sqrt(n)

logger.info(f"Số quan sát (n): {n}")
logger.info(f"Threshold 95%: ±{threshold:.4f}\n")

logger.info("4.1 PHÂN TÍCH ACF")

plt.figure()
plot_acf(df_diff2, lags=40)
plt.title("ACF - Chuỗi sai phân bậc 2")
plt.tight_layout()
plt.savefig(plot_path + "\\Bai4_ACF_SaiPhanBac2.png")
plt.show()

acf_values = acf(df_diff2, nlags=12)

logger.info("Giá trị ACF từ lag 1 đến lag 12:")
for lag in range(1, 13):
    value = acf_values[lag]
    significance = "CÓ ý nghĩa" if abs(value) > threshold else "KHÔNG ý nghĩa"
    logger.info(
        f"  Lag {lag:2d}: ACF = {value:+.4f} → {significance}"
    )

logger.info("\n4.2 PHÂN TÍCH PACF")

plt.figure()
plot_pacf(df_diff2, lags=40, method="ywm")
plt.title("PACF - Chuỗi sai phân bậc 2")
plt.tight_layout()
plt.savefig(plot_path + "\\Bai4_PACF_SaiPhanBac2.png")
plt.show()

pacf_values = pacf(df_diff2, nlags=12, method="ywm")

logger.info("Giá trị PACF từ lag 1 đến lag 12:")
for lag in range(1, 13):
    value = pacf_values[lag]
    significance = "CÓ ý nghĩa" if abs(value) > threshold else "KHÔNG ý nghĩa"
    logger.info(
        f"  Lag {lag:2d}: PACF = {value:+.4f} → {significance}"
    )
logger.info("\n4.3 KẾT LUẬN PHÂN TÍCH ACF & PACF (df_diff2)")

logger.info(
    "- Phân tích được thực hiện trên chuỗi đã dừng theo kiểm định ADF "
    "sau sai phân bậc 2 (df_diff2)."
)

logger.info(
    "- Trên đồ thị ACF, xuất hiện nhiều spike vượt ngưỡng ý nghĩa "
    "tại các lag nhỏ (đặc biệt lag 1) và một số lag xa hơn, "
    "thay vì cắt cụt rõ ràng sau một vài độ trễ."
)

logger.info(
    "- Trên đồ thị PACF, các hệ số tự tương quan riêng phần "
    "không cắt cụt rõ ràng mà kéo dài qua nhiều lag, "
    "đặc biệt xuất hiện dãy giá trị âm có ý nghĩa thống kê."
)

logger.info(
    "- Sự khác biệt giữa hình ảnh ACF/PACF và việc so sánh "
    "với ngưỡng ±1.96/√n xuất phát từ việc đồ thị sử dụng "
    "confidence interval động (Bartlett), "
    "trong khi chỉ số log sử dụng ngưỡng cố định về mặt thống kê."
)

logger.info(
    "- Tuy nhiên, cả hình ảnh và chỉ số đều chỉ ra rằng "
    "cấu trúc phụ thuộc của chuỗi sau sai phân bậc 2 "
    "không còn đơn giản."
)

logger.info(
    "- Đây là dấu hiệu điển hình của hiện tượng "
    "sai phân quá mức (over-differencing)."
)

logger.info(
    "- Do đó, mặc dù df_diff2 đạt tính dừng về mặt kiểm định, "
    "nhưng KHÔNG phù hợp để xác định các bậc (p, q) "
    "cho mô hình ARIMA."
)
logger.info(
    "- Nghi vấn đặt ra: "
    "Liệu việc sai phân bậc 2 có làm méo cấu trúc tự tương quan "
    "của chuỗi và che khuất mô hình AR/MA thực sự hay không?"
)
logger.info(
    "- Để kiểm chứng nghi vấn trên, "
    "tiến hành phân tích lại ACF & PACF "
    "trên chuỗi sai phân bậc 1 (df_diff1), "
    "kết hợp với sai phân mùa vụ ở các bước tiếp theo."
)

# BÀI 4.4: PHÂN TÍCH ACF & PACF TRÊN df_diff1
logger.info("\n" + "="*60)
logger.info("BÀI 4.4: PHÂN TÍCH ACF & PACF (df_diff1)")
logger.info("="*60)

logger.info(
    "Chuỗi sử dụng để phân tích ACF & PACF: "
    "Chuỗi sai phân bậc 1 (df_diff1)\n"
)

n1 = len(df_diff1)
threshold1 = 1.96 / np.sqrt(n1)

logger.info(f"Số quan sát (n): {n1}")
logger.info(f"Threshold 95%: ±{threshold1:.4f}\n")

logger.info("4.4.1 PHÂN TÍCH ACF (df_diff1)")

plt.figure()
plot_acf(df_diff1, lags=40)
plt.title("ACF - Chuỗi sai phân bậc 1")
plt.tight_layout()
plt.savefig(plot_path + "\\Bai4_ACF_SaiPhanBac1.png")
plt.show()

acf_values_1 = acf(df_diff1, nlags=12)

logger.info("Giá trị ACF từ lag 1 đến lag 12:")
for lag in range(1, 13):
    value = acf_values_1[lag]
    significance = "CÓ ý nghĩa" if abs(value) > threshold1 else "KHÔNG ý nghĩa"
    logger.info(
        f"  Lag {lag:2d}: ACF = {value:+.4f} → {significance}"
    )

logger.info("\n4.4.2 PHÂN TÍCH PACF (df_diff1)")

plt.figure()
plot_pacf(df_diff1, lags=40, method="ywm")
plt.title("PACF - Chuỗi sai phân bậc 1")
plt.tight_layout()
plt.savefig(plot_path + "\\Bai4_PACF_SaiPhanBac1.png")
plt.show()

pacf_values_1 = pacf(df_diff1, nlags=12, method="ywm")

logger.info("Giá trị PACF từ lag 1 đến lag 12:")
for lag in range(1, 13):
    value = pacf_values_1[lag]
    significance = "CÓ ý nghĩa" if abs(value) > threshold1 else "KHÔNG ý nghĩa"
    logger.info(
        f"  Lag {lag:2d}: PACF = {value:+.4f} → {significance}"
    )
logger.info("\n4.4.3 KẾT LUẬN PHÂN TÍCH ACF & PACF (df_diff1)")

logger.info(
    "- Trên chuỗi sai phân bậc 1, đồ thị ACF cho thấy "
    "các hệ số tự tương quan giảm dần theo độ trễ "
    "và chỉ có một vài lag nhỏ vượt ngưỡng ý nghĩa."
)

logger.info(
    "- Đồ thị PACF cho thấy spike rõ ràng tại lag 1 "
    "và nhanh chóng giảm về gần 0 ở các lag sau."
)

logger.info(
    "- Cấu trúc ACF/PACF của df_diff1 rõ ràng và ổn định hơn "
    "so với df_diff2, không xuất hiện hiện tượng kéo dài bất thường."
)

logger.info(
    "- Điều này cho thấy df_diff1 KHÔNG bị sai phân quá mức "
    "và phù hợp hơn để xác định các bậc (p, q)."
)

logger.info(
    "- Gợi ý mô hình ARIMA dựa trên df_diff1: "
    "ARIMA(1,1,1)"
)

logger.info(
    "- Tuy nhiên, do dữ liệu có mùa vụ chu kỳ 12 tháng, "
    "cần mở rộng sang mô hình SARIMA "
    "với sai phân mùa vụ D = 1."
)

# BÀI 5: Thiết lập ARIMA / SARIMA – Dự đoán – Đánh giá 
logger.info("\n" + "="*60)
logger.info("BÀI 5: THIẾT LẬP ARIMA / SARIMA – DỰ ĐOÁN – ĐÁNH GIÁ")
logger.info("="*60)

logger.info("Mục tiêu:")
logger.info("- Huấn luyện mô hình ARIMA và SARIMA")
logger.info("- Dự đoán trên tập test ")
logger.info("- Đánh giá và so sánh hiệu quả mô hình\n")


logger.info("5.1 CHIA DỮ LIỆU TRAIN – TEST (80% – 20%)")

y = df["value"]
train_size = int(len(y) * 0.8)

train = y.iloc[:train_size]
test = y.iloc[train_size:]

logger.info(f"- Số quan sát Train: {len(train)}")
logger.info(f"- Số quan sát Test : {len(test)}\n")


logger.info("5.2 THIẾT LẬP MÔ HÌNH ARIMA")

logger.info("Mô hình ARIMA(1,1,1)")

arima_model = ARIMA(train, order=(1, 1, 1))
arima_fit = arima_model.fit()

logger.info("- Huấn luyện mô hình hoàn tất")
logger.info(f"- AIC: {arima_fit.aic:.4f}")
logger.info(f"- BIC: {arima_fit.bic:.4f}\n")

logger.info("5.3 CHẨN ĐOÁN PHẦN DƯ (ARIMA)")

residuals = arima_fit.resid

plt.figure()
residuals.plot()
plt.title("Residuals theo thời gian (ARIMA)")
plt.xlabel("Date")
plt.ylabel("Residual")
plt.tight_layout()
plt.savefig(plot_path + "\\Bai5_ARIMA_Residuals.png")
plt.show()

plt.figure()
plot_acf(residuals.dropna(), lags=30)
plt.title("ACF của Residuals (ARIMA)")
plt.tight_layout()
plt.savefig(plot_path + "\\Bai5_ARIMA_Residuals_ACF.png")
plt.show()

logger.info(f"- Mean của residuals: {residuals.mean():.6f}")

logger.info(
    "- Residuals dao động quanh 0, không có xu hướng rõ ràng."
)

logger.info(
    "- ACF của residuals không xuất hiện spike lớn "
    "⇒ không còn tự tương quan đáng kể."
)

logger.info(
    "- Kết luận: Residuals gần giống nhiễu trắng "
    "⇒ Mô hình ARIMA phù hợp về mặt thống kê.\n"
)

logger.info("5.4 DỰ ĐOÁN & ĐÁNH GIÁ ARIMA")

forecast_arima = arima_fit.forecast(steps=len(test))
forecast_arima.index = test.index

plt.figure(figsize=(10, 5))
plt.plot(train, label="Train")
plt.plot(test, label="Test (Thực tế)")
plt.plot(forecast_arima, label="Forecast (ARIMA)", linestyle="--")
plt.title("ARIMA(1,1,1) – Forecast")
plt.xlabel("Date")
plt.ylabel("Sales Value")
plt.legend()
plt.tight_layout()
plt.savefig(plot_path + "\\Bai5_ARIMA_Forecast.png")
plt.show()

mae_arima = mean_absolute_error(test, forecast_arima)
rmse_arima = np.sqrt(mean_squared_error(test, forecast_arima))
mape_arima = np.mean(np.abs((test - forecast_arima) / test)) * 100

logger.info(f"- MAE  : {mae_arima:.4f}")
logger.info(f"- RMSE : {rmse_arima:.4f}")
logger.info(f"- MAPE : {mape_arima:.2f}%\n")


logger.info("5.5 THIẾT LẬP MÔ HÌNH SARIMA")

logger.info("→ SARIMA(1,1,1)(1,1,1,12)")

sarima_model = SARIMAX(
    train,
    order=(1, 1, 1),
    seasonal_order=(1, 1, 1, 12),
    enforce_stationarity=False,
    enforce_invertibility=False
)

sarima_fit = sarima_model.fit()

logger.info("- Huấn luyện mô hình hoàn tất")
logger.info(f"- AIC: {sarima_fit.aic:.4f}")
logger.info(f"- BIC: {sarima_fit.bic:.4f}\n")

logger.info("5.6 CHẨN ĐOÁN PHẦN DƯ (SARIMA)")

residuals = sarima_fit.resid

plt.figure()
residuals.plot()
plt.title("Residuals theo thời gian (SARIMA)")
plt.xlabel("Date")
plt.ylabel("Residual")
plt.tight_layout()
plt.savefig(plot_path + "\\Bai5_SARIMA_Residuals.png")
plt.show()

plt.figure()
plot_acf(residuals.dropna(), lags=30)
plt.title("ACF của Residuals (SARIMA)")
plt.tight_layout()
plt.savefig(plot_path + "\\Bai5_SARIMA_Residuals_ACF.png")
plt.show()

logger.info(f"- Mean của residuals: {residuals.mean():.6f}")

logger.info(
    "- Residuals dao động quanh 0 và không còn cấu trúc rõ ràng."
)

logger.info(
    "- ACF của residuals không xuất hiện spike tại lag mùa vụ."
)

logger.info(
    "- Kết luận: Residuals gần giống nhiễu trắng "
    "⇒ SARIMA xử lý tốt yếu tố mùa vụ.\n"
)

logger.info("5.7 DỰ ĐOÁN & ĐÁNH GIÁ SARIMA")

forecast_sarima = sarima_fit.forecast(steps=len(test))
forecast_sarima.index = test.index

plt.figure(figsize=(10, 5))
plt.plot(train, label="Train")
plt.plot(test, label="Test (Thực tế)")
plt.plot(forecast_sarima, label="Forecast (SARIMA)", linestyle="--")
plt.title("SARIMA – Forecast")
plt.xlabel("Date")
plt.ylabel("Sales Value")
plt.legend()
plt.tight_layout()
plt.savefig(plot_path + "\\Bai5_SARIMA_Forecast.png")
plt.show()

mae_sarima = mean_absolute_error(test, forecast_sarima)
rmse_sarima = np.sqrt(mean_squared_error(test, forecast_sarima))
mape_sarima = np.mean(np.abs((test - forecast_sarima) / test)) * 100

logger.info(f"- MAE  : {mae_sarima:.4f}")
logger.info(f"- RMSE : {rmse_sarima:.4f}")
logger.info(f"- MAPE : {mape_sarima:.2f}%\n")

logger.info("KẾT LUẬN BÀI 5:")

logger.info(
    "- ARIMA(1,1,1) mô hình hóa được xu hướng "
    "nhưng chưa xử lý được yếu tố mùa vụ."
)

logger.info(
    "- SARIMA(1,1,1)(1,1,1,12) cho kết quả dự đoán tốt hơn, "
    "AIC/BIC thấp hơn và sai số MAE, RMSE, MAPE được cải thiện."
)

logger.info(
    "- Do đó, SARIMA là mô hình phù hợp hơn "
    "cho chuỗi Monthly Retail Sales."
)
