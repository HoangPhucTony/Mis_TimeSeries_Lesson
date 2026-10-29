import os
from io import StringIO

import pandas as pd
import matplotlib.pyplot as plt

from utils import *
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import STL


# Đường dẫn file log
BASE_DIR = r'C:\DOHOANGPHUC\UTH\PhanTichChuoiThoiGianVaDuBao\BaiTap\Practice\HW_TimeSeries\Deep_Learning_model(C4_C5)\logs'
log_path("Data_Description", BASE_DIR)
# Đường dẫn lưu hình ảnh biểu đồ
plot_dir = r'C:\DOHOANGPHUC\UTH\PhanTichChuoiThoiGianVaDuBao\BaiTap\Practice\HW_TimeSeries\Deep_Learning_model(C4_C5)\plots\description'
data_path = r'C:\DOHOANGPHUC\UTH\PhanTichChuoiThoiGianVaDuBao\BaiTap\Practice\Data\data_lab4\weather and load dataset.csv'
df = load_dataset(data_path)

logging.info("Sử dụng Dataset: weather and load dataset.csv")
logging.info("Đường dẫn dữ liệu sử dụng:")
logging.info(r'C:\DOHOANGPHUC\UTH\PhanTichChuoiThoiGianVaDuBao\BaiTap\Practice\Data\data_lab4\weather and load dataset.csv' + "\n")
logging.info(f"Khoảng thời gian: {df.index.min()} → {df.index.max()}")


logging.info(f"Dataset shape: {df.shape}")
logging.info(f"Các cột trong dataset: {df.columns.tolist()}")
logging.info("\n")

logging.info("5 dòng đầu dữ liệu:\n")
logging.info("=> Đã set index Datetime(Gộp cột YEAR, Month, Day, Hour) và sắp xếp theo index\n")
logging.info("=> Đã drop các cột thời gian YEAR, Month, Day, Hour\n")
df.drop(columns=["YEAR", "Month", "Day", "Hour"], inplace=True)
logging.info(df.head(5).to_string() + "\n")

logging.info("DESCRIPTIVE STATISTICS:")
desc = df[['load_MW', 'humidity', 'wind_speed', 'temperature']].describe()
logging.info("Thống kê mô tả các biến trong dataset:")
logging.info(desc.to_string() + "\n")

logging.info("Data quality:")
logging.info("Kiểm tra giá trị missing:")
missing = df.isna().sum()
logging.info("Số lượng missing values theo từng cột:")
logging.info(missing.to_string() + "\n")

logging.info("Kiểm tra giá trị duplicate:")
duplicate = df.duplicated().sum()
logging.info(f"Số lượng duplicate values: {duplicate}\n")


logging.info("Kiểm tra dữ liệu ngoại lai (Outliers Detection)")
df_col = df[['load_MW', 'humidity', 'wind_speed', 'temperature']]
logging.info("Sử dụng phương pháp IQR để định lượng outliers:")
for col in df_col.columns:
    Q1 = df_col[col].quantile(0.25)
    Q3 = df_col[col].quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    outliers = df_col[
        (df_col[col] < lower_bound) | (df_col[col] > upper_bound)
    ]

    logging.info(
        f"{col}: {len(outliers)} outliers "
        f"(Lower={lower_bound:.2f}, Upper={upper_bound:.2f})"
    )

logging.info("\n")

logging.info("Vẽ biểu đồ Line plot (Load_MW) theo thời gian:")
plt.figure()
plt.plot(df['load_MW'])
plt.title("Electricity Load(MW) over Time")
plt.xlabel("Time")
plt.ylabel("Electric Load (MW)")
plt.tight_layout()
save_plot("1_load_MW_over_time", plot_dir)
plt.close()
logging.info(f"Biểu đồ đã lưu tại thư mục: {plot_dir}\n")

logging.info("Nhận xét biểu đồ Lineplot: ")
logging.info("Điện năng tiêu thụ có xu hướng tăng dần theo thời gian, Từ 11/2021 → khoảng giữa 2022: load tăng rõ rệt")
logging.info("=> Có trend => Chuỗi không dừng")
logging.info("Ngoài ra, chuỗi có dao động lặp lại dày đặc theo thời gian => Có seasonality")
logging.info("Có nhiều spike rất thấp (gần 0 MW) và spike rất cao => Có ouliers")
logging.info("\n\n")


logging.info("Vẽ biểu đồ ACF và PACF của biến load_MW để kiểm tra chuỗi: ")
plt.figure()
plot_acf(df['load_MW'], lags=200)
plt.title("ACF of Electricity Load")
plt.tight_layout()
save_plot("2_acf_load", plot_dir)
plt.close()
logging.info(f"Biểu đồ đã lưu tại thư mục: {plot_dir}\n")

logging.info("NHẬN XÉT ACF:")
logging.info(
    "- Các giá trị ACF tại các lag nhỏ (1–5) rất cao, cho thấy sự phụ thuộc ngắn hạn mạnh của Electric Load.\n"
    "- Các đỉnh (peaks) lặp lại rõ ràng tại các lag xấp xỉ 24, 48, 72,... xác nhận chu kỳ ngày (daily seasonality).\n"
    "- Ngoài ra, các peaks tại lag xấp xỉ 168 cho thấy sự tồn tại của chu kỳ tuần (weekly seasonality).\n"
    "- ACF giảm chậm theo độ trễ, cho thấy chuỗi không dừng (non-stationary) và tồn tại cả xu hướng và seasonality.\n"
)

plt.figure()
plot_pacf(df['load_MW'], lags=50, method='ywm')
plt.title("PACF of Electricity Load")
plt.tight_layout()
save_plot("3_pacf_load", plot_dir)
plt.close()
logging.info(f"Biểu đồ đã lưu tại thư mục: {plot_dir}")

logging.info("NHẬN XÉT PACF:")
logging.info(
    "- PACF có spike rất mạnh tại lag = 1, cho thấy Electric Load phụ thuộc trực tiếp vào giờ liền trước.\n"
    "- Sau vài lag đầu, các giá trị PACF dao động quanh 0 và phần lớn nằm trong khoảng tin cậy.\n"
    "- Điều này cho thấy ảnh hưởng trực tiếp của các lag xa hơn là không đáng kể khi đã loại bỏ ảnh hưởng trung gian.\n"
    "- PACF không thể hiện rõ seasonality, điều này là bình thường vì tính mùa vụ chủ yếu được phản ánh trong ACF.\n"
)


logging.info("Vẽ biểu đồ Load vs Temperature:")
plt.figure()
plt.scatter(df['temperature'], df['load_MW'], s=10)
plt.title("Electric Load vs Temperature")
plt.xlabel("Temperature (°C)")
plt.ylabel("Electric Load (MW)")
plt.tight_layout()
save_plot("4_load_vs_temperature", plot_dir)
plt.close()

logging.info(f"Biểu đồ đã lưu tại thư mục: {plot_dir}")
logging.info("NHẬN XÉT Load vs Temperature:")
logging.info(
    "- Biểu đồ scatter cho thấy electric load có xu hướng tăng khi nhiệt độ tăng.\n"
    "- Ở vùng nhiệt độ cao, mức tiêu thụ điện cao hơn và phân tán lớn hơn.\n"
    "- Điều này cho thấy nhiệt độ có ảnh hưởng rõ rệt đến điện năng tiêu thụ.\n"
    "- Xuất hiện một số điểm ngoại lai (outliers) với electric load rất cao (≈ 160–180 MW) ở vùng nhiệt độ trung bình–cao.\n"
    "- Có một dải điểm tại electric load ≈ 0 MW trải dài trên nhiều mức nhiệt độ, có thể liên quan đến dữ liệu bất thường hoặc thời điểm không tiêu thụ.\n"
    "=> Tổng thể, biểu đồ cho thấy nhiệt độ có ảnh hưởng rõ ràng đến điện năng tiêu thụ, nhưng không phải là yếu tố duy nhất quyết định load.\n"
)

logging.info("Vẽ biểu đồ Load vs Humidity:")
plt.figure()
plt.scatter(df['humidity'], df['load_MW'], s=10)
plt.title("Electric Load vs Humidity")
plt.xlabel("Humidity")
plt.ylabel("Electric Load (MW)")
plt.tight_layout()
save_plot("5_load_vs_humidity", plot_dir)
plt.close()

logging.info(f"Biểu đồ đã lưu tại thư mục: {plot_dir}")
logging.info("NHẬN XÉT Load vs Humidity:")
logging.info(
    "- Biểu đồ scatter cho thấy electric load phân tán rộng trên toàn bộ dải giá trị humidity.\n"
    "- Không quan sát thấy xu hướng tăng hoặc giảm rõ ràng của load theo humidity.\n"
    "- Tại cùng một mức humidity, electric load có thể nhận nhiều giá trị khác nhau.\n"
    "- Điều này cho thấy humidity không phải là yếu tố quyết định chính đến điện năng tiêu thụ.\n"
)

logging.info("Vẽ biểu đồ Load vs Wind Speed:")
plt.figure()
plt.scatter(df['wind_speed'], df['load_MW'], s=10)
plt.title("Electric Load vs Wind Speed")
plt.xlabel("Wind Speed")
plt.ylabel("Electric Load (MW)")
plt.tight_layout()
save_plot("6_load_vs_wind_speed", plot_dir)
plt.close()

logging.info(f"Biểu đồ đã lưu tại thư mục: {plot_dir}")
logging.info("NHẬN XÉT Load vs Wind Speed:")
logging.info(
    "- Biểu đồ scatter cho thấy electric load không có mối quan hệ rõ ràng với wind speed.\n"
    "- Electric load phân bố rải rác trên toàn bộ dải giá trị của wind speed.\n"
    "- Không quan sát được xu hướng tuyến tính hay phi tuyến rõ rệt giữa hai biến.\n"
    "- Wind speed có ảnh hưởng rất yếu đến điện năng tiêu thụ so với temperature.\n"
)
logging.info("TỔNG KẾT 3 BIỂU ĐỒ Load vs Weather Variables:")
logging.info(
    "=> Qua phân tích trực quan, temperature cho thấy mối quan hệ rõ ràng nhất với electric load, "
    "trong khi humidity và wind_speed không thể hiện xu hướng ảnh hưởng rõ rệt.\n\n"
)


logging.info("Vẽ ma trận tương quan giữa các biến:")

corr_cols = ['load_MW', 'temperature', 'humidity', 'wind_speed']
corr_matrix = df[corr_cols].corr()

plt.figure()
plt.imshow(corr_matrix)
plt.colorbar()
plt.xticks(range(len(corr_cols)), corr_cols, rotation=45)
plt.yticks(range(len(corr_cols)), corr_cols)
plt.title("Correlation Matrix: Load & Weather Variables")
plt.tight_layout()
save_plot("7_correlation_matrix", plot_dir)
plt.close()

logging.info(f"Biểu đồ đã lưu tại thư mục: {plot_dir}")
logging.info("NHẬN XÉT Correlation Matrix:")
logging.info(
    "- Electric load có tương quan dương mức trung bình với temperature.\n"
    "- Humidity và wind_speed có tương quan yếu với electric load.\n"
    "- Temperature là biến thời tiết có mức tương quan cao nhất với load trong các biến được xét.\n"
    "- Mối tương quan giữa các biến thời tiết với nhau nhìn chung ở mức thấp.\n"
)


logging.info("Thực hiện STL Decomposition cho biến load_MW:")
# STL decomposition (chu kỳ ngày: 24 giờ)
stl = STL(df['load_MW'], period=24, robust=True)
stl_result = stl.fit()

plt.figure()

plt.subplot(4, 1, 1)
plt.plot(stl_result.observed)
plt.title("Observed (Electric Load)")

plt.subplot(4, 1, 2)
plt.plot(stl_result.trend)
plt.title("Trend")

plt.subplot(4, 1, 3)
plt.plot(stl_result.seasonal)
plt.title("Seasonal")

plt.subplot(4, 1, 4)
plt.plot(stl_result.resid)
plt.title("Residual")

plt.tight_layout()
save_plot("8_stl_decomposition_load", plot_dir)
plt.close()

logging.info(f"Biểu đồ STL Decomposition đã lưu tại thư mục: {plot_dir}\n")
logging.info("NHẬN XÉT STL Decomposition:")
logging.info(
    "- Thành phần Observed cho thấy electric load dao động mạnh theo thời gian và xuất hiện nhiều spike cao/thấp bất thường.\n"
    "- Thành phần Trend thể hiện xu hướng tăng rõ rệt từ cuối 2021 đến giữa 2022, sau đó dao động và giảm nhẹ về cuối kỳ.\n"
    "- Thành phần Seasonal lặp lại ổn định theo thời gian với biên độ tương đối đều, phản ánh chu kỳ ngày (24 giờ).\n"
    "- Thành phần Residual dao động quanh 0 nhưng vẫn còn nhiều spike lớn, cho thấy tồn tại nhiễu và outliers.\n"
)



