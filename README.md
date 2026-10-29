# 📈 Phân tích Chuỗi Thời gian & Dự báo (Time Series Analysis & Forecasting)

> Tổng hợp bài thực hành môn **Phân tích Chuỗi Thời gian và Dự báo** – Trường Đại học Giao thông Vận tải TP.HCM (UTH).
> Repo đi theo lộ trình **Khám phá dữ liệu → Mô hình thống kê (ARIMA/SARIMA) → Mô hình Machine Learning (XGBoost)**, mỗi bước đều có code, biểu đồ và nhận xét.

![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)
![pandas](https://img.shields.io/badge/pandas-150458?logo=pandas&logoColor=white)
![statsmodels](https://img.shields.io/badge/statsmodels-ARIMA%2FSARIMA-4B8BBE)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?logo=scikitlearn&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-EB5E28)

---

## 📚 Mục lục

1. [Kiến thức đạt được](#-kiến-thức-đạt-được)
2. [Cấu trúc thư mục](#-cấu-trúc-thư-mục)
3. [Dữ liệu sử dụng](#-dữ-liệu-sử-dụng)
4. [Chương 2 – Làm việc với dữ liệu chuỗi thời gian](#-chương-2--làm-việc-với-dữ-liệu-chuỗi-thời-gian)
5. [Chương 3.1 – Mô hình thống kê ARIMA / SARIMA](#-chương-31--mô-hình-thống-kê-arima--sarima)
6. [Chương 3.2 – Mô hình Machine Learning](#-chương-32--mô-hình-machine-learning)
7. [Cài đặt & chạy](#-cài-đặt--chạy)

---

## 🎯 Kiến thức đạt được

| Chủ đề | Nội dung cụ thể |
|---|---|
| **Khám phá dữ liệu (EDA)** | Line plot, rolling mean/std, seasonal plot theo tháng, heatmap month × year |
| **Tự tương quan** | ACF, PACF, ngưỡng tin cậy 95% `±1.96/√n`, phân biệt chuỗi có cấu trúc với **white noise** |
| **Phân rã chuỗi** | STL decomposition: Trend – Seasonal – Residual |
| **Chất lượng dữ liệu** | Xử lý missing (Forward-fill vs Interpolation), phát hiện outlier bằng Z-score |
| **Tính dừng** | Kiểm định **ADF** (H₀: có nghiệm đơn vị), sai phân thường `d` và sai phân mùa vụ `D` |
| **Chọn bậc mô hình** | Đọc ACF/PACF để chọn `(p, q)` và `(P, Q)`, nhận diện **over-differencing** |
| **ARIMA / SARIMA** | Fit mô hình, so sánh AIC/BIC, chẩn đoán phần dư (residual ≈ white noise) |
| **Supervised learning cho chuỗi thời gian** | Tạo lag features, rolling features, chia train/test theo thời gian (không shuffle) |
| **Tránh data leakage** | Rolling feature phải `shift(1)` trước khi tính để không chứa giá trị cần dự báo |
| **Machine Learning** | XGBoost, Random Forest, Linear Regression, feature importance |
| **Đánh giá & tối ưu** | MAE, RMSE, MAPE · Multi-step forecasting (t+1…t+3) · Tune tham số bằng `TimeSeriesSplit` |

---

## 🗂 Cấu trúc thư mục

```
Mis_TimeSeries_Lesson/
├── Data/                              # Dataset (xem mục Dữ liệu)
├── WokingWithDataset(C2)/             # Chương 2: EDA, ACF/PACF, decomposition, data quality
│   ├── bai1_acf_pacf.py
│   ├── bai2_whitenoise.py
│   ├── bai3_decomposition.py
│   ├── bai4_data_quality.py
│   ├── bai5_visualize.py
│   └── bai6_report.py
├── Statistical_model(C3_P1)/          # Chương 3 – Phần 1: mô hình thống kê
│   ├── ARIMA/ARIMA_MODEL.py           #   Monthly Retail Sales
│   ├── SARIMA/SARIMA_MODEL.py         #   AirPassengers
│   └── Plot/                          #   Biểu đồ kết quả
├── Machine_Learning_model(C3_P2)/     # Chương 3 – Phần 2: mô hình ML
│   ├── MachineLearning.py             #   PJM Hourly Energy Consumption
│   └── plot/
└── khác(TaiLieuThamKhao)/             # Tài liệu & code tham khảo (z-table, ADF, ACF/PACF...)
```

---

## 💾 Dữ liệu sử dụng

| Dataset | Tần suất | Số quan sát | Dùng ở |
|---|---|---|---|
| `daily-min-temperatures.csv` – Nhiệt độ thấp nhất hằng ngày (Melbourne, 1981–1990) | Ngày | 3.650 | C2 – Bài 1, 4, 6 |
| `AirPassengers.csv` – Số hành khách hàng không (1949–1960) | Tháng | 144 | C2 – Bài 2, 3, 5 · C3 – SARIMA |
| `Monthly_Retail_Sales.csv` – Doanh số bán lẻ (1991–2008) | Tháng | 204 | C3 – ARIMA |
| `ML_DATA/PJMW_hourly.csv` – [PJM Hourly Energy Consumption](https://www.kaggle.com/datasets/robikscube/hourly-energy-consumption) | Giờ | ~143.000 | C3 – Machine Learning |

> ⚠️ Thư mục `Data/ML_DATA/` (~45 MB) không được đưa lên GitHub. Tải dataset từ Kaggle ở link trên và đặt vào `Data/ML_DATA/`.

---

## 🔍 Chương 2 – Làm việc với dữ liệu chuỗi thời gian

| Bài | Nội dung | Kết luận chính |
|---|---|---|
| **Bài 1** – ACF/PACF | Tính ACF, PACF đến 40 lags cho chuỗi nhiệt độ | ACF giảm dần, PACF có spike mạnh ở lag 1 → chuỗi có tự tương quan rõ |
| **Bài 2** – White Noise | Sinh 1.000 điểm nhiễu trắng, so sánh ACF với AirPassengers | White noise: các lag nằm trong ngưỡng 95%; AirPassengers: ACF lớn, giảm chậm → có trend & mùa vụ |
| **Bài 3** – Decomposition | STL với `period=12` trên AirPassengers | Trend tăng đều, Seasonal lặp lại hằng năm, Residual dao động quanh 0 |
| **Bài 4** – Data Quality | Giả lập 1.000 giá trị NaN, xử lý bằng ffill & interpolate; Z-score > 3 là outlier | Interpolation cho chuỗi mượt và liên tục hơn ffill |
| **Bài 5** – Visualization | Line chart, rolling mean/std (12), seasonal plot, heatmap | Lượng khách tăng theo năm, cao điểm vào tháng 7–8 |
| **Bài 6** – Report | Tổng hợp kết quả các bài trên vào file log | Chuỗi không phải white noise → phù hợp để xây dựng mô hình dự báo |

---

## 📊 Chương 3.1 – Mô hình thống kê ARIMA / SARIMA

### Quy trình Box–Jenkins đã thực hiện

```
EDA ─▶ Kiểm định ADF ─▶ Sai phân (d, D) ─▶ ACF/PACF chọn (p,q)(P,Q) ─▶ Fit mô hình ─▶ Chẩn đoán phần dư ─▶ Forecast & đánh giá
```

### 🛒 Monthly Retail Sales – `ARIMA_MODEL.py`

- Chuỗi gốc có **trend tăng + mùa vụ 12 tháng + phương sai tăng dần** → ADF xác nhận **không dừng** (p-value ≈ 1.00).
- Sai phân bậc 1 vẫn chưa dừng (p = 0.117); sai phân bậc 2 vượt qua ADF (p ≈ 3.5e-18) nhưng ACF/PACF bị kéo dài bất thường → dấu hiệu **over-differencing**. Vì vậy chọn `d = 1` và xử lý phần không dừng còn lại bằng **sai phân mùa vụ `D = 1`**.
- So sánh **ARIMA(1,1,1)** và **SARIMA(1,1,1)(1,1,1,12)** trên tập test 20%:

<table>
<tr>
<td><img src="Statistical_model(C3_P1)/Plot/ARIMA/Bai5_ARIMA_Forecast.png" alt="ARIMA forecast"></td>
<td><img src="Statistical_model(C3_P1)/Plot/ARIMA/Bai5_SARIMA_Forecast.png" alt="SARIMA forecast"></td>
</tr>
<tr>
<td align="center">ARIMA(1,1,1) – chỉ bắt được xu hướng</td>
<td align="center">SARIMA(1,1,1)(1,1,1,12) – bắt được cả mùa vụ</td>
</tr>
</table>

| Mô hình | AIC | BIC | MAE | RMSE | MAPE |
|---|---|---|---|---|---|
| ARIMA(1,1,1) | 582.47 | 591.73 | 3.667 | 4.624 | 18.04% |
| **SARIMA(1,1,1)(1,1,1,12)** | **231.11** | **245.67** | **1.502** | **1.945** | **8.69%** |

→ SARIMA giảm sai số **~58%** (RMSE 4.62 → 1.95), MAPE giảm từ 18% xuống dưới 9%.

### ✈️ AirPassengers – `SARIMA_MODEL.py`

- Chuỗi gốc không dừng (ADF p = 0.992). Sai phân thường `d=1` kết hợp sai phân mùa vụ `D=1, s=12` → chuỗi dừng (p < 0.001).
- ACF/PACF có spike ở lag 1 và lag 12 → chọn **SARIMA(1,1,1)(1,1,1,12)**.

<table>
<tr>
<td><img src="Statistical_model(C3_P1)/Plot/SARIMA/Bai3_Differenced_d1_D1.png" alt="Differenced"></td>
<td><img src="Statistical_model(C3_P1)/Plot/SARIMA/Bai6_SARIMA_Forecast.png" alt="SARIMA forecast AirPassengers"></td>
</tr>
<tr>
<td align="center">Chuỗi sau sai phân (d=1, D=1)</td>
<td align="center">Dự báo SARIMA trên tập test</td>
</tr>
</table>

| Mô hình | MAE | RMSE |
|---|---|---|
| ARIMA(1,1,1) | 85.25 | 97.50 |
| **SARIMA(1,1,1)(1,1,1,12)** | **25.27** | **31.79** |

**Kết luận:** Với dữ liệu có mùa vụ rõ, SARIMA cho sai số thấp hơn ARIMA khoảng **3 lần** vì mô hình hóa được cả xu hướng lẫn chu kỳ mùa vụ.

---

## 🤖 Chương 3.2 – Mô hình Machine Learning

**Dataset:** PJM West – điện tiêu thụ theo giờ (MW). Bài toán dự báo được chuyển thành **hồi quy có giám sát** bằng cách tạo đặc trưng từ quá khứ.

### Pipeline

1. Tạo **lag features** `lag_1 … lag_p` (mặc định `p = 3`)
2. Bổ sung **rolling mean / rolling std** (window = 3, tính trên `y.shift(1)` để tránh rò rỉ dữ liệu)
3. Chia train/test **80/20 theo thứ tự thời gian**
4. Huấn luyện **XGBoost**, so sánh với **Linear Regression** và **Random Forest**
5. **Multi-step forecasting** (dự báo t+1, t+2, t+3 bằng chiến lược direct)
6. **Tune** `max_depth`, `learning_rate` bằng `TimeSeriesSplit(n_splits=5)`

<table>
<tr>
<td><img src="Machine_Learning_model(C3_P2)/plot/ThucTe_DuBao.png" alt="Actual vs Predicted"></td>
<td><img src="Machine_Learning_model(C3_P2)/plot/CacFeatureQuanTrong.png" alt="Feature importance"></td>
</tr>
<tr>
<td align="center">Thực tế vs Dự báo (XGBoost)</td>
<td align="center">Feature importance: <code>lag_1</code> chiếm ~95%</td>
</tr>
<tr>
<td><img src="Machine_Learning_model(C3_P2)/plot/Cau2_RMSE_Lagp.png" alt="RMSE by lag"></td>
<td><img src="Machine_Learning_model(C3_P2)/plot/Cau3_RMSE_Horizon.png" alt="RMSE by horizon"></td>
</tr>
<tr>
<td align="center">RMSE theo số lag p</td>
<td align="center">RMSE theo forecast horizon</td>
</tr>
</table>

### Kết quả

**XGBoost với 3 lag** (train 114.562 giờ, test 28.641 giờ):

| MAE | RMSE | MAPE |
|---|---|---|
| 102.18 MW | 135.58 MW | **1.84%** |

**So sánh mô hình** (lag + rolling features, fold cuối của `TimeSeriesSplit`, XGBoost dùng tham số tốt nhất `max_depth=3, lr=0.1`):

| Mô hình | RMSE (MW) | MAPE |
|---|---|---|
| **Linear Regression** | **118.23** | **1.60%** |
| XGBoost (tuned) | 136.77 | 1.84% |
| Random Forest | 170.75 | – |

| Số lag `p` | 1 | 3 | 6 | 12 |
|---|---|---|---|---|
| RMSE (MW) | 213.73 | 136.07 | 135.20 | 128.61 |

| Horizon | 1 giờ | 2 giờ | 3 giờ |
|---|---|---|---|
| RMSE (MW) | 135.48 | 278.96 | 421.32 |
| MAPE | 1.85% | 3.87% | 6.00% |

### Nhận xét chính

- **`lag_1` là đặc trưng quan trọng nhất** (importance 0.95) – điện tiêu thụ giờ trước gần như quyết định giờ sau.
- **Tăng số lag**: RMSE giảm mạnh từ `p=1` lên `p=3`, sau đó chỉ cải thiện nhẹ đến `p=12` → lợi ích giảm dần khi thêm lag.
- **Dự báo càng xa sai số càng lớn**: RMSE tăng gần gấp 3 lần khi dự báo xa từ 1 giờ lên 3 giờ.
- **Linear Regression thắng XGBoost**: khi feature chỉ là lag và rolling của vài giờ gần nhất, quan hệ gần như tuyến tính nên mô hình tuyến tính đã đủ tốt. XGBoost vẫn vượt Random Forest.
- **Bài học về data leakage**: phiên bản đầu tính rolling mean có chứa `y_t`, khiến Linear Regression đạt RMSE = 0 (vì `y_t = 3·rolling_mean − lag_1 − lag_2`). Sau khi sửa bằng `shift(1)`, kết quả mới phản ánh đúng năng lực mô hình.
- **Hướng cải thiện**: thêm feature lịch (giờ trong ngày, thứ trong tuần) và lag mùa vụ (24h, 168h) để XGBoost khai thác quan hệ phi tuyến.

---

## ⚙️ Cài đặt & chạy

```bash
git clone https://github.com/HoangPhucTony/Mis_TimeSeries_Lesson.git
```

```bash
pip install pandas numpy matplotlib seaborn scipy statsmodels scikit-learn xgboost
```

```bash
python "Statistical_model(C3_P1)/SARIMA/SARIMA_MODEL.py"
```

> 📝 Các script đang dùng **đường dẫn tuyệt đối** (`C:\DOHOANGPHUC\...`). Khi chạy trên máy khác, sửa biến `path`, `LOG_DIR`, `PLOT_DIR` ở đầu mỗi file cho phù hợp.

---

## 👤 Tác giả

**Đỗ Hoàng Phúc** – Sinh viên UTH
GitHub: [@HoangPhucTony](https://github.com/HoangPhucTony)

> Repo phục vụ mục đích học tập, lưu lại kiến thức đã học trong môn Phân tích Chuỗi Thời gian và Dự báo.
