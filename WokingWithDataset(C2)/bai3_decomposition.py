import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.seasonal import STL

print('Bài tập 3: Phân rã chuỗi thời gian \n')
print('1. Chọn chuỗi dữ liệu theo tháng hoặc theo quý.\n')
print('2. Dùng phương pháp STL hoặc classical decomposition.\n')
print('3. Vẽ 4 thành phần: Observed, Trend, Seasonal, Residual.\n')
print('4. Mô tả: \n')

path = r'C:\DOHOANGPHUC\UTH\PhanTichChuoiThoiGianVaDuBao\BaiTap\Practice\Data\AirPassengers.csv'
df = pd.read_csv(path,parse_dates=["Month"],index_col="Month")

print('Dùng phương pháp STL vẽ 4 thành phần của chuỗi AirPassengers\n')
stl = STL(df['Passengers'], period=12)
res = stl.fit()

res.plot()
plt.show()
print('\n')

