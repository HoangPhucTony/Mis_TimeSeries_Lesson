import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

print('Bài tập 5: Trực quan hóa dữ liệu chuỗi thời gian \n')
print('1. Vẽ line chart toàn bộ chuỗi.\n')
print('2. Vẽ rolling mean (window=12) và rolling std\n')
print('3. Vẽ seasonal plot theo tháng (đối với dữ liệu theo tháng).\n')
print('4. Vẽ heatmap seasonality (month × year)\n')
print('5. Kết luận: Xu hướng chung, các mùa vụ nổi bật, bất thường (nếu có)\n')

path = r'C:\DOHOANGPHUC\UTH\PhanTichChuoiThoiGianVaDuBao\BaiTap\Practice\Data\AirPassengers.csv'
df = pd.read_csv(path,parse_dates=["Month"],index_col="Month")

print('1. Vẽ line chart toàn bộ chuỗi.\n')
df.plot()
plt.title('Data Air Passengers')
plt.xlabel('Month')
plt.ylabel('Passengers')
plt.show()
print('\n')

print('2. Vẽ rolling mean (window=12) và rolling std\n')
rolling_mean = df.rolling(12).mean()
rolling_std = df.rolling(12).std()

plt.plot(df, label='Observed')
plt.plot(rolling_mean, label='Rolling Mean(12)')
plt.plot(rolling_std, label='Rolling Std(12)')
plt.legend()
plt.show()
print('\n')

print('3. Vẽ seasonal plot theo tháng (đối với dữ liệu theo tháng).\n')
df['Month'] = df.index.month
df['Year'] = df.index.year
plt.figure(figsize=(12,8))
sns.lineplot(data=df, x='Month', y='Passengers', hue='Year', palette='tab10')
plt.title('Seasonal Plot theo Tháng')
plt.show()
print('\n')

print('4. Vẽ heatmap seasonality (month × year)\n')
plt.figure(figsize=(12,8))
sns.heatmap(df.pivot_table(values='Passengers', index=df.index.month, columns=df.index.year), cmap='YlGnBu')
plt.title('Heatmap Seasonality (month x year)')
plt.xlabel('Year')
plt.ylabel('Month')
plt.show()
print('\n')


