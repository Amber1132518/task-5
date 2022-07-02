import pandas as pd
import matplotlib.pyplot as plt
import openpyxl

#导入数据
# 1.电力用户用电分类
# df=pd.read_csv("F:\\软件杯\\数据\\user_log2_count2.csv")
# x_axis="power_2015_average"
# y_axis="power_2016_average"

# # 2. 能源分类
# df=pd.read_csv("F:\\软件杯\\数据\\household_power_consumption.csv")
# x_axis="Global_active_power"
# y_axis="Global_intensity"

# # 3. 峰值分类
# df=pd.read_csv("F:\\软件杯\\数据\\峰值.csv")
# x_axis="average"
# y_axis="average_norm"

# #4.电力用户缴费行文分类
# df=pd.read_csv("F:\\软件杯\\数据\\user_log_count.csv")
# x_axis="money_sum"
# y_axis="money_average"

# #5.激光企业分类
# df=pd.read_csv("F:\\软件杯\\数据\\激光企业.csv")
# x_axis="综合倍率"
# y_axis="总用电量"

#6.省份GDP电量关系
df=pd.read_csv("F:\\软件杯\\数据\\省份GDP电量关系.csv")
x_axis="近10年年均GDP排行"
y_axis="近10年年均用电量排行"

num_examples=df.shape[0]
x_train=df[[x_axis,y_axis]].values.reshape(num_examples,2)

from sklearn.cluster import KMeans
SSE = []            # 存放每次结果的误差平方和
for k in range(1, 9):
    estimator = KMeans(n_clusters=k)  # 构造聚类器
    estimator.fit(x_train)
    SSE.append(estimator.inertia_)
X = range(1, 9)

plt.figure(figsize=(15, 10))
plt.xlabel('k')
plt.ylabel('SSE')
plt.plot(X, SSE, 'o-')
plt.show()
