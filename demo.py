import numpy as np
import pandas as pd
#画图
import matplotlib.pyplot as plt
#导入KMeans
from k_means import KMeans
#导入导出模型
import joblib

#导入数据
# 1.电力用户分类
# df=pd.read_csv("F:\\软件杯\\数据\\user_log2_count2.csv")
# x_axis="power_2015_average"
# y_axis="power_2016_average"


# # 2.能源分类
# df=pd.read_csv("F:\\软件杯\\数据\\household_power_consumption.csv")
# x_axis="Global_active_power"
# y_axis="Global_intensity"

# # 3. 峰值分类
# df=pd.read_csv("F:\\软件杯\\数据\\峰值.csv")
# x_axis="average"
# y_axis="average_norm"

# #4.电力用户缴费行为分类
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

#初始图
plt.figure(figsize=(20,5))
plt.subplot(1,3,1)
plt.scatter(df[x_axis][:],df[y_axis][:])
plt.title('initial figure')

num_examples=df.shape[0]
x_train=df[[x_axis,y_axis]].values.reshape(num_examples,2)

#指定训练参数
num_clusters=3   #簇的个数
max_iteritions=50   #迭代次数

k_means=KMeans(x_train,num_clusters)
centroids,closest_centroids_ids=k_means.train(max_iteritions)

for centroid_id,centroid in enumerate(centroids):
    plt.subplot(1, 3, 2)
    print(centroid_id)
    current_examples_index=(closest_centroids_ids==centroid_id).flatten()
    plt.scatter(df[x_axis][current_examples_index],df[y_axis][current_examples_index],label=centroid_id)
    index=df[x_axis][current_examples_index].index
    print(df.loc[index.values])
    plt.subplot(1,3,3)
    plt.plot(index.values,df[x_axis][current_examples_index])
    plt.title('line chart')

plt.subplot(1, 3, 2)
for centroid_id,centroid in enumerate(centroids):
    plt.scatter(centroid[0],centroid[1],c='black',marker='*')
plt.title('kmeans figure')

# joblib.dump(k_means,"k_means.pkl")

plt.legend()
plt.show()