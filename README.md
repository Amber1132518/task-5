# task-5

电力用户分析---用户分类

## 1.计算k值

根据 k值.py 计算最佳分类个数，得到拐点值的k值

电力用户分类：k=3

电力用户分类2：k=5

能源分类：k=3

峰值分类：k=2

用户缴费行为分类：k=3

激光企业分类：k=2

省份GDP电量分类：k=3

## 2.k_means.py

具体的k_means算法实现

1)先随机选择中心点(质心)，并初始化

2)训练模型

3)计算样本点到质心的距离，找到最近的

4)进行中心点位置更新

## 3.demo.py

k_means算法的实现(需要更改由不同数据集计算出来的k值)

展现分类前后结果

## 4.根据上述步骤生成每个数据集的模型
