# 特征选择与稀疏学习
以下内容基于周志华老师的机器学习第11章归纳整理。

## 1 子集搜索与评价
我们将属性称为特征(feature)，将从给定特征集中选择相关特征子集的过程称为特征选择(feature selection)。
常见的特征选择方法大致分为三类：过滤式、包裹式和嵌入式。

## 2 过滤式
这种方法是先对数据集进行特征选择，然后再训练学习器。

过滤式选择的方法有 [CSDN](https://blog.csdn.net/pxhdky/java/article/details/86305538)
1. 移除低方差的特征；
2. 相关系数排序，分别计算每个特征与输出值之间的相关系数，设定一个阈值，选择相关系数大于阈值的部分特征；
3. 利用假设检验得到特征与输出值之间的相关性，方法有比如卡方检验、t检验、F检验等。
4. 互信息，利用互信息从信息熵的角度分析相关性。

## 3 包裹式
直接把最终学习器的性能作为特征子集的评价标准。可以理解为随机的选取特征子集，多次训练，然后用训练出最优模型的一组。

一般而言，这种方法比过滤式效果更好。但是计算开销大的多。

## 4 嵌入式与L1正则化
嵌入式特征选择是将特征选择与训练融为一体，在同一个优化过程中完成。

首先我们要引入正则化参数lambda，回顾之前的线性回归模型的平方误差损失函数：

![min \sum_{i=1}^m (y_i-w^{T}x_i)^2+ \lambda{||w||_2}^2](https://render.githubusercontent.com/render/math?math=min%20%5Csum_%7Bi%3D1%7D%5Em%20(y_i-w%5E%7BT%7Dx_i)%5E2%2B%20%5Clambda%7B%7C%7Cw%7C%7C_2%7D%5E2)

上式称为L2岭回归(ridge regression)，其中lambda为正，容易看出，要最小化上式，w就不能太大，这就帮我们降低了过拟合，避免w太大而严重依赖某一项特征。

- **Code** [Sklearn](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html)
```
from sklearn.linear_model import Ridge
clf = Ridge(alpha=1.0)
clf.fit(X, y)
```

如果去掉平方项改用L1，则称为LASSO回归：

![min \sum_{i=1}^m (y_i-w^{T}x_i)^2+ \lambda||w||_1](https://render.githubusercontent.com/render/math?math=min%20%5Csum_%7Bi%3D1%7D%5Em%20(y_i-w%5E%7BT%7Dx_i)%5E2%2B%20%5Clambda%7C%7Cw%7C%7C_1)

L1和L2都有助于降低过拟合。只不过L2会惩罚的更重一点。不过L1还有个额外的好处，它更易获得稀疏解。
意味着初始的d个特征中仅有对应着w的非零分量特征才会出现在最终模型中。
换言之，基于L1的正则化是一种嵌入式的特征选择方法。

- **Code** [Sklearn](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Lasso.html)
```
from sklearn import linear_model
clf = linear_model.Lasso(alpha=0.1)
clf.fit(X, y)
```

另外，比如在逻辑回归中默认使用了L2正则化，调整参数penalty为l1即可使用L1正则化。

## 面试问题
1. 什么是L1正则化 What is L1 regularization?
2. 什么是L2正则化 What is L2 regularization?
3. 对比L1和L2正则化 Compare L1 and L2 regularization.
4. 为什么L1正则化可以得到稀疏模型 Why does L1 regularization result in sparse models?

## References
- [【机器学习】特征选择（过滤式、包裹式、嵌入式）](https://blog.csdn.net/pxhdky/java/article/details/86305538)
