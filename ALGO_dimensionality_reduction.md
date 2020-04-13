# 降维
以下内容基于周志华老师的机器学习第10章归纳整理。

## 1 k近邻学习(k-Nearest Neighbor, kNN)
KNN是一种常用的监督学习方法。其原理非常简单：给定测试样本，找出其最靠近的已知的k个样本，然后基于这些样本给出预测。
背后的思想就是近朱者赤，近墨者黑。分类任务中使用投票法，回归任务中使用平均法。

同时KNN是一种懒惰学习，就是不需要提前训练模型。

- **Code** [Sklearn](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html)
```
from sklearn.neighbors import KNeighborsClassifier
neigh = KNeighborsClassifier(n_neighbors=3)
neigh.fit(X, y)
```

## 2 主成份分析(Principal Component Analysis, PCA)
PCA是一种常用的降维方法。简单的来说，就是找到一组新的正交基向量，将原来的高维表示投影到新的坐标系中进行降维，同时希望投影后方差最大化。

显然，我们舍弃掉了一部分维度就等于丢失了一定的信息，但这是必要的。降维后样本的密度会增大，且噪声影响减小。

- **Code** [Sklearn](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html)
```
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
pca.fit(X)
```

## 面试问题
1. 问什么我们需要降维 Why do we need dimensionality reduction techniques?
2. 为什么我们需要PCA及它是怎么运行的 Why do we need PCA and what does it do?
2. PCA之前需要做哪两步预处理 What are the two pre-processing steps that should be applied before doing PCA?
