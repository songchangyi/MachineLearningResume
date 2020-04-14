# 半监督学习 (Semi-supervised Learning)
以下内容基于周志华老师的机器学习第13章归纳整理。

## 1 未标记样本
未标记的样本就是没有被人工标注的数据。虽然没有直接包含标记信息，但是如果他们与已标记的部分从同样的数据源独立同分布采用而来，就包含了大量的信息。
我们需要找到利用它们帮助建模的方法。

让学习器不依赖外界交互、自动利用未标记样本来提升学习性能，就是半监督学习。该需求在现实中很强烈，因为大多数情况下我们更容易收集到的是未标注数据。

这里首先需要做一些相关假设。

- **聚类假设(cluster assumption)**：假设数据存在簇结构，同一个簇属于同一个类别。

比如下面这个例子，看左图我们无法确定待测样本的正负。但如果我们发现该样本属于左边这个簇，那我们就认为它应该归类到左边。

![Image of cluster label](https://github.com/songchangyi/MachineLearningResume/blob/master/img/cluster_label.PNG)

- **流形假设(manifold assumption)**：假设数据分布在一个流形结构上，邻近样本的输出值类似。可以看作是聚类假设的推广。

当然，以上两个假设背后的思想都是**相似输入拥有相似输出**。

半监督学习可以进一步划分为纯半监督学习和直推学习。前者假定未标记样本不是我们要预测的，后者则假设就是，其目的是在这些样本上获得最优的泛化性能。

![Image of semi learn type](https://github.com/songchangyi/MachineLearningResume/blob/master/img/semi_learn_type.PNG)

## 2 生成式方法(Generative methods)
生成式方法假设所有数据都是由一个潜在的模型生成的。其中未标记数据的标记可以看作模型的缺失参数，通常可基于EM算法进行极大似然估计求解。

在神经网络中，该方法的实现叫做生成式对抗网络，也就是著名的GAN。

## 3 半监督SVM
半监督支持向量机(Semi-Supervised Support Vector Machine, S3VM)是SVM在半监督学习上的推广。
与SVM不同的是，这里的S3VM不仅要划分开已有标签的样本，同时还要满足这个超平面穿过数据低密度区域。

![Image of s3vm](https://github.com/songchangyi/MachineLearningResume/blob/master/img/s3vm.PNG)

S3VM中最著名的算法是TSVM。它的具体流程如下：
- 利用标记样本学得一个SVM
- 用这个SVM预测无标记样本
- 结合所有样本求解新的超平面和松弛向量，这时要赋予有标记样本更大的权重
- 找出两个被预测为不同类且很可能发生错误的未标记样本，交换它们的标记再重新求解超平面
- 标记调整完成后，逐渐增大未标记的权重并重复调整，直到标记和未标记数据的权重相同

## 4 图半监督学习


## 面试问题

## References
