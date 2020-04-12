# 神经网络(Neural Network)
以下内容基于周志华老师的机器学习第5章归纳整理。

## 1 神经元模型
一个神经元(neuron)就是一个接收多个输入的单元，根据每个输入的相应权重进行加权计算，再与一个定义的阈值进行比较，最后产生输出。

![y_j=f(\sum_i w_i x_i-\theta_j)](https://render.githubusercontent.com/render/math?math=y_j%3Df(%5Csum_i%20w_i%20x_i-%5Ctheta_j))

理想中的f为单位阶跃函数，但是由于其不连续、不光滑，实际经常使用sigmoid。

![Image of activation func](https://github.com/songchangyi/MachineLearningResume/blob/master/img/activation_func.PNG)

神经网络就是许多个这样的神经元按一定层次结构连接得到的。

## 2 感知机与多层网络
感知机(Perception)由两层神经元组成，一层输入层，一层输出层。输入层接收信号并传递给输出层，输出层是MP单元，又叫阈值逻辑单元。

感知机能容易的实现与、或、非运算。比如，w1 = w2 = 1，theta = 2，此时仅有x1 = x2 = 1能输出1。

更一般的，给定训练集，权重wi和阈值theta可以通过学习得到。然后阈值又可以看作一个固定输入为-1的哑结点所对应的权重，这样权重和阈值的学习就统一成了权重学习。

![w_i \leftarrow w_i+ \Delta w_i](https://render.githubusercontent.com/render/math?math=w_i%20%5Cleftarrow%20w_i%2B%20%5CDelta%20w_i)

![\Delta w_i= \eta (y- \widehat{y})x_i](https://render.githubusercontent.com/render/math?math=%5CDelta%20w_i%3D%20%5Ceta%20(y-%20%5Cwidehat%7By%7D)x_i)

其中eta称为学习率(learning rate)，取值在0和1之间。学习率强的意思就是每次更新权重时向y修正的幅度大。

**注意**，感知机只能处理线性可分的问题，即用一个线性超平面可以进行划分的情形。

为了解决非线性可分问题，我们就需要考虑使用多层神经元。更常见的是多层前馈神经网络，即每层神经元与下一层全互联，不存在同层连接，也不存在跨层连接。
输入层获取外界输入，隐层与输出层进行加工并最终由输出层输出。

神经网络的学习过程，就是根据训练数据来调整神经元之间连接权(connection weight)以及神经元的阈值。

- **Code** [Sklearn](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Perceptron.html)
```
from sklearn.linear_model import Perceptron
clf = Perceptron()
clf.fit(X, y)
```

## 3 误差逆传播算法
多层网络的学习能力比单层感知机强得多，同时也需要使用更强大的学习算法。反向误差传播(error BackPropagation，BP)是其中的杰出代表。

![Image of BP](https://github.com/songchangyi/MachineLearningResume/blob/master/img/BP.PNG)

由于反向传播涉及的公式推导实在比较多，并且网上也有太多的文章对之进行阐述，这里就不一一列举了。请移步[一文弄懂神经网络中的反向传播法——BackPropagation](https://www.cnblogs.com/charlotte77/p/5629865.html)

标准BP算法每次更新只针对单个样例，参数更新非常频繁。因此，我们也可以使用累积误差最小化(accumulated error backpropagation)。它在读取整个训练集一遍后，才对参数更新。但往往在累积误差下降到一定程度后，进一步下降会非常缓慢。这时标准BP往往会更好。两者的差别类似于随机梯度下降和标准梯度下降的区别。

已有证明显示，只需一个包含足够多神经元的隐层，多层前馈网络就能以任意精度逼近任意复杂度的连续函数。实践中一般用试错法来确定个数。

正是由于其强大的表示能力，NN时常有过拟合的问题。这里介绍两个改善策略：

1. 早停。将数据分为训练集和验证集，当验证集的误差开始升高的时候停止训练。
2. 正则化。即在误差目标函数中增加一个部分用于描述网络复杂度，例如权重与阈值的平方和。这样训练出的权重和阈值更加平衡。

- **Code** [Sklearn](https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html)
```
from sklearn.neural_network import MLPClassifier
clf = MLPClassifier(hidden_layer_sizes=(100,100,100), max_iter=500, alpha=0.0001,
                     solver='sgd', verbose=10,  random_state=21,tol=0.000000001)
clf.fit(X, y)
```

## 4 全局最小与局部极小
神经网络在训练集上的误差E是关于连接权w和阈值theta的函数。那么训练的过程可以看作一个寻找最优参数的过程。即在参数空间中，寻找一组最优参数使得误差最小。

这里的最优可以是局部极小(local minimum)和全局最小(global minimum)。
简单来说，局部极小就是该点的邻域误差均不小于该点。全局最小就是参数空间中所有点的误差都不小于该点。
可能存在多个局部极小，但是全局最小只有一个。

梯度下降法就是每一步计算误差函数在当前点的梯度，然后沿着梯度下降最快的方向寻找最优解。当梯度等于0时，达到局部极小。显然，局部极小不一定是全局最小。
所以我们试图跳出这个局部极小：

- 以多组不同参数初始化多个网络，取其中误差最小者。背后的思想是从不同的初始点开始，陷入同一个局部极小的可能性较低。
- 模拟退火。以一定概率接受比当前解更差的结果。有助于跳出局部极小，但也有可能跳出全局最小。
- 随机梯度下降(Stochastic gradient descent, SGD)。每次使用单个样本或者小批量样本(mini-batch)，于是求得的梯度也就带有了随机性。

- **Code** [Keras](https://keras.io/optimizers/)
```
from keras import optimizers
sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='mean_squared_error', optimizer=sgd)
```

## 面试问题
