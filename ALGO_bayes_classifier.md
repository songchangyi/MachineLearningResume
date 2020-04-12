# 贝叶斯分类器 (Bayes Classifier)
以下内容基于周志华老师的机器学习第7章归纳整理。

## 1 贝叶斯决策论(Bayesian decision theory)
**贝叶斯判定准则(Bayes decision rule)**：为最小化整体风险，只需在每个样本上选择那个能是条件风险最小的类别标记，即：

![h^{*}(x)=argminR(c|x)](https://render.githubusercontent.com/render/math?math=h%5E%7B*%7D(x)%3DargminR(c%7Cx))

h*称为贝叶斯最优分类器，与之对应的总体风险R(h*)称为贝叶斯风险。1-R(h*)反映了分类器所能达到的最好性能。
注意到R=1-P，那么：

![h^{*}(x)=argmaxP(c|x)](https://render.githubusercontent.com/render/math?math=h%5E%7B*%7D(x)%3DargmaxP(c%7Cx))

所以关键是获得后验概率P(c|x)，由贝叶斯定理有：

![P(c|x)=\frac{P(c)P(x|c)}{P(x)} ](https://render.githubusercontent.com/render/math?math=P(c%7Cx)%3D%5Cfrac%7BP(c)P(x%7Cc)%7D%7BP(x)%7D%20)

## 2 朴素贝叶斯分类器(Naive Bayes Classifier)
基于贝叶斯公式来估计后验概率的主要困难在于，P(x|c)是所有属性上的联合概率，难以估计。因此，朴素贝叶斯分类器采用了属性条件独立性假设。
即每个属性独立地对分类结果产生影响。我们的公式可以重写为：

![P(c|x)=\frac{P(c)}{P(x)} \prod_{i=1}^d P(x_i|c) ](https://render.githubusercontent.com/render/math?math=P(c%7Cx)%3D%5Cfrac%7BP(c)%7D%7BP(x)%7D%20%5Cprod_%7Bi%3D1%7D%5Ed%20P(x_i%7Cc)%20)

由于对所有类别来说P(x)相同，因此：

![h_{nb}(x)=argmaxP(c)\prod_{i=1}^d P(x_i|c) ](https://render.githubusercontent.com/render/math?math=h_%7Bnb%7D(x)%3DargmaxP(c)%5Cprod_%7Bi%3D1%7D%5Ed%20P(x_i%7Cc)%20)

对于第c类和第i个属性上取值为xi，离散属性用频率表概率。连续属性假定服从正太分布，使用概率密度函数求解。

举个简单的例子，垃圾邮件分类。
我们现在需要判断一封含有常年代开发票的邮件是不是垃圾邮件，就比较垃圾邮件中，
出现常年的概率乘以出现代开的概率再乘以发票的概率，除上普通邮件中这些词出现概率的乘积。
当然，本来每组还应该有个分母，就是同时出现常年，代开和发票3个词的概率。但由于两边都有就抵消了。即比较：
- P(垃圾)*P(常年|垃圾)*P(代开|垃圾)*P(发票|垃圾)
- P(普通)*P(常年|普通)*P(代开|普通)*P(发票|普通)

依次带入各组概率。如果垃圾组的乘积是普通组的100倍，就说明大概率是垃圾邮件。如果觉得不够详细可以戳[CSDN](https://blog.csdn.net/saltriver/article/details/72571876)

实践中为了避免连乘结果趋近于0，可以取对数。这样就转化为了加法。

还有一个问题是，对于新样本。因为没有出现过，所以算出来的概率就为1。这里就需要使用一个叫拉普拉斯平滑的技巧：

![\widehat{P} (x_i|c)=\frac{|D_{c,x_i}|+1}{|D_c|+N_i} ](https://render.githubusercontent.com/render/math?math=%5Cwidehat%7BP%7D%20(x_i%7Cc)%3D%5Cfrac%7B%7CD_%7Bc%2Cx_i%7D%7C%2B1%7D%7B%7CD_c%7C%2BN_i%7D%20)

这样就可以修正因样本不充分导致的估计概率为零的问题，随着训练集变大，估值会逐渐趋向实际概率值。

实际使用中，如果对预测速度要求高，就先计算所有概率估值并存储。
若数据更替频繁，可使用懒惰学习(lazy learning)方式，收到预测请求时再估值。
若数据不断增加，则仅对新样本相关的概率估值进行计数修正即可实现增量学习。

如果觉得意犹未尽，看这里[Naive Bayes * 垃圾邮件分类](http://www.iequa.com/2017/08/10/ml/naive-bayes-1/)

- **Code** [Sklearn](https://scikit-learn.org/stable/modules/naive_bayes.html)
```
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
y_pred = gnb.fit(X_train, y_train).predict(X_test)
```

这里有一个不依赖调包实现垃圾邮件分类的例子：[朴素贝叶斯 | 垃圾邮件识别](https://www.jianshu.com/p/b41d2e1f2a3a)

## 3 半朴素贝叶斯分类器(Semi-naive Bayes Classifiers)
现实中，属性的条件独立性假设难以成立。于是人们尝试适当的放松条件。
即考虑一部分属性间的相互依赖信息，从而既不需要进行完全联合概率计算，又不至于忽略了强属性依赖关系。

常用的一种策略是独依赖估计(One-Dependent Estimator, ODE)。即假设每个属性在类别之外最多仅依赖于一个属性。
那么关键就在于如何为每个类确定父属性。比较典型的实现有SPODE，TAN和AODE。这里细节不予阐述。

## 4 贝叶斯网(Bayesian Network)
贝叶斯网是一种经典的概率图模型，它借助有向无环图(Directed Acyclic Graph, DAG)来刻画属性之间的依赖关系，
并使用条件概率表(Conditional Probability Table, CPT)来描述属性的联合概率分布。

由于笔者目前对概率图模型也不是太熟练，只能先贴一点资源，日后再补。
- 理论：[概率图模型之贝叶斯网络](https://zhuanlan.zhihu.com/p/30139208)
- 实战：[机器学习sklearn之贝叶斯网络实战（一）](https://blog.csdn.net/weixin_41599977/article/details/90320390)

## 5 最大期望算法(Expectation Maximization Algorithm, EM)
之前的讨论中，我们一直假设所有属性变量的值都已被观测到。如果有未观测到的变量(隐变量，latent variable)，是否仍能对模型参数进行估计呢。

我们需要用到EM算法。它使用两个步骤交替计算：
1. 期望E步，利用当前估计的参数值来计算对数似然的期望值
2. 最大化M步，寻找能使E步产生似然期望最大化的参数值

进一步理解戳[简书](https://www.jianshu.com/p/1121509ac1dc)

## 面试问题
1. 贝叶斯学派和频率学派的方法有什么区别 What are the differences between “Bayesian” and “Freqentist” approach for Machine Learning?
2. 对比最大似然和最大后验估计 Compare and contrast maximum likelihood and maximum a posteriori estimation.
3. 贝叶斯方法如何自动筛选特征 How does Bayesian methods do automatic feature selection?
4. 贝叶斯正则化是什么意思 What do you mean by Bayesian regularization?
5. 什么时候你会用贝叶斯方法而不是频率方法 When will you use Bayesian methods instead of Frequentist methods?

## References
- [垃圾邮件是如何用贝叶斯方法过滤掉的](https://blog.csdn.net/saltriver/article/details/72571876)
- [Naive Bayes * 垃圾邮件分类](http://www.iequa.com/2017/08/10/ml/naive-bayes-1/)
- [朴素贝叶斯 | 垃圾邮件识别](https://www.jianshu.com/p/b41d2e1f2a3a)
- [概率图模型之贝叶斯网络](https://zhuanlan.zhihu.com/p/30139208)
- [机器学习sklearn之贝叶斯网络实战（一）](https://blog.csdn.net/weixin_41599977/article/details/90320390)
- [如何感性地理解EM算法](https://www.jianshu.com/p/1121509ac1dc)
