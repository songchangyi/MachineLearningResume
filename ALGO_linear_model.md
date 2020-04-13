# 线性模型 (Linear model)
以下内容基于周志华老师的机器学习第3章归纳整理。

## 1 基本形式
线性模型试图学得一个通过属性的线性组合来进行预测的函数，即：

![f(x)=w^{T}+b](https://render.githubusercontent.com/render/math?math=f(x)%3Dw%5E%7BT%7D%2Bb)

其中w和b可以通过学习确定。

- **Pros** ：
  1. 形式简单，易于建模
  2. 可解释性好
  
## 2 线性回归(Linear Regression)

### 单变量
我们从最简单的情形开始考虑，即只有一个变量(也叫属性，variable)：

![f(x)=wx_i+b](https://render.githubusercontent.com/render/math?math=f(x)%3Dwx_i%2Bb)

现在我们需要确定w和b的值，我们希望的w和b应该能让f(x)和y的差别尽量小。同时，这里我们要解决的是回归任务，那么可让均方误差最小化：

![( w^{*}, b^{*})=argmin\sum_{i=1}^m {(y_i-wx_i-b)^2} ](https://render.githubusercontent.com/render/math?math=(%20w%5E%7B*%7D%2C%20b%5E%7B*%7D)%3Dargmin%5Csum_%7Bi%3D1%7D%5Em%20%7B(y_i-wx_i-b)%5E2%7D%20)

由于均方误差对应了常用的欧式距离，于是我们可以使用最小二乘法(least square method)求解。该方法的原理是试图找到一条直线，使所有样本到直线上的欧式距离之和最小。分别对w和b求导，并令其等于零可得：

![w=\frac{\sum_{i=1}^m y_i(x_i-\overline{x})}{ \sum_{i=1}^m x_i^2-\frac{1}{m}(\sum_{i=1}^m x_i)^2} ](https://render.githubusercontent.com/render/math?math=w%3D%5Cfrac%7B%5Csum_%7Bi%3D1%7D%5Em%20y_i(x_i-%5Coverline%7Bx%7D)%7D%7B%20%5Csum_%7Bi%3D1%7D%5Em%20x_i%5E2-%5Cfrac%7B1%7D%7Bm%7D(%5Csum_%7Bi%3D1%7D%5Em%20x_i)%5E2%7D%20)

![b=\frac{1}{m} \sum_{i=1}^m (y_i-wx_i) ](https://render.githubusercontent.com/render/math?math=b%3D%5Cfrac%7B1%7D%7Bm%7D%20%5Csum_%7Bi%3D1%7D%5Em%20(y_i-wx_i)%20)

### 多变量
更一般的情形是多个属性，即多元线性回归。这里涉及的数学推导会比较多，按道理来讲只有偏research方向的岗位才会在求职的时候考察手动推导，感兴趣的小伙伴可以移步以下链接：

- [多元线性回归推导过程](https://blog.csdn.net/weixin_39445556/article/details/81416133)
- [多元线性回归求解过程 解析解求解](https://blog.csdn.net/weixin_39445556/article/details/83543945)

线性模型虽然简单，但变化却很丰富。例如：
- 为了实现了非线性函数的映射，我们加入对数函数得到**对数线性回归**：
![lny=w^{T}+b](https://render.githubusercontent.com/render/math?math=lny%3Dw%5E%7BT%7D%2Bb)

- 更一般的，我们考虑单调可微函数g，来实现**广义线性回归**：
![y=g^{-1}(w^{T}+b)](https://render.githubusercontent.com/render/math?math=y%3Dg%5E%7B-1%7D(w%5E%7BT%7D%2Bb))

- **Code** [Sklearn](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html)
```
from sklearn.linear_model import LinearRegression
reg = LinearRegression().fit(X, y)
reg.coef_ # get w
reg.intercept_ # get b
reg.predict(X_test) # prediction
```

## 3 逻辑回归（Logistic Regression，也叫逻辑斯蒂回归）
接下来我们看分类问题。神奇的线性回归模型产生的预测值是实数值，理论上可以从负无穷到正无穷。所以我们需要一个函数，来将实数值映射到0和1的范围内，然后再判断输出为0还是1。并且我们希望该函数单调可微，便于求解。这里我们的sigmoid函数（中文的对数几率函数或者S型函数）就出场了：

![y=\frac{1}{1+e^{-z}} ](https://render.githubusercontent.com/render/math?math=y%3D%5Cfrac%7B1%7D%7B1%2Be%5E%7B-z%7D%7D%20)

从下图可以看出，它完美的符合我们的预期：

![Image of sigmoid](https://github.com/songchangyi/MachineLearningResume/blob/master/img/sigmoid.PNG)

这里的z其实就是线性回归的输出。带入线性回归的公式得：

![y=\frac{1}{1+e^{-(w^{T}x+b)}}](https://render.githubusercontent.com/render/math?math=y%3D%5Cfrac%7B1%7D%7B1%2Be%5E%7B-(w%5E%7BT%7Dx%2Bb)%7D%7D)

等价于：

![ln\frac{y}{1-y}=w^{T}x+b](https://render.githubusercontent.com/render/math?math=ln%5Cfrac%7By%7D%7B1-y%7D%3Dw%5E%7BT%7Dx%2Bb)

将y视为正例可能性，则y/(1-y)即为正例和负例可能性的比值，也就几率。因此等式左边得名对数几率(logit)。

- **Pros**
  1. 直接对分类可能性进行建模，无需假设数据分布
  2. 不仅预测出类别，还得到近似概率预测
  3. 对数几率函数是任意阶可导的凸函数，方便求取最优解。比如利用[梯度下降法](https://blog.csdn.net/ligang_csdn/article/details/53838743)和[牛顿法](https://blog.csdn.net/Fishmemory/article/details/51603836)

- **Code** [Sklearn](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)
```
from sklearn.linear_model import LogisticRegression
clf = LogisticRegression(random_state=0).fit(X_train, y_train)
clf.predict(X_test)
clf.predict_proba(X_test)
```

To use cross validation with LogisticRegression : [Sklearn](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegressionCV.html)

## 4 多分类学习
对于多分类问题，我们的基本思路是拆解法。即将多分类任务拆为若干个二分类任务。比较经典的拆分策略有三种：

1. 一对一（One vs. One，OvO）

将N个类两两配对，产生N(N-1)/2个二分类任务。在测试阶段，新样本将同时提交给所有分类器，我们得到N(N-1)/2个分类结果。然后我们选取预测的最多的类别最为最终结果。

2. 一对其余（One vs. Rest，OvR）

OvR则是每次将一个类的样例作为正例，其余作为反例得到N个分类器。如果测试的时候仅有一个预测为正，则对应的为最终结果。否则选择置信度最大的类别。

**PS**：OvO的存储开销和测试时间开销通常比OvR大，但由于OvO训练时每次只使用两个类的样例。因此类别多时OvO的时间开销反而小。一般情况下两者性能相近。

在sklearn的LogisticRegression中，使用multi_class='ovr'来实现。

3. 多对多（Many vs. Many，MvM）

MvM是每次多个类做为正例，剩下的作为反例。其构造必须有特殊设计，例如使用[纠错输出码](http://shichaoxin.com/2019/12/05/%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0%E5%9F%BA%E7%A1%80-%E7%AC%AC%E4%B9%9D%E8%AF%BE-%E5%A4%9A%E5%88%86%E7%B1%BB%E5%AD%A6%E4%B9%A0/)

## 5 类别不平衡问题
前面介绍的方法都共同假设了不同类别的训练样例数目相当。在极端情况下，我们训练出的分类器会倾向于预测样例占大多数的那一类。比如我们之前提到的肿瘤检测将所有人都预测为良性。

我们假设负例数目远大于正例数目。现有技术大体有三类做法：

1. 欠采样(undersampling)：

去除一些负例使得剩下的正负例数目相当。因为丢弃了一部分数据，时间开销较小。实际应用在集成学习中，会将反例划分为不同的集合，这样就不会丢失信息。

2. 过采样(oversampling)：

增加一些正例。常见的算法有SMOTE

- **Code** [Imbalanced Learn](https://imbalanced-learn.readthedocs.io/en/stable/over_sampling.html)
```
from imblearn.over_sampling import SMOTE
X_resampled, y_resampled = SMOTE().fit_resample(X, y)
```

3. 阈值移动(threshold-moving)：

在决策过程中考虑到样本比例。例如xgboost算法的如下参数：
```
scale_pos_weight = count(negative examples)/count(Positive examples)
```

## 面试问题
- 解释线性回归 What’s linear regression
- 为什么线性模型中需要假设残差服从正态分布 why linear model needs to assume the residual is normally distributed
- 如何解释变量参数 How to explain coefficients
- 变量的参数是无偏估计量吗 Are coefficients unbiased estimators
- 如何进行特征选择 How to select features
- 如何得到最优参数 How to get optimal parameters (gradient descent)
- 如何处理共线性 How to handle collinearity? What’s the effect?
- 当两个变量完全线性相关如何处理 What if two variables are perfectly correlated
- 什么是正则化，有什么作用 What is regularization? What’s the impact of regularization?
- 岭回归和Lasso回归的异同 Ridge vs Lasso?
- 线性回归于逻辑回归的异同 Linear regression vs logistic regression
- 逻辑回归的代价函数 Cost function of logistic regression
- 讲述一下梯度下降法 Gradient Descent

## References
- [机器学习--Logistic回归计算过程的推导](https://blog.csdn.net/ligang_csdn/article/details/53838743)
- [逻辑回归（Logistic Regression）-牛顿法求解参数](https://blog.csdn.net/Fishmemory/article/details/51603836)
- [【机器学习基础】第九课：多分类学习](http://shichaoxin.com/2019/12/05/%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0%E5%9F%BA%E7%A1%80-%E7%AC%AC%E4%B9%9D%E8%AF%BE-%E5%A4%9A%E5%88%86%E7%B1%BB%E5%AD%A6%E4%B9%A0/)
