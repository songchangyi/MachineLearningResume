# 集成学习 (Ensemble Learning)
以下内容基于周志华老师的机器学习第8章归纳整理。

## 1 个体与集成
集成学习就是先产生一组个体学习器，然后再用某种策略将它们结合起来。

同质集成中使用一样的基学习器，比如全是决策树。异质集成就是使用了不同的组件学习器，比如决策树和神经网络。

当学习器间存在强依赖关系、必须串行生成的序列方法是Boosting。当学习器间不存在强依赖关系、可同时生成的并行化方法是Bagging和随机森林(Random Forest)。

## 2 Boosting
Boosting的原理是，先从初始训练集训练出一个集学习器，再根据其表现对样本分布进行调整。
训练错的样本的权重会被加大以得到更多关注，然后训练下一个基学习器。如此重复，直到基学习器达到事先指定的值。

Boosting族算法最著名的代表是Adaboost（其实笔者感觉该算法已经过时了，现在主宰比赛和业界的基本是Xgboost和LightGBM，偶尔有GDBT和Catboost。笔者自己的经验是，跟后几个算法相比Adaboost几乎没有赢过）。

![Image of adaboost](https://github.com/songchangyi/MachineLearningResume/blob/master/img/adaboost.PNG)

大致流程是：
1. 初始化样本权值分布
2. 基于数据训练出分类器ht
3. 估计ht的误差
4. 确定ht的权重
5. 更新样本分布
6. 训练下一个分类器
7. 最后输出H(x)

这里![H(x)=sign(\sum_{t=1}^T  \alpha_t h_t(x))](https://render.githubusercontent.com/render/math?math=H(x)%3Dsign(%5Csum_%7Bt%3D1%7D%5ET%20%20%5Calpha_t%20h_t(x)))

这里符号函数sign跟其字面意思一样。为负的就是-1，为零的就是0，为正的就是1。

更多推导细节可以参考[AdaBoost算法详述](https://zhuanlan.zhihu.com/p/42915999)

刚刚说了，在每一轮训练中，我们会调整训练样本的权重。对于不接受带权样本的基学习算法，我们可以使用重采样来处理。两种做法一般没有显著差别。

**注意**，每一轮我们都会检查当前生成的基学习器是否满足基本条件。比如一个精度不到0.5的学习器，性能还不如使用随机猜测，就会丢弃掉，训练就会提前停止。
这时，如果是重采样，就可以重新启动训练。

从偏差——方差分解角度看，Boosting主要是降低偏差，关注提升学习能力。

- **Code** [Sklearn](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html)
```
from sklearn.ensemble import AdaBoostClassifier
clf = AdaBoostClassifier(n_estimators=100, random_state=0)
clf.fit(X, y)
clf.feature_importances_
clf.predict(X_test)
```

## 3 Bagging与随机森林
### 3.1 Bagging
Bagging是并行式集成学习方法最著名的代表。它基于自助采样法(bootstrap sampling)即重采样法。
经过m次随机采样操作，得到含m个样本的采样集，理论上初始训练集中约63.2%的样本会出现在其中。

这样我们可以采样出T个含m个样本的采样集。最每个采样集训练出一个基学习器再进行结合。如果是分类任务，就使用简单投票法，如果是回归任务，就使用简单平均法。

![Image of bagging](https://github.com/songchangyi/MachineLearningResume/blob/master/img/bagging.PNG)

- **Pros**
  - 由于投票/平均的复杂度很小，Bagging基础与其基学习器算法复杂度同阶，因此很高效
  - 训练集中未被使用的剩下约36.8%的样本可作为验证集来进行包外估计。从而减轻过拟合等。

从偏差——方差分解角度看，Boosting主要是降低方差，它在易受样本扰动的学习器上效果更明显。

- **Code** [Sklearn](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.BaggingClassifier.html)
```
from sklearn.ensemble import BaggingClassifier
clf = BaggingClassifier(base_estimator=SVC(), n_estimators=10, random_state=0).fit(X, y)
```

### 3.2 随机森林
随机森林是Bagging比较成功的一个变体。在以决策树为基学习器构建Bagging集成的基础上，进一步引入了随机属性选择。
比如说本来有d个属性，RF会随机选择其中k个进行训练。一般选取k=log2d。

- **Pros**
  - 简单，容易实现，计算开销小，性能强大。（几乎总是第二好的算法）
  - 属性、样本两层随机性，泛化性能进一步提升。几乎不存在过拟合问题。
  - 一般而言，随机森林训练效率优于Bagging

- **Code** [Sklearn](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)
```
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier()
clf.fit(X, y)
```

## 4 结合策略
### 4.1 平均法
对数值型输出，比如回归任务，通常使用平均法
- 简单平均法

![H(x)=\frac{1}{T} \sum_{i=1}^T h_i(x) ](https://render.githubusercontent.com/render/math?math=H(x)%3D%5Cfrac%7B1%7D%7BT%7D%20%5Csum_%7Bi%3D1%7D%5ET%20h_i(x)%20)

- **Code** [Sklearn](https://scikit-learn.org/stable/modules/ensemble.html#voting-regressor)
```
from sklearn.ensemble import VotingRegressor
ereg = VotingRegressor(estimators=[('gb', reg1), ('rf', reg2), ('lr', reg3)])
```

- 加权平均法。

![H(x)=\sum_{i=1}^T w_i h_i(x) ](https://render.githubusercontent.com/render/math?math=H(x)%3D%5Csum_%7Bi%3D1%7D%5ET%20w_i%20h_i(x)%20)

即性能好的学习器权重大。当然，权重之和为1。比较适合个体学习器性能相差较大的情况。

### 4.2 投票法
对分类任务一般采用投票法。

- 绝对多数投票法(majority voting)

采纳仅得票超过半数的预测结果。

- **Code** [Sklearn](https://scikit-learn.org/stable/modules/ensemble.html#voting-classifier)
```
from sklearn.ensemble import VotingClassifier
eclf = VotingClassifier(estimators=[('lr', clf1), ('rf', clf2), ('gnb', clf3)], voting='hard')
```

- 相对多数投票法(plurality voting)

采纳最高票数的预测结果。

- 加权投票法(weighted voting)

投票时加入权重。

- **Code** [Sklearn](https://scikit-learn.org/stable/modules/ensemble.html#voting-classifier)
```
from sklearn.ensemble import VotingClassifier
eclf = VotingClassifier(estimators=[('dt', clf1), ('knn', clf2), ('svc', clf3)], voting='soft', weights=[2, 1, 2])
```

### 4.3 学习法
当训练数据很多时，一种更为强大的结合策略是通过另一个学习器来结合。典型代表是Stacking。
这里个体学习器成为初级学习器，用于结合的学习器成为次级学习器或者元学习器(meta-learner)。

Stacking先从初始训练集中训练出初级学习器，然后生成一个新数据集用于训练次级学习器。
在这个新数据集中，初级学习器的输出被作为输入，初始样本的标记被当作样例标记。这里假设初级学习器是异质的：

![Image of stacking](https://github.com/songchangyi/MachineLearningResume/blob/master/img/stacking.PNG)

为了减少过拟合的风险，一般使用交叉验证或者留一法，用之前未使用的样本来产生次级学习器的训练样本。

有研究表明，多响应线性回归(Multi-response Linear Regression, MLR)作为次级学习算法效果较好。

- **Code** [Sklearn](https://scikit-learn.org/stable/modules/ensemble.html#stacked-generalization)
```
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import StackingRegressor
reg = StackingRegressor(estimators=estimators, final_estimator=GradientBoostingRegressor(random_state=42))
```

## 5 多样性
### 5.1 误差——分歧分解
个体学习器准确率越高，多样性越大则集成越好。

笔者联想到一个例子，足球赛中，一个强队应该是世界顶级的前锋加顶级的中场加顶级的后卫的阵容，而不会直接是10个顶级的前锋上场吧。

### 5.2 多样性度量
常见的一些多样性度量有不合度量(disagreement measure)，相关系数(correlation coefficient)，Q-统计量和Kappa统计量。

笔者自己比较喜欢的是相关系数。实践中把每个分类器的输出作为一列，构成一个pandas里的dataframe，然后由df.corr()就可以看出列与列之间的相关性。

### 5.3 多样性增强
- 数据样本扰动：基于采样法产生不同数据子集
- 输入属性扰动：基于不同属性子集。不过如果数据属性不多则效果不好。
- 输入表示扰动：改变标签。例如随机改变一些样本标记，或将分类任务变为回归任务，或将原任务拆解为多个子任务
- 算法参数扰动：使用不同的参数组合进行训练

## 面试问题
1. 决策树与随机森林的关系 Decision tree vs random forest
2. 为什么随机森林仅使用一部分数据和一部分属性 Why use random forest to subset of data, why take subset of features
3. Boosting和Bagging的区别 boosting vs bagging

## References
- [AdaBoost算法详述](https://zhuanlan.zhihu.com/p/42915999)
