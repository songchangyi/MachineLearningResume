# Evaluation Methods
以下内容基于周志华老师的机器学习2.2节归纳整理。

## 基本理论
为了对学习器的泛化误差进行评估并进而做出选择，我们需要一个测试集(testing set)来测试模型对新样本的判别能力。然后以测试误差(testing error)作为泛化误差的近似。**要点**：测试样本是从样本真实分布中独立同分布(Independent and identically distributed, IID)采样得到，并与训练集互斥。

### 1 留出法(hold-out)
直接将数据集划分为两个互斥的集合。使用分层采样(stratified sampling)。

- **Cons**
> 单次估计结果不够稳定可靠。解决方法：若干次随机划分，取平均值。
>
> 测试集较小，降低了保真性(fidelity)。解决方法：无完美解决方案，常见做法是将2/3-4/5的样本用于训练，剩下用于测试。

- **Code** [Sklearn](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html)
```
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
```

### 2 交叉验证法(K-fold cross validation)
将数据集划分为k个大小相似的互斥子集，尽可能保持数据分布的一致性。每次用k-1个子集训练，剩下的那个测试。最终返回这k个测试结果的均值。结果的稳定性和保真性在很大程度上取决于k的取值，最常用的k值为10。

为了减小因样本划分不同而引入的差别，通常要重复p次k折交叉验证。则最终结果为p次k折的均值。常见的有10次10折CV。

- **Code** [Sklearn](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.KFold.html)
```
from sklearn.model_selection import KFold
kf = KFold(n_splits=10)
for train_index, test_index in kf.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
```

以上的操作只是提供了一种将数据划分开来的方法，但是实际上我们更常用的是以下两个函数：cross_val_score和cross_val_predict。 [知乎](https://zhuanlan.zhihu.com/p/37787407)

1. **cross_val_score** [Sklearn](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_val_score.html)
```
from sklearn import datasets, linear_model
from sklearn.model_selection import cross_val_score
lasso = linear_model.Lasso()
print(cross_val_score(lasso, X, y, cv=3))
```

2. **cross_val_predict** [Sklearn](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_val_predict.html#sklearn.model_selection.cross_val_predict)
```
from sklearn import datasets, linear_model
from sklearn.model_selection import cross_val_score
lasso = linear_model.Lasso()
y_pred = cross_val_predict(lasso, X, y, cv=3)
```

如果我们希望返回CV的模型，可以使用：

3. **cross_validate** [Sklearn](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_validate.html)
```
from sklearn import datasets, linear_model
from sklearn.model_selection import cross_val_score
lasso = linear_model.Lasso()
scores = cross_validate(lasso, X, y, cv=3, 
                        scoring=('r2', 'neg_mean_squared_error'), 
                        return_estimator=True)
```

**留一法(Leave One Out, LOO)**
特别的，如有m个样本，令k=m，此时为留一法。

- **Pros**
> 该方法不受划分方式的影响。
>
> 与真实模型很相似，从而结果比较精确。

- **Cons**
> 数据量大时开销难以忍受
>
> 未必永远比其他方法精确。“没有免费的午餐”

- **Code** [Sklearn](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.LeaveOneOut.html)
```
from sklearn.model_selection import LeaveOneOut
loo = LeaveOneOut()
for train_index, test_index in loo.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
```

### 3 自助法(bootstrapping)
以自助采样法(bootstrap sampling)为基础的有放回的随机采样。样本在m次采样中始终不被采到的概率为：
<img src="https://render.githubusercontent.com/render/math?math=(1 - \frac {1}{m})^m">

取极限：
<img src="https://render.githubusercontent.com/render/math?math=\lim_{m \to \infty}{(1 - \frac {1}{m})^m}=0.368">

于是仍然有约1/3的数据没在训练集中出现，可用作测试集。称为**包外估计(out-of-bag estimate)**。

- **Pros**
> 对小数据集很有用
>
> 能从初始数据集中产生多个不同训练集，对集成学习有很大好处

- **Cons**
> 改变了初始数据集分布，会引入估计偏差(bias)。不太适合数据量足够的情况。

**Code I** [Sklearn](https://ogrisel.github.io/scikit-learn.org/sklearn-tutorial/modules/generated/sklearn.cross_validation.Bootstrap.html)
```
from sklearn import cross_validation
bs = cross_validation.Bootstrap(9, random_state=0)
for train_index, test_index in bs:
    print ("TRAIN:", train_index, "TEST:", test_index)
```

**Code II** [CSDN](https://blog.csdn.net/bqw18744018044/article/details/81024520)
```
train = data.sample(frac=1.0,replace=True)
test = data.loc[data.index.difference(train.index)].copy()
```

### 4 调参与最终模型
在模型评估和选择过程中，我们只使用了一部分数据用于训练。在确定学习算法和参数配置后，应使用整个数据集重新训练模型。
为了区分实际使用中的测试数据，我们把评估时的测试数据称为验证集(validation set)。

## 面试问题

## References
1. 怎么理解 P 问题和 NP 问题 
https://www.zhihu.com/question/27039635
2. 机器学习理论与Scikit-Learn对应操作
https://zhuanlan.zhihu.com/p/37787407
