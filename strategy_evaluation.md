# Evaluation

## 基本理论

### 1 评估方法
为了对学习器的泛化误差进行评估并进而做出选择，我们需要一个测试集(testing set)来测试模型对新样本的判别能力。然后以测试误差(testing error)作为泛化误差的近似。**要点**：测试样本是从样本真实分布中独立同分布(Independent and identically distributed, IID)采样得到，并与训练集互斥。

#### 1.1 留出法(hold-out)
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

#### 1.2 交叉验证法(K-fold cross validation)
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

**Code** [Sklearn](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.LeaveOneOut.html)
```
from sklearn.model_selection import LeaveOneOut
loo = LeaveOneOut()
for train_index, test_index in loo.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
```

#### 1.3 自助法(bootstrapping)
以自助采样法(bootstrap sampling)为基础的有放回的随机采样。样本在m次采样中始终不被采到的概率为：(1-1/m)^m
<img src="https://render.githubusercontent.com/render/math?math=(1 - \frac {1}{m})^m">

取极限：
<img src="https://render.githubusercontent.com/render/math?math=\lim_{m \to \infty}{(1 - \frac {1}{m})^m}=0.368">



## 面试问题

## References
1. 怎么理解 P 问题和 NP 问题 
https://www.zhihu.com/question/27039635
