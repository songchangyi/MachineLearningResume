# 决策树 (Decision Tree)
以下内容基于周志华老师的机器学习第4章归纳整理。

## 1 基本流程
简单的来说，决策树遵循分而治之的思想。在树的每一个节点都会对应一个属性进行划分，直到遍历所有属性。最后的结果会被划分到某一个叶结点，即一个类中。

## 2 划分选择
决策树学习的关键在于选择最优划分属性。这样随着划分的进行，结点的纯度(purity)越来越高。

### 2.1 信息增益
信息熵是度量纯度最常用的指标。信息熵越大，则越混乱，也就是纯度低。笔者有个比较方便记忆的例子，想必读者都知道熵增加原理里面一个著名的例子吧。就是墨水滴入一盆水中会扩散开来，不会聚成一滴。那就是说，熵增加=混乱=纯度降低。反之，熵越小，纯度就越高。我们的决策树的目标就是要分出尽可能纯的类，整个过程就是在减熵。如果有人对热力学的熵，信息学的熵，信息，信息量有更多的疑问，请戳[知乎](https://www.zhihu.com/question/274997106)。

**信息熵** 

![Ent(D)=-\sum_{k=1}^{|y|} p_klog_2 p_k ](https://render.githubusercontent.com/render/math?math=Ent(D)%3D-%5Csum_%7Bk%3D1%7D%5E%7B%7Cy%7C%7D%20p_klog_2%20p_k%20)

这里我们要再定义一个概念，信息增益。著名的ID3决策树算法就是用信息增益来选择划分属性。

**信息增益** 

![Gain(D, a)=Ent(D)-\sum_{v=1}^{V}\frac{|D^{v}|}{|D|} Ent(D^{v})](https://render.githubusercontent.com/render/math?math=Gain(D%2C%20a)%3DEnt(D)-%5Csum_%7Bv%3D1%7D%5E%7BV%7D%5Cfrac%7B%7CD%5E%7Bv%7D%7C%7D%7B%7CD%7C%7D%20Ent(D%5E%7Bv%7D))

我们规定信息增益为划分前的熵Ent(D)减去划分后的熵Ent(D')。我们的目标是划分后熵越小越好，也就是信息增益越大越好。举个具体的例子，假设根节点中有正例8，负例9，则可算出初始信息熵：

![Ent(D)=-\sum_{k=1}^{2} p_klog_2 p_k=-(\frac{8}{17}log_2 \frac{8}{17}+\frac{9}{17}log_2 \frac{9}{17})=0.998](https://render.githubusercontent.com/render/math?math=Ent(D)%3D-%5Csum_%7Bk%3D1%7D%5E%7B2%7D%20p_klog_2%20p_k%3D-(%5Cfrac%7B8%7D%7B17%7Dlog_2%20%5Cfrac%7B8%7D%7B17%7D%2B%5Cfrac%7B9%7D%7B17%7Dlog_2%20%5Cfrac%7B9%7D%7B17%7D)%3D0.998)

现在根据某属性进行划分后，得到3个子集。子集1中包含3正3反，子集2中包含4正2反，子集3中包含1正4反。那么现在：

![Ent(D^1)=-(\frac{3}{6}log_2 \frac{3}{6}+\frac{3}{6}log_2 \frac{3}{6})=1.000](https://render.githubusercontent.com/render/math?math=Ent(D%5E1)%3D-(%5Cfrac%7B3%7D%7B6%7Dlog_2%20%5Cfrac%7B3%7D%7B6%7D%2B%5Cfrac%7B3%7D%7B6%7Dlog_2%20%5Cfrac%7B3%7D%7B6%7D)%3D1.000)

![Ent(D^2)=-(\frac{4}{6}log_2 \frac{4}{6}+\frac{2}{6}log_2 \frac{2}{6})=0.918](https://render.githubusercontent.com/render/math?math=Ent(D%5E2)%3D-(%5Cfrac%7B4%7D%7B6%7Dlog_2%20%5Cfrac%7B4%7D%7B6%7D%2B%5Cfrac%7B2%7D%7B6%7Dlog_2%20%5Cfrac%7B2%7D%7B6%7D)%3D0.918)

![Ent(D^3)=-(\frac{1}{5}log_2 \frac{1}{5}+\frac{4}{5}log_2 \frac{4}{5})=0.722](https://render.githubusercontent.com/render/math?math=Ent(D%5E3)%3D-(%5Cfrac%7B1%7D%7B5%7Dlog_2%20%5Cfrac%7B1%7D%7B5%7D%2B%5Cfrac%7B4%7D%7B5%7Dlog_2%20%5Cfrac%7B4%7D%7B5%7D)%3D0.722)

从而信息增益为：

![Gain(D, V)=0.998-(\frac{6}{17}*1+\frac{6}{17}*0.918+\frac{5}{17}*0.722)=0.109](https://render.githubusercontent.com/render/math?math=Gain(D%2C%20V)%3D0.998-(%5Cfrac%7B6%7D%7B17%7D*1%2B%5Cfrac%7B6%7D%7B17%7D*0.918%2B%5Cfrac%7B5%7D%7B17%7D*0.722)%3D0.109)

依次类推，我们可以算出所有属性的信息增益。简单的说就是在每一步，把剩余的属性都试一遍，然后选取信息增益最大那个。

### 2.2 增益率
细心的读者可能发现了上述方法有个漏洞。假如我们有一个变量，叫做编号，从1-N。那么该变量一次就划分完毕，并直接将熵减小到0，每个节点都是最纯的。显然这不是我们想要的。为了惩罚这种不利影响，我们引入信息增益率(gain rate)的概念。著名的C4.5决策树就使用了这一点。

**信息增益率**

![Gain_ratio(D, a)= \frac{Gain(D, a)}{IV(a)} ](https://render.githubusercontent.com/render/math?math=Gain_ratio(D%2C%20a)%3D%20%5Cfrac%7BGain(D%2C%20a)%7D%7BIV(a)%7D%20)

其中

![IV(a)=-\sum_{v=1}^{V}\frac{|D^{v}|}{|D|}log_2 \frac{|D^{v}|}{|D|}](https://render.githubusercontent.com/render/math?math=IV(a)%3D-%5Csum_%7Bv%3D1%7D%5E%7BV%7D%5Cfrac%7B%7CD%5E%7Bv%7D%7C%7D%7B%7CD%7C%7Dlog_2%20%5Cfrac%7B%7CD%5E%7Bv%7D%7C%7D%7B%7CD%7C%7D)

称为属性a的固有值(intrinsic value)。当划分的类越多，V越大，IV(a)就越大。

值得注意的是，增益率偏好可取值数目较少的属性。所以C4.5中，先选择增益高于平均水平的属性，再从中选取增益率最高的。

### 2.3 基尼系数
卡特(CART)决策树采用不同于熵的一个指数，基尼系数(Gini index)来选择划分属性。

**基尼值**

![Gini(D)=1-\sum_{k=1}^{|y|} p_k^2 ](https://render.githubusercontent.com/render/math?math=Gini(D)%3D1-%5Csum_%7Bk%3D1%7D%5E%7B%7Cy%7C%7D%20p_k%5E2%20)

**基尼指数**

![Gini\_index(D,a)=\sum_{v=1}^{|V|}\frac{|D^v|}{|D|}Gini(D^v)](https://render.githubusercontent.com/render/math?math=Gini%5C_index(D%2Ca)%3D%5Csum_%7Bv%3D1%7D%5E%7B%7CV%7C%7D%5Cfrac%7B%7CD%5Ev%7C%7D%7B%7CD%7C%7DGini(D%5Ev))

简单来说，基尼指数就是熵的近似替代品。使用它是因为求平方和比熵的指数运算更简便。请看下图：

![Image of Gini index](https://github.com/songchangyi/MachineLearningResume/blob/master/img/gini.jpg)

更多关于CART树的细节请参考：[决策树算法原理(下)](https://www.cnblogs.com/pinard/p/6053344.html)

- **Code** [Sklearn](https://scikit-learn.org/stable/modules/tree.html#tree)
```
from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier()
clf = clf.fit(X, Y)
```

## 3 剪枝处理
剪枝(pruning)的主要作用是防止决策树的过拟合。基本策略有预剪枝(prepruning)和后剪枝(post-pruning)。

### 3.1 预剪枝
预剪枝是在决策树生成过程中，若当前划分不能带来有效的性能提升，则放弃划分并将当前节点标记为叶结点。这里我们可以使用验证集来进行评估，如果划分后验证集的精度提升了，那么就是值得划分。否则划分将被禁止。

- **Pros**：
  - 降低过拟合风险
  - 减少训练和测试的时间及开销
- **Cons**：
  - 过早的剪枝可能会欠拟合

### 3.2 后剪枝
后剪枝是先生成一颗完整的决策树，然后自底向上的将一些不必要的子树替换为叶结点。同样的，我们使用验证集，考察将某子树替换成叶结点前后的精度变化。

- **Pros**：
  - 欠拟合风险较小，泛化性能往往优于预剪枝决策树
- **Cons**：
  - 需要逐一考察树中的非叶结点，训练时间开销比较大

在实际代码中，以上思想对应的诸如树深(max_depth)、叶节点最少数目(min_samples_leaf)、最少划分样本数量(min_samples_split)等参数。

## 4 连续与缺失值

### 4.1 连续值的处理
目前为止，我们仅讨论了离散属性。如果我们遇到连续属性，就需要考虑将其离散化。

最简单的是二分法(bi-partition)，该方法被C4.5决策树算法所使用。

举个例子，我们考虑一个连续属性对应的值：0.1，0.5，0.6，1.5，2.5。我们的目标是寻找一个最佳的划分点将他们分开。显而易见，对于0.1和0.5，当我们取他们之间任意的一个值来划分(0.3, 0.4, 0.499...)的效果都是一样的。于是我们干脆就取中间值，这就是二分的来源。对于上述5个值的列表，我们就能定义出4个划分点：0.3，0.55，1.05，2.0。

下一步就是挨个计算每个划分点对应的信息增益，选取最大的即可。

**注意**，与离散属性不同，如果当前结点划分属性为连续属性，该属性还可作为其后代结点的划分属性。意思就是假设这个结点按照是否小于1.05来划分，之后的子结点仍可以再按照是否小于0.8来进一步细分，仍然是基于同一属性。而连续变量，比如黑白，当我们第一次划分开黑和白以后，就不会再次划分黑白了。

### 4.2 缺失值处理
实际问题中，完整的数据是极其罕见的。面对有缺失的数据，我们需要解决两个问题：

1. 如何在有缺失值的情况下进行属性划分？
2. 给点划分属性，如果样本在该属性上缺失了，怎么办？

对于问题1，答案是我们只考虑无缺失值样本所占的比例。也就是在划分一个属性时，不考虑在该属性上有缺失的数据。

对于问题2，则是按一定的权重将样本划入所有子节点。比如说在某个结点，通常60%的样例会进入左子树，40%的样例会进入右子树。那么对一个没有该属性的样例，我们将其同时划分到两个子树中，不过左子树的权重为0.6，右子树为0.4。

### 5 多变量决策树
读者可能注意到了，根据我们之前说的属性划分，在数据量大或者属性量大的时候，会造成大量的计算开销。我们可以利用多变量决策树来简化这一点。

例如，多变量决策树中的一个结点可能是这样的：-0.8 * 密度 - 0.044 * 含糖率 <= -0.313

这样我们可以大大的减少时间复杂度。

## 面试问题
- 如何处理过拟合问题 how to control overfitting?
- 解释一下基尼系数 What is Gini index?

## References
- [决策树算法原理(下)](https://www.cnblogs.com/pinard/p/6053344.html)
