# 决策数 (Decision Tree)
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

## 3 剪枝处理
