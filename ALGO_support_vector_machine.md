# 支持向量机 (Support Vector Machine, SVM)
以下内容基于周志华老师的机器学习第6章归纳整理。

## 1 间隔与支持向量
分类学习最基本的想法就是基于训练集D在样本空间中找到一个划分超平面。直观的看，应该去找位于两类样本正中间的超平面。这样对训练数据局部扰动容忍性最好。

样本空间中，划分超平面可通过如下方程描述：

![w^{T}+b=0](https://render.githubusercontent.com/render/math?math=w%5E%7BT%7D%2Bb%3D0)

其中w为法向量，b为位移项。那么空间中任意点x到该平面的距离为：

![r=\frac{|w^{T}+b|}{||w||} ](https://render.githubusercontent.com/render/math?math=r%3D%5Cfrac%7B%7Cw%5E%7BT%7D%2Bb%7C%7D%7B%7C%7Cw%7C%7C%7D%20)

离平面最近的点到平面的距离称为支持向量。两个异类支持向量到超平面的距离之和为：

![r=\frac{2}{||w||} ](https://render.githubusercontent.com/render/math?math=r%3D%5Cfrac%7B2%7D%7B%7C%7Cw%7C%7C%7D%20)

称为间隔。

显然，为了最大化间隔，仅需最大化||w||^-1，等价于最小化||w||^2。

## 2 对偶问题
我们首先需要使用拉格朗日乘子法得到其对偶问题，再用SMO算法进行求解。

这里公式较多，详情请戳[CSDN](https://blog.csdn.net/u014433413/article/details/78427574)

## 3 核函数
前面的讨论中，我们假设训练样本是线性可分的。但有时候这个假设并不成立。此时，我们需要将样本从原始空间映射到一个更高维的特征空间。
例如，二维不可分的时候，放在三维也许可分。一般的，职业属性数目有限，一定存在一个高维特征空间使样本可分。

在特征空间中划分超平面对应的模型可表示为：

![f(x)=w^{T}\phi(x)+b](https://render.githubusercontent.com/render/math?math=f(x)%3Dw%5E%7BT%7D%5Cphi(x)%2Bb)

其中phi(x)是x映射后的特征向量。

在求解时，我们需要使用核函数。关于相关的细节和推导，请戳[掘金](https://juejin.im/post/5ad1c5f75188255cb07d8c33)。以下是几种常用的核函数：

![Image of svm kernel](https://github.com/songchangyi/MachineLearningResume/blob/master/img/svm_kernel.PNG)


- **Code** [Sklearn](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html)
```
from sklearn.svm import SVC
clf = SVC()
clf.fit(X, y)
```

## 4 软间隔与正则化
在现实任务中，很难确定是否存在一个合适的核函数来进行划分。我们需要融入一定程度的出错，这就是软间隔(soft margin)。

在sklearn中，可以调整参数C来控制。C越大，对误分类的惩罚增大，趋向于对训练集全分对的情况。反之则运行容错，泛化能力较强。

## 5 支持向量回归(SVR)
传统回归模型中，只有预测值与真值完全相等时损失才为零。而SVR中假设我们能容忍两者有一定误差e，仅当误差绝对值大于e时才计算损失。
相当于以f(x)为中心构建了一个宽度为2e的间隔带，只要在间隔带之间就被认为是预测正确。

SVR可表示为：

![f(x)=\sum_{i=1}^m (\widehat{\alpha}_i-\alpha_i) \kappa(x, x_i)+b ](https://render.githubusercontent.com/render/math?math=f(x)%3D%5Csum_%7Bi%3D1%7D%5Em%20(%5Cwidehat%7B%5Calpha%7D_i-%5Calpha_i)%20%5Ckappa(x%2C%20x_i)%2Bb%20)

- **Code** [Sklearn](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html)
```
from sklearn.svm import SVR
clf = SVC(C=1.0, epsilon=0.2)
clf.fit(X, y)
```

## 6 原理推导补充
SVM涉及的数学推导很多，限于笔者水平以及篇幅，这里偷懒不予列出。
但为感兴趣的小伙伴准备了一个近乎无敌的链接[CSDN](https://blog.csdn.net/v_july_v/article/details/7624837)

## 面试问题
1. 如何从逻辑回归优化函数推导出SVM的优化函数 How can the SVM optimization function be derived from the logistic regression optimization function?
2. 什么是大间隔分类器 What is a large margin classifier?
3. 为什么支持向量机是一个大间隔分类器的例子 Why SVM is an example of a large margin classifier?
4. 支持向量机作为一个大间隔分类器，会受异常值影响吗 SVM being a large margin classifier, is it influenced by outliers?
5. 支持向量机里面的软间隔C的作用 What is the role of C in SVM?
6. 支持向量机中，决策边界和theta的夹角是多少 In SVM, what is the angle between the decision boundary and theta?
7. 大间隔分类器的数学推导 What is the mathematical intuition of a large margin classifier?
8. 什么是核函数，为什么我们在支持向量机里使用核函数 What is a kernel in SVM? Why do we use kernels in SVM?
9. 逻辑回归与无核函数的支持向量机的区别 What is the difference between logistic regression and SVM without a kernel?
10. 支持向量机里的C是怎么影响偏差/方差权衡的 How does the SVM parameter C affect the bias/variance trade off?
11. 如何决定什么时候使用逻辑回归与什么时候使用SVM Logistic regression vs. SVMs: When to use which one?

## References
- [【机器学习】支持向量机SVM原理及推导](https://blog.csdn.net/u014433413/article/details/78427574)
- [面试前抢救一下--核函数与非线性SVM](https://juejin.im/post/5ad1c5f75188255cb07d8c33)
- [支持向量机通俗导论（理解SVM的三层境界）](https://blog.csdn.net/v_july_v/article/details/7624837)
