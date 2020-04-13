# 模型评估指标 (Evaluation Metrics)
以下内容基于周志华老师的机器学习2.3节，2.5节归纳整理。

## 基本理论I：性能度量
为了对学习器的泛化性能进行评估，除了有效可行的实验估计方法，还需要有衡量模型泛化能力的评价标准，即性能度量(performance mesure)。
不同的性能度量往往会导致不同的评判结果，所以模型的好坏的相对的。应该根据算法，数据和任务需求来综合选择。

例如，回归任务最常用的性能度量是均方误差(mean squared error)：

<img src="https://render.githubusercontent.com/render/math?math=E = \frac{1}{m}  \sum_1^m (f(x_i)-y_i)^2">

- **Code** [Sklearn](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_squared_error.html)
```
from sklearn.metrics import mean_squared_error
mean_squared_error(y_true, y_pred)
```

### 1 错误率(error rate)与精度(accuracy)
错误率与精度是**分类任务**中最常用的两种性能度量，适用于二分类任务和多分类任务。分别是分类错误样本数和分类正确样本数占总数的比例。

- **错误率**：
<img src="https://render.githubusercontent.com/render/math?math=E = \frac{1}{m}  \sum_1^m (f(x_i) \neq y_i)">

- **精度** = 1 - E

- **Code** [Sklearn](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html)
```
from sklearn.metrics import accuracy_score
accuracy_score(y_true, y_pred)
```

### 2 查准率(precision)、查全率(recall)与F1
错误率和精度并不能满足所有需求。
我们考虑一个肿瘤检测的例子，100个肿瘤病人中99个都是良性，仅有1个是恶性。这时，如果我们简单粗暴的把所有人都判定为良性，我们的精度仍然高达99%。
但这样的结果不是我们希望的。我们更关心的指标可以是，我们检测出的恶性肿瘤有多少比例真的是恶性以及有多大比例的恶性肿瘤被检测了出来。
前者即对应查准率，追求每一次查出的恶性都尽量准确。后者对应查全率，力求不留余力地检测出所有的恶性。

为了便于解释，我们需要引入混淆矩阵(confusion matrix)的概念：

1. **混淆矩阵**

![Image of cm](https://github.com/songchangyi/MachineLearningResume/blob/master/img/cm.png)

- **Code** [Sklearn](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html)
```
from sklearn.metrics import confusion_matrix
confusion_matrix(y_true, y_pred)
```

2. **查准率P** = TP/(TP+FP)

- **Code** [Sklearn](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_score.html)
```
from sklearn.metrics import precision_score
precision_score(y_true, y_pred)
```

3. **查全率R** = TP/(TP+FN)

- **Code** [Sklearn](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.recall_score.html)
```
from sklearn.metrics import recall_score
recall_score(y_true, y_pred)
```

P和R通常是一对矛盾的度量。举一个极端的例子，为了找出所有得恶性肿瘤的病人，我们把所有病人全部归为恶性。这样我们确实没有遗漏，但这时的准确率仅为1%。
只有在一些简单的机器学习任务中，才可能两者都很高。

4. **PR曲线**：
PR曲线可以直观的展示出查准率和查全率的关系。绘制步骤如下：
    1. 将预测样本按照概率（或者置信度，可能性）进行排序。比如0.99, 0.98, 0.95, 0.65, 0.32, 0.01
    2. 从0到1或者从1到0变换阈值(threshold)。比如阈值为0.4，就划分为0.99, 0.98, 0.95, 0.65和0.32, 0.01两个集合。前一个都当作1处理，后一个都作为0
    3. 对于每个阈值，计算P和R

大概会得到如下曲线：

![Image of PR curve](https://github.com/songchangyi/MachineLearningResume/blob/master/img/P_R.png)

- **特点**：
  - 如果分类器A的曲线可以包住分类器B，则A优于B
  - 如果A和B的曲线交叉，则需要在具体条件下比较。曲线下的面积也不太容易估算，此时我们可以看平衡点(Break Event Point, BEP)，即P=R时的取值。
但是更常用的还是F1度量。

- **Code** [Sklearn](https://scikit-learn.org/stable/auto_examples/model_selection/plot_precision_recall.html)
```
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import plot_precision_recall_curve
import matplotlib.pyplot as plt

disp = plot_precision_recall_curve(classifier, X_test, y_test)
disp.ax_.set_title('2-class Precision-Recall curve: '
                   'AP={0:0.2f}'.format(average_precision))
```

5. **F1 score** = 2 * P * R/(P+R)

- **Code** [Sklearn](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html)
```
from sklearn.metrics import f1_score
f1_score(y_true, y_pred)
```

在多分类任务或者多次二分类任务时，我们会得到多个P和R。这时我们可以使用**宏F1(macro-F1)**和**微F1(micro-F1)**。
前者是对多组P和R直接取平均值，后者则是先计算TP，FP等的平均值再重新计算F1。

### 3 ROC与AUC
ROC全称是受试者工作特征(Receiver Operating Characteristic)曲线。与PR曲线类似，根据预测结果对测试样例进行排序，每次计算出2个重要量的值作为横纵坐标。
与PR曲线不同的是，ROC曲线的纵轴使用真正例率(True Positive Rate)，横轴使用假正例率(False Positive Rate)：

![Image of ROC curve](https://github.com/songchangyi/MachineLearningResume/blob/master/img/ROC.png)

- **TPR** = TP/(TP+FN)，即正确预测的正例占所有正例的比例。
- **FPR** = FP/(TN+FP)，即错误预测的正例占所有负例的比例。

举个例子，我们一开始设置阈值为1。就是说凡是预测分值小于1的结果我们都判定为负例。此时没有正例，自然TP和FP均为0。随着阈值的下降，我们开始预测出正例，当然其中有一部分是真正的正例，另外一部分其实是负例被错误预测了，所以TPR和FPR都开始增长。**注意**，分母分别是所有的正例和负例，也就是说分母是不会变的。这样一直到最后，我们设置阈值为0，判断所有样本均为正。这时候我们找出了所有真正的正样本，同时也把在负样本上犯的错误达到最大化。

形象的来讲，好的分类器的ROC曲线会接近于覆盖整个图。意味着从原点开始，当FPR每增加一点，TPR就会获得极大的增长。即降低阈值后预测的绝大部分正例都是真正的正例。

- **Code** [Sklearn](https://scikit-learn.org/stable/auto_examples/plot_roc_curve_visualization_api.html)
```
from sklearn.svm import SVC
from sklearn.metrics import plot_roc_curve
svc = SVC(random_state=42)
svc.fit(X_train, y_train)
svc_disp = plot_roc_curve(svc, X_test, y_test)
plt.show()
```

- **特点**：
  - 如果分类器A的曲线可以包住分类器B，则A优于B
  - 如果A和B的曲线交叉，则需要比较曲线下的面积AUC(Area Under ROC Curve)

AUC可以通过ROC下各部分的面积求和而得，估算为：

<img src="https://render.githubusercontent.com/render/math?math=AUC=\frac{1}{2} \sum_{i=1}^{m-1} (x_{i%2B1}-x_i) \cdot (y_i%2By_{i%2B1})">

- **Code** [Sklearn](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.auc.html)
```
from sklearn import metrics
fpr, tpr, thresholds = metrics.roc_curve(y, pred, pos_label=2)
metrics.auc(fpr, tpr)
```

## 基本理论II：偏差与方差 (bias & variance)
除了通过实验估计其泛化性能，人们往往还希望了解其原因。偏差-方差分解(bias-variance decomposition)是解释算法泛化性能的一种重要工具。
即将泛化误差分解为偏差、方差和噪声之和：

![E=bias^2(x)+var(x)+\varepsilon^2](https://render.githubusercontent.com/render/math?math=E%3Dbias%5E2(x)%2Bvar(x)%2B%5Cvarepsilon%5E2)

这里需要回顾一下偏差、方差和噪声的含义：
- 偏差：度量算法的期望预测与真实结果的偏离程度，即算法本身的拟合能力
- 方差：度量了同样大小的训练集变动所导致的学习性能的变化，即数据扰动所造成的影响
- 噪声：在当前任务上任何算法所能达到的期望泛化误差下界，即学习问题本身的难度

正如P与R，偏差和方差是有冲突的。学习器一开始训练不足的时候，拟合能力不够，数据扰动不会影响太大，则偏差主导泛化错误率。随着训练加深，数据扰动产生的影响越来越大，方差开始主导错误率。当模型拟合能力非常强的时候，就过拟合了。

> 总结一下:学习能力不行造成的误差是偏差，学习能力太强造成的误差是方差。[1]

## 面试问题
- 讲一下模型的评估指标 what are the evaluation metrics
- 讲一下混淆矩阵 what is a confusion matrix
- 查准率和查全率的定义及取舍 precision / recall definition and tradeoff
- 如何计算并解释ROC曲线和PR曲线中的AUC how to calculate and interpret AUC of ROC / PR curve
- 讲一下偏差与方差 Bias vs variance

## References
1. [偏差和方差有什么区别](https://www.zhihu.com/question/20448464)
2. [偏差（Bias）与方差（Variance）](https://zhuanlan.zhihu.com/p/38853908)
