# Evaluation Metrics
以下内容基于周志华老师的机器学习2.3节归纳整理。

## 基本理论
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
|  | 实际正例 | 实际反例 |
| --- | --- | --- |
| 预测为正 | 真正例(TP) | 假正例(FP) |
| 预测为反 | 假反例(FN) | 真反例(TN) |

- **Code** [Sklearn](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html)
```
from sklearn.metrics import confusion_matrix
confusion_matrix(y_true, y_pred)
```

- **查准率P**：
<img src="https://render.githubusercontent.com/render/math?math=P= \frac{TP}{TP\+FP}">

- **Code** [Sklearn](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_score.html)
```
from sklearn.metrics import precision_score
precision_score(y_true, y_pred)
```

- **查全率R**：
<img src="https://render.githubusercontent.com/render/math?math=R=TP/(TP+FN)">

- **Code** [Sklearn](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.recall_score.html)
```
from sklearn.metrics import recall_score
recall_score(y_true, y_pred)
```

P和R通常是一对矛盾的度量。举一个极端的例子，为了找出所有得恶性肿瘤的病人，我们把所有病人全部归为恶性。这样我们确实没有遗漏，但这时的准确率仅为1%。
只有在一些简单的机器学习任务中，才可能两者都很高。

- **PR曲线**：
PR曲线可以直观的展示出查准率和查全率的关系。绘制步骤如下：
1. 将预测样本按照概率（或者置信度，可能性）进行排序。比如0.99, 0.98, 0.95, 0.65, 0.32, 0.01
2. 从0到1或者从1到0变换阈值(threshold)。比如阈值为0.4，就划分为0.99, 0.98, 0.95, 0.65和0.32, 0.01两个集合。前一个都当作1处理，后一个都作为0
3. 对于每个阈值，计算P和R

大概会得到如下曲线：

![Image of PR curve](https://github.com/songchangyi/MachineLearningResume/blob/master/img/P_R.png)

- **Code** [Sklearn](https://scikit-learn.org/stable/auto_examples/model_selection/plot_precision_recall.html)
```
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import plot_precision_recall_curve
import matplotlib.pyplot as plt

disp = plot_precision_recall_curve(classifier, X_test, y_test)
disp.ax_.set_title('2-class Precision-Recall curve: '
                   'AP={0:0.2f}'.format(average_precision))
```

**特点**：
- 如果分类器A的曲线可以包住分类器B，则A优于B
- 如果A和B的曲线交叉，则需要在具体条件下比较。曲线下的面积也不太容易估算，此时我们可以看平衡点(Break Event Point, BEP)，即P=R时的取值。
但是更常用的还是F1度量。

- **F1 score**：
<img src="https://render.githubusercontent.com/render/math?math=F1=2*P*R/(P+R)">

- **Code** [Sklearn](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html)
```
from sklearn.metrics import f1_score
f1_score(y_true, y_pred)
```

在多分类任务或者多次二分类任务时，我们会得到多个P和R。这时我们可以使用**宏F1(macro-F1)**和**微F1(micro-F1)**。
前者是对多组P和R直接取平均值，后者则是先计算TP，FP等的平均值再重新计算F1。

### 3 ROC与AUC
