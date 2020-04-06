# Evaluation

## 基本理论
### 1 评估方法
为了对学习器的泛化误差进行评估并进而做出选择，我们需要一个测试集(testing set)来测试模型对新样本的判别能力。然后以测试误差(testing error)作为泛化误差的近似。**要点**：测试样本是从样本真实分布中独立同分布(Independent and identically distributed, IID)采样得到，并与训练集互斥。
#### 1.1 留出法(hold-out)
直接将数据集划分为两个互斥的集合。使用分层采样(stratified sampling)。
**Cons**
- 单次估计结果不够稳定可靠。解决方法：若干次随机划分，取平均值。
- 测试集较小，降低了保真性(fidelity)。解决方法：无完美解决方案，常见做法是将2/3-4/5的样本用于训练，剩下用于测试。

## 面试问题

## References
1. 怎么理解 P 问题和 NP 问题 
[https://www.zhihu.com/question/27039635]
