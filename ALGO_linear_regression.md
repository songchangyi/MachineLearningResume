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

## 3 逻辑回归（Logistic Regression，也叫逻辑斯蒂回归）
接下来我们看分类问题。神奇的线性回归模型产生的预测值是实数值，理论上可以从负无穷到正无穷。所以我们需要一个函数，来将实数值映射到0和1的范围内，然后再判断输出为0还是1。这里我们的sigmoid函数（中文的对数几率函数）就出场了：

![y=\frac{1}{1+e^{-z}} ](https://render.githubusercontent.com/render/math?math=y%3D%5Cfrac%7B1%7D%7B1%2Be%5E%7B-z%7D%7D%20)

从下图可以看出，它完美的符合我们的预期：

![Image of sigmoid](https://github.com/songchangyi/MachineLearningResume/blob/master/img/sigmoid.PNG)
