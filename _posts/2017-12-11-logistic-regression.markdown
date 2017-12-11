
deeplearning.ai 是机器学习领域大牛Andrew Ng在Coursera上公布的新的深度学习的课程，相比之前机器学习的课程，本课程更偏重于深度学习的领域。

本文是课程一《Neural Networks and Deep Learning》的第二周笔记，第一周主要是介绍了一些深度学习的背景知识，所以有兴趣的同学们可以直接去Coursera上看视频，另外网易云课堂目前也有此课程，知识没有课后测验与编程作业，但视频字幕做的很完善，大家也可以不用翻墙去学习。

注：如果没有任何机器学习基础，直接学习本课程可能会有些费力，建议大家先去学习Andrew Ng机器学习课程，传送门在此 [Coursera Machine Learning](https://www.coursera.org/learn/machine-learning)

### 基础概念
Logistic Regression即机器学习中的逻辑回归，它是神经网络的基础，有基础的同学们可以知道逻辑回归是在线性回归的基础上，添加了sigmoid激活函数，这里就不细讲线性回归了，有时间我会补充一份线性回归的笔记。逻辑回归可以看成是一种只有输入层和输出层的神经网络。

如下图，即逻辑回归的整体构架。

![image.png](http://upload-images.jianshu.io/upload_images/3913020-b27546d1282e310d.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

下图则表述了逻辑回归，这里需要注意两个概念。
Loss(error) function: 是对于单个样本预计值与真实值的偏差
Cost function: 是对于所有样本loss的平均值

![image.png](http://upload-images.jianshu.io/upload_images/3913020-1a856fd79fb049af.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

下图则是梯度下降。

![image.png](http://upload-images.jianshu.io/upload_images/3913020-c5d8cab12708d010.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)


如上知识点都是最基础的，如果不是很了解最好先去补补这块的知识。

### 计算图与前向反向传播
在神经网络中，forward propogation 用来计算输出，back propogation用来计算梯度，在得到梯度后就可以更新对应的参数。在此我们引入Computation Graph（计算图）来描述着两个过程，来看一个简单的例子

```math
J(a,b,c) = 3(a+bc)
```
![image.png](http://upload-images.jianshu.io/upload_images/3913020-e73b69161c594d36.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)


如上图，通过前向传播，我们可以得到J=33。 反向传播本质上就是通过链式法则不断求出前面各个变量的导数的过程。在代码实现中，为了方便起见，我们将这些导数用dw1,db1等表示。
![image.png](http://upload-images.jianshu.io/upload_images/3913020-26f55304a07944fb.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

有了计算图的概念后，我们将其运用到Logistic Regression上。如下图，我们可以将途中的式子表示为下方的计算图。

![image.png](http://upload-images.jianshu.io/upload_images/3913020-ad92ceca96fc72d7.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

用上面的图，我们来计算反向传播。
\begin{align} \frac{dL}{da} & = - (\frac{y}{a} - \frac{(1-y)}{(1-a)}) \end{align}

\left( \sum_{k=1}^n a_k b_k \right)^{\!\!2} 
\leq 
\left( \sum_{k=1}^n a_k^2 \right) 
\left( \sum_{k=1}^n b_k^2 \right)

前面我们说过，
![image.png](http://upload-images.jianshu.io/upload_images/3913020-55cebd5b7b4f9302.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

### 向量化

### python代码

而段内插入 LaTeX 公式是这样的： $ \{\,z\in C \mid z^2 = {\alpha}\,\} $，试试看看吧