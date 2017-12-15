---
layout:     post
title:      "Logistic Regression"
subtitle:   "Coursera DeepLearning week2"
date:       2017-12-11
author:     "HerryZ"
header-img: "img/post-bg-js-version.jpg"
tags:
    - 深度学习
---


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
在神经网络中，forward propogation 用来计算输出，back propogation用来计算梯度，在得到梯度后就可以更新对应的参数。在此我们引入Computation Graph（计算图）来描述着两个过程，来看一个简单的例子 $ J(a, b, c)=3(a+bc) $

![image.png](http://upload-images.jianshu.io/upload_images/3913020-e73b69161c594d36.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)


如上图，通过前向传播，我们可以得到J=33。 反向传播本质上就是通过链式法则不断求出前面各个变量的导数的过程。在代码实现中，为了方便起见，我们将这些导数用dw1,db1等表示。
![image.png](http://upload-images.jianshu.io/upload_images/3913020-26f55304a07944fb.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

有了计算图的概念后，我们将其运用到Logistic Regression上。如下图，我们可以将途中的式子表示为下方的计算图。
![image.png](http://upload-images.jianshu.io/upload_images/3913020-ad92ceca96fc72d7.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

用上面的图，我们来计算反向传播。

首先我们来计算$ \frac{dL}{da} $：

$$ 
\begin{align} \frac{dL}{da} & = - (\frac{y}{a} - \frac{(1-y)}{(1-a)}) \end{align}
$$

通过链式法则，计算$ \frac{dL}{dz} $：

$$
\begin{align} \frac{dL}{dz} & = \frac{dL}{da}\frac{da}{dz} \\ \\ & = - (\frac{y}{a} - \frac{(1-y)}{(1-a)})\sigma(z)(1-\sigma(z)) \\ \\ & = - (\frac{y}{a} - \frac{(1-y)}{(1-a)})a(1-a)) \\ \\ & = -y(1-a) + (1-y)a \\ \\ & = a - y \end{align}
$$

最后计算$ \frac{dL}{dw1}, \frac{dL}{dw2}, \frac{dL}{db} $：

$$
\begin{align} \frac{dL}{dw_2} & = \frac{dL}{dz}\frac{dz}{dw_2} = (a - y)x_2  \\ \\ & \frac{dL}{dw_2} = \frac{dL}{dz}\frac{dz}{dw_2} = (a - y)x_2 \\ \\ &

\frac{dL}{db} = \frac{dL}{dz}\frac{dz}{db} = a - y \end{align}
$$

需要注意，上面我们说过Loss function是对单个样本偏差的计算，那么对于整个样本集，我们需要计算Cost function。

$$
J(w, b) = \frac{1}{m}(L(a^{(1)}, y^{(1)}) + L(a^{(2)}, y^{(2)}) + … + L(a^{(m)}, y^{m)}))
$$


下图即在m个样本上进行逻辑回归的流程，对于每一个样本都有一个对应的 $ dz^{(i)}$，而对于$dw, db$来说是对于所有样本求平均值。
![image.png](http://upload-images.jianshu.io/upload_images/3913020-55cebd5b7b4f9302.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

### 向量化
如果用遍历每一个样本的方式来实现梯度计算，更新参数的话，效率会非常地低，而向量化就是用来解决计算效率的问题。用python计算库numpy实现向量化非常简单。
如下代码：
```
import numpy as np
import time

a = np.random.rand(1000000)
b = np.random.rand(1000000)

tic = time.time()
c = np.dot(a, b)
toc = time.time()

print(c)
print('Vectorized version:{}ms'.format(1000*(toc-tic)))

c = 0
tic = time.time()
for i in range(1000000):
    c += a[i] * b[i]
toc = time.time()

print(c)
print('For loop:{}ms'.format(1000*(toc-tic)))
```

上面代码输出为：
```
250187.092541
Vectorized version:1.1589527130126953ms
250187.092541
For loop:424.5340824127197ms
```

两个版本计算结果相同，但时间效率上却差了近400倍，足以说明向量化计算的高效性。再加上神经网络结构复杂，而且训练样本的数量和大小都很大，效率显得尤为重要，所以我们应尽量避免使用for循环去遍历样本。

首先我们将$w$写成向量的形式，$dw=np.zeros((n_x, 1))$，这样就省去了内层关于$w$的循环
![image.png](http://upload-images.jianshu.io/upload_images/3913020-78737f1b066a6548.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

然后通过$W^T+b$得到z的向量化表达，a的向量化表达即是对z的每一个元素进行$\sigma$ 操作。在代码中，我们在写好向量化表达后最好使用numpy.shape()检查变量的维度是否和我们预想的一样

![image.png](http://upload-images.jianshu.io/upload_images/3913020-8e78cfb1a923c143.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

最后是梯度的向量化，通过前面的描述，我们知道了$A,Y$的向量化表示，前面我们还推导过dz的表示，所以如下图，我们写出dz的向量化表示$dZ$。而$db$是$dZ$元素的均值。
$X$的维度为$(n,m)$， $dZ$的维度为$(1,m)$， $dW$的维度和$W$的维度一样为$(n,1)$，这样我们便得到了$dW=\frac{1}{m}XdZ^T$

通过上面的描述，我们将之前的for循环版本改成了向量化的表示，这样代码的效率会大大提高。
![image.png](http://upload-images.jianshu.io/upload_images/3913020-936c9db6088d349a.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

### Python实现Logistic Regression
本周的编程作业是用Python+Numpy来实现一个完整的Logistic Regression，由于网易云课堂上没有相关作业的描述，所以在此我贴出作业连接
[Logistic Regression](https://github.com/herryz/deeplearning_note)

另外Jupyter Notebook的用法我会另外写一篇博客说明。

### 本周内容回顾
通过本周学习，我们学习到了：
- Logistic Regression的概念
- loss和cost的区别和联系
- 前向传播和反向传播
- 链式法则
- Numpy和Jupyter Notebook的基本用法

最后贴出网易云课堂的链接，有兴趣的可以关注下我的知乎或博客，链接在最下方，可以交流下学习经验哈

[网易云课堂-深度学习](http://mooc.study.163.com/smartSpec/detail/1001319001.htm)


