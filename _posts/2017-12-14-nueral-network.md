---
layout:     post
title:      "Shallow Neural Networks"
subtitle:   "Coursera DeepLearning week3"
date:       2017-12-14
author:     "HerryZ"
header-img: "img/post-bg-unix-linux.jpg"
tags:
    - 机器学习
---


deeplearning.ai 是机器学习领域大牛Andrew Ng在Coursera上公布的新的深度学习的课程，相比之前机器学习的课程，本课程更偏重于深度学习的领域。

本文是课程一《Neural Networks and Deep Learning》的第三周笔记，上周我们给大家介绍了Logistic Regression，从本周开始，我们将正式开始学习神经网络，我们先从只有一个隐藏层的简单神经网络开始学习。在学习完本周课程后，我们将使用python实现一个单隐藏层的神经网络。

注：如果没有任何机器学习基础，直接学习本课程可能会有些费力，建议大家先去学习Andrew Ng机器学习课程，传送门在此 [Coursera Machine Learning](https://www.coursera.org/learn/machine-learning)

### 基本概念
上周我们说过，逻辑回归是神经网络的基础，神经网络相比逻辑回归是添加了很多隐藏层。下图则是逻辑回归和神经网络的对比，在神经网络中，我们使用中括号的上角标来表示第几层，比如$a^{[0]}$表示第0层，即输入层；$a^{[1]}$则表示第一层隐藏层。
![image.png](http://upload-images.jianshu.io/upload_images/3913020-f0621739374688ee.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

在这里，我们同样使用计算图的概念来表示该神经网络，相比逻辑回归，我们多计算了一个a和z。并且我们同样使用前向传播和反向传播（梯度下降）来计算神经网络。

如下图就是神经网络的基本结构，它分为输入层，隐藏层和输出层，除了输入层和输出层的所有层都叫隐藏层。
![image.png](http://upload-images.jianshu.io/upload_images/3913020-ced3d3b9b01a33a1.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

### 神经网络的前向传播
如下图右上部分，我们先拿输入层和隐藏层的第一个神经元来看，其实这就是一个简单的逻辑回归，这样理解的话这个神经网络的隐藏层就是四个逻辑回归放在了一起。
这里我们需要注意下符号的表示，前面我们讲过中括号的上角标来表示第几层，那下角标则表示第几个神经元，比如$z1$为第一个神经元，a同理。
![image.png](http://upload-images.jianshu.io/upload_images/3913020-29026e9354235e37.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

下图则为隐藏层的矩阵表示。
![image.png](http://upload-images.jianshu.io/upload_images/3913020-4bc4a022d5b2e8eb.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

接下来我们来看隐藏层和输出层的连接，我们将隐藏层的a当做输入，发现了吗，这又是另一个逻辑回归。
![image.png](http://upload-images.jianshu.io/upload_images/3913020-d3d2d88408b63198.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

通过上述描述，我们总结一下各个变量的维度
- $w.shape : (n_L, n_{(L-1)})$
- $b.shape : (n_L, 1)$
- $z.shape : (n_L, 1)$
- $a.shape : (n_L, 1)$

注意我们前面讲的是针对一个样本，在逻辑回归时我们讲到了向量化，那么在这里我们同样需要对变量进行向量化，避免for循环，提高效率。
类比逻辑回归输入层x的向量化形式$X$，在神经网络里，$Z$就是将每个样本算出来的z横向叠加（$A$同理），具体如下图

![image.png](http://upload-images.jianshu.io/upload_images/3913020-873d725ea04a2cf5.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

### 神经网络中的激活函数
神经网络常用的激活函数有以下四种：Sigmoid, Tanh, ReLU, Leaky ReLU。其中Sigmoid函数我们在逻辑回归有介绍过，它的输出值在0~1之间，可以看成一个概率值，往往用在输出层，对于中间层来说，ReLU的效果最好。
![image.png](http://upload-images.jianshu.io/upload_images/3913020-3451635cddbee931.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

激活函数都是非线性函数，那么我们为什么需要非线性的激活函数呢？
目前已有证明，如果激活函数使用线性的激活函数，那么不论有多少隐藏层，最终都是线性的组合，相当于一个线性回归。
![image.png](http://upload-images.jianshu.io/upload_images/3913020-41fd401cd8caf3c4.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

在反向传播中，我们需要计算梯度，所以需要计算激活函数的导数，以下是四个激活函数的导数推导过程。
![image.png](http://upload-images.jianshu.io/upload_images/3913020-7b9a713469deeecf.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

![image.png](http://upload-images.jianshu.io/upload_images/3913020-10cbcbc31f311e3d.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

![image.png](http://upload-images.jianshu.io/upload_images/3913020-69c46aed5a87bb03.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

### 神经网络中的反向传播
通过上一周逻辑回归的反向传播过程，我们在这里同样使用计算图来计算神经网络中的各种梯度。
![image.png](http://upload-images.jianshu.io/upload_images/3913020-aa39d4dbec7c2213.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

下图左侧为各个梯度的计算结果，右侧则为m个训练样本上的向量化表达。
![image.png](http://upload-images.jianshu.io/upload_images/3913020-75f106293338ca02.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

### 神经网络中的参数初始化
在逻辑回归中，我们将参数$w$初始化为0，但如果我们在神经网络中还将$w$初始化为0，那么不管经过多少次前向传播和反向传播，隐藏层的所有节点都是相同的。所以我们应该随机地初始化$w$。
![image.png](http://upload-images.jianshu.io/upload_images/3913020-2efe6269184f5ebd.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

下图为具体初始化的代码实现，注意我们在初始化时乘了0.01，原因是如果我们使用了tanh作为我们的激活函数，若参数初始过大时，$z$也会比较大，此时梯度接近为0，梯度下降的速度就会非常的慢。
![image.png](http://upload-images.jianshu.io/upload_images/3913020-864a1bf7de3a0c0b.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

### 本周内容回顾
通过本周学习，我们学习到了：
- 神经网络的基本概念
- 神经网络的前向传播和反向传播
- 神经网络的激活函数
- 神经网络的参数初始化

最后贴出网易云课堂的链接，有兴趣的可以关注下我的知乎或博客，链接在最下方，可以交流下学习经验哈

[网易云课堂-深度学习](http://mooc.study.163.com/smartSpec/detail/1001319001.htm)