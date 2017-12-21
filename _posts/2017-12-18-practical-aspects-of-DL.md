---
layout:     post
title:      "Practical aspects of DL(1)"
subtitle:   "Coursera-Improving Deep Neural Networks-week1"
date:       2017-12-20
author:     "HerryZ"
header-img: "img/contact-bg.jpg"
tags:
    - 深度学习
---

deeplearning.ai 是机器学习领域大牛Andrew Ng在Coursera上公布的新的深度学习的课程，相比之前机器学习的课程，本课程更偏重于深度学习的领域。

本文是课程二《Improving Deep Neural Networks》的第一周笔记，本周主要内容包括：数据集分割，偏差与方差，正则化，梯度检查等，由于本周概念性的知识比较多，所以我分上下俩部分来写。

注：如果没有任何机器学习基础，直接学习本课程可能会有些费力，建议大家先去学习Andrew Ng机器学习课程，传送门在此 [Coursera Machine Learning](https://www.coursera.org/learn/machine-learning)

### 一、训练集/验证集/测试集
通常在训练完一个模型后，我们需要知道当前模型的训练效果如何，所以我们需要一个额外的数据集，我们常称为dev/hold out/validation set，在这里我们称之为验证集。如果我们还需要知道模型效果的无偏差估计，我们还需要test set即测试集。
在以往的机器学习中，通常将数据集按照70/30的比例划分为训练集和测试集，或按60/20/20的比例划分为训练集、验证集和测试集。但在如今的背景下，我们所用的数据集的量级非常大（比如百万级），所以我们可以不用给验证集和测试集那么大的比例，比如98/1/1即可。
![image.png](http://upload-images.jianshu.io/upload_images/3913020-61c1acb0b194e009.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

在这里我们需要注意的是，我们需要确保训练集、验证集和测试集来自同一个分布。例如我们需要去做一个识别猫的模型，那么我们就不能将网络上获取的所有猫图片作为训练集，将自己app上的所有猫图片作为验证和测试集，这是个典型的数据划分错误。
![image.png](http://upload-images.jianshu.io/upload_images/3913020-76687561885861f0.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

### 二、偏差与方差
偏差和方差用来描述模型的训练效果，下图是Andrew经典的图例解释：
![image.png](http://upload-images.jianshu.io/upload_images/3913020-5067c8fefd0f6513.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

根据上图，我们知道高偏差对应欠拟合，高方差对应过拟合。那么我们有什么方法知道当前模型所处的状态？如下图所示，一共有四种情况。
![image.png](http://upload-images.jianshu.io/upload_images/3913020-cef8aaa246409eba.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

- 当训练误差很小，验证误差和训练误差相差很大时为高方差(high variance)
- 当训练误差和验证误差都很大，但两者很接近时为高偏差(high bias)
- 当训练误差和验证误差都很大，且两者相差很大时为高方差高偏差(high variance & high bias)
- 当训练误差和验证误差都很小，且两者相差很小时为低方差低偏差(low varance & low bias)

当我们定位好模型所处的状态后，可以按照下图的方案来解决问题：
![image.png](http://upload-images.jianshu.io/upload_images/3913020-02f5327f977302b8.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

由上图我们可知，若模型为高方差，我们可以尝试使用更复杂的模型，比如使用更大的神经网络结构，或延长训练时间；若模型为高偏差，我们可以尝试使用更多的训练数据，或用接下来我们将讲的正则化来处理问题。

### 三、正则化
在神经网络中，正则化是用来降低过拟合的。通过上面的内容，我们知道减少过拟合的方法有：增加书训练集数量等，但在数据集有限的情况下，我们通常会使用降低模型复杂度的方法，比如：L2正则化、dropout等。

#### L2 Regularization
首先我们来看简单逻辑回归的L2正则化。如下图，即在损失函数后添加了正则化项，该项就是神经网络中的权重之和。
![image.png](http://upload-images.jianshu.io/upload_images/3913020-71fd05d237c51dc1.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

神经网络的L2正则化如下所示，也是添加了一个正则化项$ \frac{\lambda}{2m}\sum_{l=1}^L||w^{[l]}||^2_F$, 其中F代表Frobenius Norm。
在添加了正则项后，对应的梯度也会变化，所以在更新参数时需要加上对应的项。这里需要注意，根据经验我们只对参数w正则，不对参数b正则，因为对于每一层来说，w有很高的维度，而b只是一个标量，其对模型的影响度远没有w大。
![image.png](http://upload-images.jianshu.io/upload_images/3913020-7f3f7ecbb00c93ae.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

下图直观地显示了L2正则化是如何避免过拟合的。当$\lambda$比较大时，模型就会加大对参数w的惩罚，这样有些w就会变得很小，所以L2正则化也叫权重衰减(weight decay)，从下图左上角的神经网络看，效果就是整个神经网络的结构变简单了，从而降低了过拟合的风险。
![image.png](http://upload-images.jianshu.io/upload_images/3913020-1bf1126d59f4cc74.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

从另一个角度看，以tanh激活函数为例，当$\lambda$增加时，w会变小，这样$z = wa + b$也会变小，此时的激活函数如下图红色区域大致是线性的，这样模型的复杂度就降低了，从而降低了过拟合的风险。
![image.png](http://upload-images.jianshu.io/upload_images/3913020-25d865ff44b4c8c8.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

#### Dropout Regularization
Dropout也是一种正则化的手段，它指的是在神经网络的训练过程中，对于神经网络单元，按照一定的概率将其暂时从网络中丢弃，如下图所示。
![image.png](http://upload-images.jianshu.io/upload_images/3913020-0e356768939bd4b7.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

具体实现方式参考下图，在前向传播时将a中的某些值置为0，为了保证大概的大小不受添加的dropout的影响，再将处理后的a除以keep_porb。
![image.png](http://upload-images.jianshu.io/upload_images/3913020-ad8cb8725c4295cb.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

#### Other Regularization
1.Data augmentation
将数据集中的图片进行水平翻转，随机旋转裁剪等以变相增加数据集的数量。
![image.png](http://upload-images.jianshu.io/upload_images/3913020-ded8d373a3af6b60.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

2.Early stopping
![image.png](http://upload-images.jianshu.io/upload_images/3913020-d2cd134404a15404.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)


### 四、总结
由于本周的概念性内容过于多，所以本文主要讲了以下内容：
- 训练集、验证集和测试集
- 偏差与方差
- 正则化

下文我们将讲述归一化、梯度消失与梯度爆炸、梯度检查等内容。
最后贴出网易云课堂的链接，有兴趣的可以关注下我的知乎或博客，链接在最下方，可以交流下学习经验哈

- [网易云课堂-深度学习](http://mooc.study.163.com/smartSpec/detail/1001319001.htm)
- [Assignments](https://github.com/herryz/deeplearning_note)