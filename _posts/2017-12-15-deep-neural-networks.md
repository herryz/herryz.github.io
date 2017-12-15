---
layout:     post
title:      "Deep Neural Networks"
subtitle:   "Coursera DeepLearning week4"
date:       2017-12-15
author:     "HerryZ"
header-img: "img/post-bg-unix-linux.jpg"
tags:
    - 深度学习
---

deeplearning.ai 是机器学习领域大牛Andrew Ng在Coursera上公布的新的深度学习的课程，相比之前机器学习的课程，本课程更偏重于深度学习的领域。

本文是课程一《Neural Networks and Deep Learning》的第四周笔记，上周给大家介绍了Shallow Neural Networks，本周我们将为大家介绍Deep Neural Networks。

注：如果没有任何机器学习基础，直接学习本课程可能会有些费力，建议大家先去学习Andrew Ng机器学习课程，传送门在此 [Coursera Machine Learning](https://www.coursera.org/learn/machine-learning)

### 基本概念
深层神经网络，顾名思义，就是有很多隐藏层的神经网络，所以其符号表示与神经网络是一样的，即使用中括号的上角标来表示第几层，比如$a^{[0]}$表示第0层，即输入层；$a^{[2]}$则表示第二层隐藏层。

下图即逻辑回归、浅层神经网络与深层神经网络的对比。
![image.png](http://upload-images.jianshu.io/upload_images/3913020-318584cf2a4e1e1b.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

由于其符号表示、前向传播、反向传播、参数及向量化已在前一章介绍过了，这里就不再赘述了，不清楚的可以回顾下上一周的笔记。
![image.png](http://upload-images.jianshu.io/upload_images/3913020-e985a3a60492c777.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

这里只是多了层数而已，完全套用前一章神经网络的前向传播和反向传播即可。
![image.png](http://upload-images.jianshu.io/upload_images/3913020-9ce467f0c9f2b59c.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

![image.png](http://upload-images.jianshu.io/upload_images/3913020-d148a91c384f0e7a.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

![image.png](http://upload-images.jianshu.io/upload_images/3913020-ec612091e67e0cfa.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

### 直观解释
下图为深层神经网络的直观解释，举了图像识别、语音识别等例子，详细讲解可以去看下视频。 [为什么使用深层表示.](http://mooc.study.163.com/learn/2001281002?tid=2001392029#/learn/content?type=detail&id=2001701022)
![image.png](http://upload-images.jianshu.io/upload_images/3913020-cb09edc727927516.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

### 参数与超参数
在神经网络中的参数指的是$W,b$，它们是在一次次梯度下降算法的迭代中不断优化的。
超参数指的是学习率、迭代数、隐藏层的层数和隐藏神经元的个数（神经网络的结构）及激活函数的选择等，超参数是需要我们在训练前手动设定的。
![image.png](http://upload-images.jianshu.io/upload_images/3913020-0d6448b565548b73.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

超参数的选择会决定了最终的参数$W,b$，还会影响模型的训练速度等，所以超参数的选择非常重要，在下一章中我们会讲解如何选择超参数。

### 本周内容回顾
通过本周学习，我们学习到了：
- 深层神经网络的基本概念
- 深层神经网络的前向传播和反向传播
- 参数与超参数的解释

最后贴出网易云课堂的链接，有兴趣的可以关注下我的知乎或博客，链接在最下方，可以交流下学习经验哈

- [网易云课堂-深度学习](http://mooc.study.163.com/smartSpec/detail/1001319001.htm)
- [Assignments](https://github.com/herryz/deeplearning_note)
